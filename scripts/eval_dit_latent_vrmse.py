#!/usr/bin/env python3
"""
Full-validation-set latent-space reconstruction error for a trained DiT.

WHY THIS EXISTS (2026-08-31): the DiT's own training loss is a flow-matching
loss at random noise levels -- it does not directly report "how close is a
fully-denoised rollout to the true latent trajectory". scripts/eval_dit_vrmse.py
answers a related but different question in DECODED PIXEL SPACE (VAE-only
recon vs VAE+DiT rollout, both compared to the ground-truth physical fields
after a VAE decode) -- a bad number there conflates DiT rollout error with
whatever the VAE decoder itself adds/removes. This script instead compares
the DiT's rolled-out latent directly against the VAE ENCODER's ground-truth
latent for the same sim, with no decode step in between -- an isolated
measure of DiT forecasting quality in the space it actually operates in.

Same rollout as eval_dit_vrmse.py (condition on frame 0, DiT predicts the
rest) but pulled with `output_type="latent"` instead of `"pt"` so no VAE
decode happens on the predicted side. The SAME predicted latent is then
decoded here (reusing eval_dit_vrmse.py's vae_decode helper) purely so the
already-established pixel-space VRMSE can be reported side by side for
context -- this does not re-run the transformer, just one extra decode.

Usage:
    python scripts/eval_dit_latent_vrmse.py configs/dit/inference_dit_lrz_ai.yaml \\
        --preprocessed_data_root /path/to/dit_preprocessed/<vae_run_name> \\
        --checkpoint /path/to/checkpoints/model_weights_step_09360.safetensors \\
        --scalar_checkpoint /path/to/checkpoints/scalar_embedding_step_09360.safetensors \\
        --vae_checkpoint /path/to/finetune_vae_outputs/.../checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 675 \\
        --out_dir eval_dit_latent_vrmse_out/

See scripts/eval_dit_vrmse.py's own docstring for what each of these paths
means -- this script accepts the exact same arguments plus nothing new,
deliberately, so the two can be pointed at the same checkpoint pair.
"""

import argparse
import json
import sys
from pathlib import Path

# scripts/ has no __init__.py -- add it to sys.path so eval_dit_vrmse (a
# sibling script, not a package) can be imported below. Must happen before
# that import, not just under __main__, since the import runs at module
# load time.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.config import ScalarConditioningConfig
from windinet.inference.model_loader import select_vae_env
from windinet.losses import rmse_loss, vrms_loss, vrms_per_channel
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
    load_channel_normalization,
)
from windinet.utils import get_default_device

# Reused verbatim from scripts/eval_dit_vrmse.py -- see that script's own
# comment on why these are copied rather than imported (scripts/ has no
# __init__.py) -- imported here instead of re-copied since both scripts now
# live side by side and importing avoids the two drifting apart.
from eval_dit_vrmse import (  # noqa: E402
    DTYPE,
    build_initial_condition,
    load_scalar_embedding,
    load_transformer_weights,
    make_pipe,
    pad_to,
    vae_decode,
    vae_encode,
    verify_latent_space,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="Path to inference YAML config")
    ap.add_argument("--preprocessed_data_root", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--scalar_checkpoint", type=str, default=None)
    ap.add_argument("--vae_checkpoint", type=str, default=None)
    ap.add_argument("--normalization", type=str, default=None)
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--default_temb", type=float, default=0.0)
    ap.add_argument("--out_dir", type=str, default="outputs/eval_dit_latent_vrmse")
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "scalar_checkpoint", "vae_checkpoint", "normalization",
                "num_inference_steps", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    return cfg


def trim_latent_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, C, F, H, W] pair -> both trimmed to the shorter F. The DiT's
    rollout and the VAE's direct encode of the (possibly differently
    zero-padded) ground-truth video are not guaranteed to land on the exact
    same latent frame count -- see eval_dit_vrmse.py's pad_to() for the
    pixel-space equivalent of this same alignment issue."""
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)
    device = get_default_device()

    preprocessed_data_root = Path(args.preprocessed_data_root)
    manifest_path = preprocessed_data_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    h5_path = manifest["data_root"]
    val_ids = manifest["val_ids"]
    print(f"Held-out split: {len(val_ids)} sims (seed={manifest['split_seed']}) from {manifest_path}")
    print(f"h5: {h5_path}")

    if args.num_samples is not None:
        val_ids = val_ids[: args.num_samples]
    print(f"Evaluating {len(val_ids)} sim(s)")

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_frames = cfg.get("num_frames", 105)
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg.get("vae_checkpoint")
    if not vae_ckpt:
        raise SystemExit("vae_checkpoint is required (inflate-mode finetuned checkpoint)")
    select_vae_env(ensure_checkpoint(vae_ckpt))

    stats = load_channel_normalization(cfg["normalization"])
    print(f"Normalization from {cfg['normalization']}: clip={stats['normalization_clip']}")
    verify_latent_space(ensure_checkpoint(vae_ckpt), checkpoint, stats)

    sc = cfg.get("scalar_conditioning", {})
    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=sc.get("scalar_names", ["gamma"]),
        scalar_ranges={k: tuple(v) for k, v in sc.get("scalar_ranges", {"gamma": [1.0, 2.0]}).items()},
        embedding_dim=sc.get("embedding_dim", 4096),
        num_tokens_per_scalar=sc.get("num_tokens_per_scalar", 4),
    )

    dataset = ShockWaveDataset(h5_path)
    missing = [s for s in val_ids if s not in dataset.ids]
    if missing:
        raise SystemExit(f"val_ids not found in {h5_path}: {missing[:5]}...")

    pipe = make_pipe(model_source, device)
    load_transformer_weights(pipe, checkpoint)
    scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    sum_lat_vrmse, sum_lat_rmse, sum_pixel_vrmse = 0.0, 0.0, 0.0
    n_lat_channels = None
    sum_lat_channel = None

    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample["density"].shape[-2:]
        orig_F = sample["density"].shape[0]

        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
        )
        target_pixel = pad_to(gt_video, orig_F).float()

        # Ground-truth latent trajectory -- the VAE encoder's own output,
        # zero DiT involvement. This is the "correct answer" the DiT's
        # rollout below is compared against.
        gt_latent = vae_encode(pipe.vae, gt_video.to(DTYPE)).float()

        # --- DiT rollout, latent output (no decode) ---
        cond_video = build_initial_condition(sample, stats, device)
        scalar_values = [sample["meta"][n] for n in scalar_cfg.scalar_names]
        scalars = torch.tensor([scalar_values], device=device, dtype=DTYPE)
        prompt_embeds = scalar_emb(scalars)
        prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

        g = torch.Generator(device=device).manual_seed(seed + i)
        with autocast(device.type, dtype=DTYPE, enabled=(device.type in ("cuda", "xpu"))):
            out = pipe(
                prompt=None, negative_prompt=None,
                video=cond_video, frame_index=0, strength=1.0,
                width=W, height=H,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                image_cond_noise_scale=image_cond_noise_scale,
                generator=g,
                output_reference_comparison=False,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                negative_prompt_attention_mask=prompt_mask.clone(),
                output_type="latent",
            )
        # out.frames is the raw predicted latent, [B, C, F_lat, H_lat, W_lat],
        # in the SAME rescaled space vae_encode() above returns (see
        # LTXConditionPipeline's output_type=="latent" branch: it returns the
        # unpacked latents BEFORE _denormalize_latents, which is exactly the
        # inverse of vae_encode's own rescale) -- directly comparable to
        # gt_latent with no further conversion.
        dit_pred_latent = out.frames.float()

        pred_lat, gt_lat = trim_latent_frames(dit_pred_latent, gt_latent)

        lat_vrmse = float(vrms_loss(pred_lat, gt_lat).item())
        lat_rmse = float(rmse_loss(pred_lat, gt_lat).item())
        lat_vrmse_channel = vrms_per_channel(pred_lat, gt_lat)

        if sum_lat_channel is None:
            n_lat_channels = lat_vrmse_channel.shape[0]
            sum_lat_channel = torch.zeros(n_lat_channels)
        sum_lat_channel += lat_vrmse_channel.cpu()

        # --- Same predicted latent, decoded -- pixel-space VRMSE for context,
        # directly comparable to eval_dit_vrmse.py's "vae+dit" number. One
        # extra decode, zero extra transformer forward passes. ---
        dit_pred_pixel = vae_decode(pipe.vae, pred_lat.to(DTYPE), args.default_temb).float()
        dit_pred_pixel = pad_to(dit_pred_pixel, orig_F)
        n_px = min(dit_pred_pixel.shape[2], target_pixel.shape[2])
        pixel_vrmse = float(vrms_loss(dit_pred_pixel[:, :, :n_px], target_pixel[:, :, :n_px]).item())

        per_sample.append({
            "id": sid, "gamma": float(sample["meta"]["gamma"]),
            "latent_vrmse": lat_vrmse, "latent_rmse": lat_rmse, "pixel_vrmse": pixel_vrmse,
        })
        sum_lat_vrmse += lat_vrmse
        sum_lat_rmse += lat_rmse
        sum_pixel_vrmse += pixel_vrmse

        print(f"[{i+1}/{len(val_ids)}] {sid}: latent_vrmse={lat_vrmse:.5f}  "
              f"latent_rmse={lat_rmse:.5f}  pixel_vrmse(decoded)={pixel_vrmse:.5f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    n = len(val_ids)
    mean_lat_channel = (sum_lat_channel / n).tolist()
    ranked = sorted(range(n_lat_channels), key=lambda c: mean_lat_channel[c], reverse=True)
    summary = {
        "n_samples": n,
        "h5": h5_path,
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_ckpt),
        "latent_vrmse_mean": sum_lat_vrmse / n,
        "latent_rmse_mean": sum_lat_rmse / n,
        "pixel_vrmse_mean_decoded_from_same_latent": sum_pixel_vrmse / n,
        "latent_vrmse_per_channel": mean_lat_channel,
        "worst_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[:5]],
        "best_5_latent_channels": [{"channel": c, "vrmse": mean_lat_channel[c]} for c in ranked[-5:]],
        "per_sample": per_sample,
    }
    (out_dir / "latent_vrmse_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"n={n} sims")
    print(f"Latent-space VRMSE (DiT rollout vs VAE-encoder ground truth): {summary['latent_vrmse_mean']:.5f}")
    print(f"Latent-space RMSE  (same comparison, unnormalized)          : {summary['latent_rmse_mean']:.5f}")
    print(f"Pixel-space  VRMSE (same predicted latent, decoded)         : {summary['pixel_vrmse_mean_decoded_from_same_latent']:.5f}")
    print(f"Worst 5 latent channels (by VRMSE): {summary['worst_5_latent_channels']}")
    print(f"\nSaved: {out_dir / 'latent_vrmse_summary.json'}")


if __name__ == "__main__":
    main()
