#!/usr/bin/env python3
"""
Diagnose WHERE a DiT rollout's pixel-space error comes from, frame by frame.

Motivation: lundquist's `sds_untrained` arm has ~10x better latent_rmse than
the plain `kl0` baseline (confirmed real, not a normalization artifact --
see scripts/diagnose_latent_scale.py and EXPERIMENTS.md's "Why does a real
~10x latent-space accuracy win only produce a ~7% pixel-space win" open
question), yet its decoded pixel_vrmse is only ~7% better (0.393 vs 0.422,
eval_dit_vrmse.py jobs 22776/22764). Per-CHANNEL breakdown (already in
eval_dit_vrmse.py's vrmse_summary.json) shows the ~7% gain is spread evenly
across density/momentum_x/momentum_y/pressure, not hiding in one channel --
so the discrepancy isn't a channel-specific effect. This script checks the
other axis: per-FRAME (time), which eval_dit_vrmse.py does not report.

Two candidate explanations, each with a different frame-index signature:
  (a) Decoder sensitivity: the VAE decoder is locally high-Lipschitz around
      the data manifold (small changes in the physical field -- e.g. once a
      shock has formed and gradients are steep -- amplify a fixed latent
      error into a bigger pixel error more than a smooth early frame would)
      -- predicts pixel error growing over frames even if the DiT's OWN
      latent-space accuracy (pred_latent vs gt_latent) stays roughly flat.
  (b) Compounding rollout drift: the DiT's own latent predictions get less
      accurate later in the rollout (standard autoregressive-ish error
      accumulation in generative video models, on top of shockwave/
      compressible-flow dynamics being sensitive to small perturbations
      once a shock has developed) -- predicts BOTH latent-space accuracy
      AND pixel error degrading over frames together.

Reports three per-frame curves per sample (vrms_loss's own formula, reduced
over channels+H+W but keeping the frame axis, then averaged over samples):
  - vae_only_per_frame: decode(gt_latent) vs actual ground truth -- pure
    VAE reconstruction floor, should be roughly flat over frames (no DiT
    rollout involved) if the decoder's own fidelity doesn't depend on
    which frame it's decoding.
  - vae_dit_per_frame: decode(dit_rollout_latent) vs actual ground truth --
    the full end-to-end error whose average IS eval_dit_vrmse.py's
    vae_dit_vrmse_mean.
  - latent_per_frame: dit_rollout_latent vs gt_latent, no decode -- the
    DiT's own forecast accuracy in isolation, matches
    diagnose_latent_scale.py's per-sample number but broken out by frame.

Usage: same checkpoint/config args as eval_dit_vrmse.py / diagnose_latent_scale.py.
    python scripts/diagnose_frame_drift.py configs/dit/inference_dit.yaml \\
        --preprocessed_data_root dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \\
        --checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \\
        --scalar_checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/scalar_embedding_step_08033.safetensors \\
        --vae_checkpoint finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 20 --out_dir diag_out/sds_untrained_frame_drift

--num_samples defaults to 20, same diagnostic-not-final-eval reasoning as
diagnose_latent_scale.py.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.config import ScalarConditioningConfig
from windinet.inference.model_loader import load_ltxv_components, select_vae_env
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    CHANNEL_NAMES, ShockWaveDataset, build_shockwave_video, load_channel_normalization, normalize_fields,
)
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------
# Copied verbatim from scripts/eval_dit_vrmse.py / scripts/diagnose_latent_scale.py
# -- scripts/ has no __init__.py, so these small already-tested helpers are
# duplicated rather than imported. Any change there should be mirrored here.
# ----------------------------------------------------------------------

def verify_latent_space(vae_checkpoint, dit_checkpoint, stats) -> None:
    provenance_path = Path(dit_checkpoint).parent.parent / "latent_provenance.json"
    if not provenance_path.is_file():
        print(f"WARNING: no {provenance_path} -- cannot verify VAE/DiT latent-space match.")
        return
    expected = json.loads(provenance_path.read_text()).get("latent_fingerprint")
    if not expected:
        return
    actual = latent_space_fingerprint(
        Path(vae_checkpoint), stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
    )
    if actual != expected:
        raise SystemExit(f"VAE / DiT latent space mismatch: DiT trained on {expected}, this VAE gives {actual}.")
    print(f"Latent space verified: fingerprint {actual} matches the DiT's training latents")


def make_pipe(model_source, device):
    c = load_ltxv_components(model_source=model_source, transformer_dtype=DTYPE, vae_dtype=DTYPE)
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler), vae=c.vae, text_encoder=None, tokenizer=None, transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.video_processor.register_to_config(do_normalize=False)
    return pipe


def load_transformer_weights(pipe, checkpoint):
    sd = load_file(str(checkpoint))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    pipe.transformer.load_state_dict(sd, strict=False)


def load_scalar_embedding(checkpoint, scalar_cfg, device):
    emb = ScalarEmbedding(scalar_cfg)
    emb.load_state_dict(load_file(str(checkpoint)))
    return emb.to(device=device, dtype=DTYPE).eval()


def build_initial_condition(sample, stats, device):
    fields = torch.stack([sample[name][0] for name in CHANNEL_NAMES]).unsqueeze(0)
    fields = normalize_fields(fields, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"])
    return fields.unsqueeze(1).to(device=device, dtype=DTYPE)


def vae_encode(vae, video: torch.Tensor) -> torch.Tensor:
    out = vae.encode(video)
    posterior_mean = out.latent_dist.mean
    norm_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    norm_std = vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    return (posterior_mean - norm_mean) * sf / norm_std


def vae_decode(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


def trim_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


def vrms_per_frame(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same VRMSE formula as windinet.losses.vrms.vrms_loss, but reduced
    over channels+H+W only (dims 1, 3, 4), keeping the frame axis (dim 2)
    distinct -- lets "does error grow with frame index" be read off
    directly instead of only ever seeing the all-frames-collapsed scalar.
    [B, C, T, H, W] -> [T]."""
    dims = (1, 3, 4)
    diff = pred - target
    mse = diff.square().mean(dim=dims)
    variance = target.var(dim=dims, unbiased=False)
    return torch.sqrt(mse / (variance + eps)).mean(dim=0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--preprocessed_data_root", type=Path, required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--scalar_checkpoint", type=Path, default=None)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, default=None)
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--default_temb", type=float, default=0.0)
    ap.add_argument("--out_dir", type=Path, default=Path("diag_out"))
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("checkpoint", "scalar_checkpoint", "vae_checkpoint", "normalization",
                "num_inference_steps", "guidance_scale"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    return cfg


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)
    device = get_default_device()

    manifest = json.loads((args.preprocessed_data_root / "split_manifest.json").read_text())
    h5_path = manifest["data_root"]
    val_ids = manifest["val_ids"][: args.num_samples]
    print(f"Evaluating {len(val_ids)} held-out sim(s) from {h5_path}")

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_frames = cfg.get("num_frames", 105)
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg["vae_checkpoint"]
    select_vae_env(ensure_checkpoint(vae_ckpt))
    stats = load_channel_normalization(cfg["normalization"])
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
    pipe = make_pipe(model_source, device)
    load_transformer_weights(pipe, checkpoint)
    scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    sum_vae_only, sum_vae_dit, sum_latent, n_frames_seen = None, None, None, 0
    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample["density"].shape[-2:]
        orig_F = sample["density"].shape[0]

        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
        )
        target = gt_video[:, :, :orig_F].float()
        gt_latent = vae_encode(pipe.vae, gt_video.to(DTYPE))
        vae_recon = vae_decode(pipe.vae, gt_latent, args.default_temb).float()[:, :, :orig_F]

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
        pred_lat, cmp_gt_lat = trim_frames(out.frames.float(), gt_latent.float())
        dit_pred = vae_decode(pipe.vae, pred_lat.to(DTYPE), args.default_temb).float()

        n_px = min(dit_pred.shape[2], target.shape[2], vae_recon.shape[2])
        vo_frame = vrms_per_frame(vae_recon[:, :, :n_px], target[:, :, :n_px]).cpu()
        vd_frame = vrms_per_frame(dit_pred[:, :, :n_px], target[:, :, :n_px]).cpu()
        n_lat = min(pred_lat.shape[2], cmp_gt_lat.shape[2])
        lat_frame = vrms_per_frame(pred_lat[:, :, :n_lat], cmp_gt_lat[:, :, :n_lat]).cpu()

        if sum_vae_only is None:
            sum_vae_only, sum_vae_dit, sum_latent = (
                torch.zeros(n_px), torch.zeros(n_px), torch.zeros(n_lat),
            )
        sum_vae_only[: vo_frame.shape[0]] += vo_frame
        sum_vae_dit[: vd_frame.shape[0]] += vd_frame
        sum_latent[: lat_frame.shape[0]] += lat_frame
        n_frames_seen += 1
        print(f"[{i+1}/{len(val_ids)}] {sid}: vae_only[0]={vo_frame[0]:.4f} vae_only[-1]={vo_frame[-1]:.4f}  "
              f"vae_dit[0]={vd_frame[0]:.4f} vae_dit[-1]={vd_frame[-1]:.4f}  "
              f"latent[0]={lat_frame[0]:.4f} latent[-1]={lat_frame[-1]:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = n_frames_seen
    summary = {
        "n_samples": n,
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_ckpt),
        "vae_only_per_frame": (sum_vae_only / n).tolist(),
        "vae_dit_per_frame": (sum_vae_dit / n).tolist(),
        "latent_per_frame": (sum_latent / n).tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "frame_drift.json").write_text(json.dumps(summary, indent=2))

    def _trend(xs: list[float]) -> str:
        first_q = xs[: max(1, len(xs) // 4)]
        last_q = xs[-max(1, len(xs) // 4):]
        f, l = sum(first_q) / len(first_q), sum(last_q) / len(last_q)
        return f"first-quarter mean={f:.4f}  last-quarter mean={l:.4f}  ratio={l/f:.2f}x"

    print("\n" + "=" * 70)
    print(f"n={n} sample(s), {len(summary['vae_dit_per_frame'])} frames")
    print(f"vae_only_per_frame trend (pure decode, no DiT): {_trend(summary['vae_only_per_frame'])}")
    print(f"vae_dit_per_frame  trend (full rollout+decode): {_trend(summary['vae_dit_per_frame'])}")
    print(f"latent_per_frame   trend (DiT forecast, no decode): {_trend(summary['latent_per_frame'])}")
    print("\nInterpretation: if latent_per_frame's ratio is ~1 (flat) but vae_dit_per_frame's")
    print("ratio is >> 1 (grows), that points at decoder sensitivity (mechanism (a) in this")
    print("script's docstring). If BOTH ratios grow together, that points at compounding")
    print("rollout drift in the DiT itself (mechanism (b)).")
    print(f"\nSaved: {args.out_dir / 'frame_drift.json'}")


if __name__ == "__main__":
    main()
