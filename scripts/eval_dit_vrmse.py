#!/usr/bin/env python3
"""
Full-validation-set VRMSE: VAE-only reconstruction vs VAE+DiT rollout.

Answers "how much extra error does the DiT add on top of the VAE's own
reconstruction floor" -- using the SAME held-out sims the VAE was validated
on (read from <preprocessed_data_root>/split_manifest.json, written by
preprocess_dataset.py -- see that script's own comment: recorded so the val
set can be "reproduced ... without re-deriving the permutation") and the
SAME variance-normalized-RMSE formula/space VaeTrainer's own val_vrmse uses
(windinet.losses.vrms, computed on NORMALIZED fields, matching
vae_trainer.py's _validate -- NOT denormalized physical units).

Two passes per sample, same ground truth, same metric:
  - vae_only:  encode the full ground-truth sequence with the VAE, decode,
               compare -- pure autoencoding fidelity, no forecasting.
  - vae_dit:   encode only frame 0 (the initial condition), let the DiT
               roll the rest out in latent space, decode, compare -- the
               full inference-time pipeline inference_shockwave.py runs.

The DiT checkpoint can be a mid-training checkpoint (this run may not have
finished) -- this script does not require or check a "final" checkpoint.

Usage:
    python scripts/eval_dit_vrmse.py configs/dit/inference_dit_lrz_ai.yaml \\
        --preprocessed_data_root finetune_vae_outputs/sng_pvc/dit_preprocessed/finetune_vae_whole_structure_baseline_256res \\
        --checkpoint /path/to/checkpoints/model_weights_step_09177.safetensors \\
        --scalar_checkpoint /path/to/checkpoints/model_weights_step_09177.state.pt \\
        --vae_checkpoint /path/to/finetune_vae_outputs/.../checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 675 \\
        --save_npz_samples 3 \\
        --out_dir eval_dit_vrmse_out/

--num_samples defaults to all val_ids in the manifest (675 for the 256res
runs) -- pass a smaller number for a quick check, since each sample costs
one VAE encode + one DiT rollout + two VAE decodes.

--save_vis_samples N additionally renders GT/Prediction/Residual panels, at
--frame_numbers (default 25/50/75/100, matching the VAE/DiT trainers' own
periodic-visualization convention -- windinet.training.vae_visualization
.save_reconstruction_panels), for the first N samples, for BOTH passes --
vae_only and vae_dit -- under <out_dir>/visualizations/{vae_only,vae_dit}/
<sample_id>/frame_XXXX.png -- so the before/after gap from adding the DiT
can be read off visually, same layout the training-time panels use.
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
from windinet.losses import vrms_loss, vrms_per_channel
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    CHANNEL_NAMES,
    ShockWaveDataset,
    build_shockwave_video,
    load_channel_normalization,
    normalize_fields,
)
from windinet.training.vae_visualization import denormalize_fields, save_reconstruction_panels
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------
# Reused verbatim from scripts/inference_shockwave.py (that script is not
# an importable package -- scripts/ has no __init__.py -- so the small,
# already-tested pipeline-setup functions are copied rather than imported;
# any change there should be mirrored here).
# ----------------------------------------------------------------------

def verify_latent_space(vae_checkpoint, dit_checkpoint, stats) -> None:
    provenance_path = Path(dit_checkpoint).parent.parent / "latent_provenance.json"
    if not provenance_path.is_file():
        print(
            f"WARNING: no {provenance_path} -- cannot verify that {vae_checkpoint} is the VAE "
            "the DiT's latents were built with. A mismatch here produces wrong physics silently."
        )
        return
    expected = json.loads(provenance_path.read_text()).get("latent_fingerprint")
    if not expected:
        print(f"WARNING: {provenance_path} records no latent_fingerprint; skipping the check.")
        return
    actual = latent_space_fingerprint(
        Path(vae_checkpoint), stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
    )
    if actual != expected:
        raise SystemExit(
            f"VAE / DiT latent space mismatch.\n"
            f"  DiT was trained on latents with fingerprint : {expected}\n"
            f"  this VAE + normalization fingerprints as    : {actual}\n"
            f"  vae_checkpoint : {vae_checkpoint}\n"
            f"  provenance     : {provenance_path}\n"
            f"Point vae_checkpoint at the VAE recorded in {provenance_path}."
        )
    print(f"Latent space verified: fingerprint {actual} matches the DiT's training latents")


def make_pipe(model_source, device):
    c = load_ltxv_components(model_source=model_source, transformer_dtype=DTYPE, vae_dtype=DTYPE)
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler), vae=c.vae, text_encoder=None, tokenizer=None, transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    # Our fields are already in [-1, 1] via the CFD channel stats -- see
    # inference_shockwave.py's own comment on why do_normalize is disabled.
    pipe.video_processor.register_to_config(do_normalize=False)
    return pipe


def load_transformer_weights(pipe, checkpoint):
    print(f"Loading transformer: {checkpoint}")
    sd = load_file(str(checkpoint))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    pipe.transformer.load_state_dict(sd, strict=False)


def load_scalar_embedding(checkpoint, scalar_cfg, device):
    print(f"Loading scalar embedding: {checkpoint}")
    emb = ScalarEmbedding(scalar_cfg)
    emb.load_state_dict(load_file(str(checkpoint)))
    return emb.to(device=device, dtype=DTYPE).eval()


def build_initial_condition(sample, stats, device):
    fields = torch.stack([sample[name][0] for name in CHANNEL_NAMES]).unsqueeze(0)  # [1, 4, H, W]
    fields = normalize_fields(fields, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"])
    return fields.unsqueeze(1).to(device=device, dtype=DTYPE)  # [B, F, C, H, W]


# ----------------------------------------------------------------------
# VAE-only encode/decode, copied from VaeTrainer._encode/_decode
# (windinet/training/vae_trainer.py) so the "VAE-only" pass matches the
# exact rescaling (latents_mean/std, scaling_factor, default_temb) that
# produced the already-committed val_vrmse numbers -- reimplementing this
# math independently would risk a silent mismatch against those numbers.
# ----------------------------------------------------------------------

def vae_encode(vae, video: torch.Tensor) -> torch.Tensor:
    """video: [B, C, F, H, W], normalized. Returns rescaled latents."""
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


def pad_to(t: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Trim a [B,C,F,H,W] tensor's frame dim to n_frames (both passes may
    return a few more frames than the ground truth due to 8k+1 padding)."""
    return t[:, :, :n_frames]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--preprocessed_data_root", type=Path, required=True,
                     help="Same preprocessed_data_root the DiT training config used -- gives the "
                          "h5 path and the exact held-out val_ids via split_manifest.json")
    ap.add_argument("--num_samples", type=int, default=None,
                     help="How many val sims to evaluate (default: all in the manifest)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Override transformer checkpoint")
    ap.add_argument("--scalar_checkpoint", type=Path, default=None, help="Override scalar embedding checkpoint")
    ap.add_argument("--vae_checkpoint", type=Path, default=None, help="Override inflate-mode VAE checkpoint")
    ap.add_argument("--normalization", type=Path, default=None,
                     help="Override normalization source (a training_config.yaml or stats yaml) -- "
                          "point this at the VAE run's own training_config.yaml for that run's exact stats")
    ap.add_argument("--num_inference_steps", type=int, default=None, help="Override denoising steps")
    ap.add_argument("--guidance_scale", type=float, default=None, help="Override guidance scale")
    ap.add_argument("--default_temb", type=float, default=0.0,
                     help="temb passed to the VAE-only decode pass (matches adapter.default_temb "
                          "in every VAE training config seen in this repo so far)")
    ap.add_argument("--save_vis_samples", type=int, default=3,
                     help="Render GT/Prediction/Residual panels (both passes) for the first N samples")
    ap.add_argument("--frame_numbers", type=int, nargs="+", default=[25, 50, 75, 100],
                     help="1-indexed frame numbers to render panels for (default matches "
                          "DitVisualizationConfig's own default)")
    ap.add_argument("--dpi", type=int, default=150, help="Panel image DPI")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/eval_dit_vrmse"))
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

    manifest_path = args.preprocessed_data_root / "split_manifest.json"
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
    num_output_frames = cfg.get("num_output_frames", 101)
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg.get("vae_checkpoint")
    if not vae_ckpt:
        raise SystemExit("vae_checkpoint is required (inflate-mode finetuned checkpoint)")
    select_vae_env(ensure_checkpoint(vae_ckpt))  # must run BEFORE make_pipe: sets which VAE it loads

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

    pipe = make_pipe(model_source, device)  # pipe.vae is now the finetuned VAE from vae_ckpt
    load_transformer_weights(pipe, checkpoint)
    scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = []
    sum_overall = {"vae_only": 0.0, "vae_dit": 0.0}
    sum_channel = {"vae_only": [0.0] * 4, "vae_dit": [0.0] * 4}

    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample["density"].shape[-2:]
        orig_F = sample["density"].shape[0]

        # Ground truth, normalized, [1, 4, orig_F, H, W] -- same call
        # preprocess_dataset.py used to build the VAE's own training/eval input.
        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
        )
        target = pad_to(gt_video, orig_F).float()

        # --- VAE-only pass: encode + decode the full ground-truth sequence ---
        latents = vae_encode(pipe.vae, gt_video.to(DTYPE))
        vae_recon = vae_decode(pipe.vae, latents, args.default_temb).float()
        vae_recon = pad_to(vae_recon, orig_F)

        vo_overall = float(vrms_loss(vae_recon, target).item())
        vo_channel = vrms_per_channel(vae_recon, target).tolist()

        # --- VAE+DiT pass: condition on frame 0 only, roll the rest out ---
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
                output_type="pt",
            )
        # out.frames: [B, F, C, H, W] in the SAME normalized [-1,1] space as
        # `target` (do_normalize=False on the video processor) -- do not
        # denormalize here, val_vrmse is a normalized-space metric.
        dit_pred = out.frames[0].float().cpu()[:num_output_frames].permute(1, 0, 2, 3).unsqueeze(0)
        dit_pred = pad_to(dit_pred.to(target.device), orig_F)

        vd_overall = float(vrms_loss(dit_pred, target).item())
        vd_channel = vrms_per_channel(dit_pred, target).tolist()

        per_sample.append({
            "id": sid, "gamma": float(sample["meta"]["gamma"]),
            "vae_only_vrmse": vo_overall, "vae_dit_vrmse": vd_overall,
            "vae_only_per_channel": dict(zip(CHANNEL_NAMES, vo_channel)),
            "vae_dit_per_channel": dict(zip(CHANNEL_NAMES, vd_channel)),
        })
        sum_overall["vae_only"] += vo_overall
        sum_overall["vae_dit"] += vd_overall
        for c in range(4):
            sum_channel["vae_only"][c] += vo_channel[c]
            sum_channel["vae_dit"][c] += vd_channel[c]

        print(f"[{i+1}/{len(val_ids)}] {sid}: vae_only={vo_overall:.5f}  vae+dit={vd_overall:.5f}")

        if i < args.save_vis_samples:
            # Physical-units ground truth, straight from the dataset -- same
            # approach dit_visualization.py's own periodic panels use (not
            # `target`, which is the normalized-space vrmse comparand above).
            gt_physical = torch.stack([sample[name] for name in CHANNEL_NAMES]).unsqueeze(0)  # [1,C,F,H,W]
            gt_physical = pad_to(gt_physical, orig_F)

            vae_recon_physical = denormalize_fields(
                vae_recon, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )
            dit_pred_physical = denormalize_fields(
                dit_pred, stats["channel_mean"], stats["channel_std"], stats["normalization_clip"],
            )

            for label, pred_physical in (("vae_only", vae_recon_physical), ("vae_dit", dit_pred_physical)):
                save_reconstruction_panels(
                    prediction=pred_physical[0],
                    target=gt_physical[0],
                    sample_id=sid,
                    label=label,
                    frame_numbers=args.frame_numbers,
                    channel_names=CHANNEL_NAMES,
                    output_dir=args.out_dir,
                    dpi=args.dpi,
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu":
            torch.xpu.empty_cache()

    n = len(val_ids)
    summary = {
        "n_samples": n,
        "h5": h5_path,
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_ckpt),
        "vae_only_vrmse_mean": sum_overall["vae_only"] / n,
        "vae_dit_vrmse_mean": sum_overall["vae_dit"] / n,
        "vae_only_vrmse_per_channel": {name: sum_channel["vae_only"][c] / n for c, name in enumerate(CHANNEL_NAMES)},
        "vae_dit_vrmse_per_channel": {name: sum_channel["vae_dit"][c] / n for c, name in enumerate(CHANNEL_NAMES)},
        "per_sample": per_sample,
    }
    (args.out_dir / "vrmse_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print(f"n={n} sims")
    print(f"VAE-only  val_vrmse : {summary['vae_only_vrmse_mean']:.5f}")
    print(f"VAE+DiT   val_vrmse : {summary['vae_dit_vrmse_mean']:.5f}")
    delta = summary["vae_dit_vrmse_mean"] - summary["vae_only_vrmse_mean"]
    pct = delta / summary["vae_only_vrmse_mean"] * 100
    print(f"Delta (DiT forecasting cost on top of VAE recon): {delta:+.5f} ({pct:+.1f}%)")
    for name in CHANNEL_NAMES:
        vo = summary["vae_only_vrmse_per_channel"][name]
        vd = summary["vae_dit_vrmse_per_channel"][name]
        print(f"  {name:12s} vae_only={vo:.5f}  vae+dit={vd:.5f}  delta={vd - vo:+.5f}")
    print(f"\nSaved: {args.out_dir / 'vrmse_summary.json'}")
    if args.save_vis_samples > 0:
        print(f"Saved GT/Prediction/Residual panels for {min(args.save_vis_samples, n)} sample(s), "
              f"frames {args.frame_numbers}, under:")
        print(f"  {args.out_dir / 'visualizations' / 'vae_only'}/<sample_id>/frame_XXXX.png")
        print(f"  {args.out_dir / 'visualizations' / 'vae_dit'}/<sample_id>/frame_XXXX.png")


if __name__ == "__main__":
    main()
