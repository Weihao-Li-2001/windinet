#!/usr/bin/env python3
"""
Diagnose whether a low latent_vrmse is a genuine forecast-accuracy win or an
artifact of the target latent's own variance shrinking.

Motivation: comparing lundquist's cosine_kl0 arm (latent_vrmse ~0.61-0.64)
against dit_sds_untrained (latent_vrmse ~0.073, an ~9x improvement) but
finding pixel_vrmse essentially unchanged between the two (~0.42-0.44 for
both) is suspicious. windinet.losses.vrms.vrms_loss normalizes by the
TARGET's own per-sample variance:

    VRMSE = sqrt(mean((pred - target)^2) / var(target))

so it is invariant to a uniform rescaling of the whole latent tensor (pred
and target would scale together, canceling in the ratio) -- but it is NOT
protected against the target's own INTERNAL variance (across C,T,H,W within
one sample) shrinking, e.g. because an SDS-distilled VAE encoder learned to
make its latents spatially/channel-wise flatter (easier for the frozen
critic to denoise) rather than more physically predictable. That would show
up as: latent_vrmse looking great (small mse over small variance) while the
UNNORMALIZED latent RMSE and the decoded pixel_vrmse (which uses the actual
physical field's own variance, not the latent's) don't improve at all --
exactly what was observed.

This script isolates the effect by reporting, per arm, in the SAME latent
space eval_dit_vrmse.py uses:
    - target_std_mean:   sqrt(the same per-sample variance vrms_loss's
                          denominator uses) -- the target's own internal
                          dynamic range. If this differs a lot between arms,
                          the vrmse numbers aren't on a level footing.
    - latent_rmse_mean:  raw, UNNORMALIZED sqrt(mse(pred, target)) -- the
                          actual absolute prediction error, independent of
                          how "easy" the target's own scale makes it look.
    - latent_vrmse_mean: the normal, self-normalized metric (matches
                          dit_vrmse_metrics.csv / eval_dit_vrmse.py).
    - cross_normalized_vrmse (only when --reference_std is passed): this
                          arm's raw latent_rmse_mean divided by a FIXED
                          reference std (typically the baseline arm's own
                          target_std_mean) instead of this arm's own target
                          variance -- answers "how good would this arm look
                          if judged on the baseline's yardstick instead of
                          its own".

Usage (one arm per invocation, same checkpoint/config args as
eval_dit_vrmse.py -- run once for the baseline arm to get its
target_std_mean, then again for the arm in question with
--reference_std <that number> to get the cross-normalized comparison):

    python scripts/diagnose_latent_scale.py configs/dit/inference_dit.yaml \\
        --preprocessed_data_root dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_kl0 \\
        --checkpoint dit_outputs/lundquist/dit_cosine_kl0/checkpoints/model_weights_step_08033.safetensors \\
        --scalar_checkpoint dit_outputs/lundquist/dit_cosine_kl0/checkpoints/scalar_embedding_step_08033.safetensors \\
        --vae_checkpoint finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_kl0/checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 20 --out_dir diag_out/kl0

    python scripts/diagnose_latent_scale.py configs/dit/inference_dit.yaml \\
        --preprocessed_data_root dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \\
        --checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \\
        --scalar_checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/scalar_embedding_step_08033.safetensors \\
        --vae_checkpoint finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \\
        --num_samples 20 --reference_std <kl0's target_std_mean from the run above> --out_dir diag_out/sds_untrained

--num_samples defaults to 20 (this is a diagnostic, not a final eval -- no
need for the full 675-sample val set eval_dit_vrmse.py uses).
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
from windinet.losses import rmse_loss, vrms_loss
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, load_channel_normalization
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16


# ----------------------------------------------------------------------
# Copied verbatim from scripts/eval_dit_vrmse.py (itself copied from
# scripts/inference_shockwave.py) -- scripts/ has no __init__.py, so these
# small already-tested helpers are duplicated rather than imported. Any
# change there should be mirrored here.
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
        raise SystemExit(
            f"VAE / DiT latent space mismatch: DiT trained on {expected}, this VAE gives {actual}."
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
    from windinet.training.shockwave_data import normalize_fields
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


def trim_latent_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


# ----------------------------------------------------------------------
# The actual diagnostic.
# ----------------------------------------------------------------------

def target_std(target: torch.Tensor, eps: float = 1e-8) -> float:
    """sqrt of the SAME per-sample variance vrms_loss's denominator uses
    (dims 1..end, i.e. C,T,H,W collapsed together) -- the target's own
    internal dynamic range, in the same units vrms_loss operates in."""
    dims = tuple(range(1, target.dim()))
    return float(target.var(dim=dims, unbiased=False).add(eps).sqrt().mean().item())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--preprocessed_data_root", type=Path, required=True)
    ap.add_argument("--num_samples", type=int, default=20,
                     help="Diagnostic, not a final eval -- default 20 (vs eval_dit_vrmse.py's full val set)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--scalar_checkpoint", type=Path, default=None)
    ap.add_argument("--untrained_dit", action="store_true")
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, default=None)
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--reference_std", type=float, default=None,
                     help="Another arm's target_std_mean (usually the baseline being compared against) -- "
                          "if given, this arm's raw latent_rmse_mean is ALSO divided by this fixed value "
                          "instead of its own target variance, to see how good this arm looks on the "
                          "baseline's yardstick rather than its own.")
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
    checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = None if args.untrained_dit else ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_frames = cfg.get("num_frames", 105)
    image_cond_noise_scale = cfg.get("image_cond_noise_scale", 0.0)
    seed = cfg.get("seed", 42)

    vae_ckpt = cfg["vae_checkpoint"]
    select_vae_env(ensure_checkpoint(vae_ckpt))
    stats = load_channel_normalization(cfg["normalization"])
    if not args.untrained_dit:
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
    if args.untrained_dit:
        scalar_emb = ScalarEmbedding(scalar_cfg).to(device=device, dtype=DTYPE).eval()
    else:
        load_transformer_weights(pipe, checkpoint)
        scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    from windinet.training.shockwave_data import build_shockwave_video

    sum_target_std, sum_raw_rmse, sum_vrmse = 0.0, 0.0, 0.0
    for i, sid in enumerate(val_ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        H, W = sample["density"].shape[-2:]

        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
        )
        gt_latent = vae_encode(pipe.vae, gt_video.to(DTYPE)).float()

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
        pred_lat, gt_lat = trim_latent_frames(out.frames.float(), gt_latent)

        s = target_std(gt_lat)
        raw_rmse = float(rmse_loss(pred_lat, gt_lat).item())
        vrmse = float(vrms_loss(pred_lat, gt_lat).item())
        sum_target_std += s
        sum_raw_rmse += raw_rmse
        sum_vrmse += vrmse
        print(f"[{i+1}/{len(val_ids)}] {sid}: target_std={s:.5f}  raw_rmse={raw_rmse:.5f}  vrmse={vrmse:.5f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(val_ids)
    summary = {
        "n_samples": n,
        "checkpoint": str(checkpoint) if checkpoint else "untrained",
        "vae_checkpoint": str(vae_ckpt),
        "target_std_mean": sum_target_std / n,
        "latent_rmse_mean": sum_raw_rmse / n,
        "latent_vrmse_mean": sum_vrmse / n,
    }
    if args.reference_std is not None:
        summary["reference_std"] = args.reference_std
        summary["cross_normalized_vrmse"] = (sum_raw_rmse / n) / args.reference_std

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scale_diagnostic.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print(f"n={n} sample(s)")
    print(f"target_std_mean   (target's own internal dynamic range) : {summary['target_std_mean']:.5f}")
    print(f"latent_rmse_mean  (raw, UNNORMALIZED prediction error)  : {summary['latent_rmse_mean']:.5f}")
    print(f"latent_vrmse_mean (self-normalized, matches training viz): {summary['latent_vrmse_mean']:.5f}")
    if args.reference_std is not None:
        print(f"\nCross-normalized against reference_std={args.reference_std:.5f}:")
        print(f"  cross_normalized_vrmse = latent_rmse_mean / reference_std = {summary['cross_normalized_vrmse']:.5f}")
        print("  (compare this to latent_vrmse_mean above -- if it's much larger, the small")
        print("   latent_vrmse_mean was mostly this arm's own target variance shrinking, not")
        print("   genuinely more accurate absolute prediction)")
    print(f"\nSaved: {args.out_dir / 'scale_diagnostic.json'}")


if __name__ == "__main__":
    main()
