#!/usr/bin/env python3
"""
Fine-tune ONLY the VAE decoder against the DiT's own real rollout latents,
instead of the encoder's clean posterior-mean latents it has only ever seen
during normal VAE training -- an exposure-bias-style train/inference
mismatch diagnosed via scripts/diagnose_frame_drift.py (see EXPERIMENTS.md's
"Why does a real ~10x latent-space accuracy win only produce a ~7%
pixel-space win" open question): within one arm's own rollout, latent
accuracy varying 44% (best vs worst quarter of frames) only moved decoded
pixel accuracy by 7% -- the decode(latent_error) -> pixel_error mapping
itself looks sharply sublinear/saturating, and the leading hypothesis is
that the decoder was never trained on the kind of latents a real DiT
rollout actually produces.

This is deliberately narrower than retraining the DiT itself (which would
mean threading a pixel-space loss through flow-matching's random-timestep
training and either running full multi-step sampling every training step,
prohibitively expensive, or an SNR-weighted single-step x0 estimate,
which touches windinet/training/dit_trainer.py's main training loop used
by every DiT arm in the project) -- decoder-only finetuning touches
nothing but the decoder submodule, on a small precomputed dataset, and is
architecturally already anticipated by this codebase: see
windinet.vae_adapter.latent_space_fingerprint's own docstring ("The
decoder is deliberately excluded [from the fingerprint]: a decoder-only
refinement improves reconstruction without invalidating a single
precomputed latent"). The output checkpoint is a byte-for-byte copy of the
input checkpoint except its `decoder.*` tensors -- same safetensors
format/metadata (`ltx-inflated-io-v1`), same encoder/adapter weights, same
latent-space fingerprint -- so it's a drop-in replacement anywhere
`--vae_checkpoint` is accepted (eval_dit_vrmse.py, inference_shockwave.py,
etc.), no re-encoding or DiT retraining required.

Two phases:
  1. Roll out the FROZEN DiT (no_grad) on --num_rollout_samples TRAIN-split
     sims once, caching (pred_latent, ground_truth_pixels) pairs on CPU --
     a one-time cost, not repeated per epoch.
  2. Train ONLY the decoder (AdamW, same rmse/h1/ssim loss composition and
     weights as VaeTrainer's own baseline config, windinet.losses
     .reconstruction_losses) to decode those cached pred_latents closer to
     their real ground truth, for --epochs passes over the cached set.

Before/after check: the SAME rollout + decode pipeline is also run on
--num_eval_samples VAL-split sims (disjoint from the finetuning set, and
seeded identically to eval_dit_vrmse.py's own val rollout so this is a
directly comparable same-sample-count slice, not an unrelated noise draw)
BEFORE finetuning and after EVERY epoch -- never used for a gradient
update, only this check. The decoder state from whichever epoch had the
best val pixel vrmse (possibly epoch 0 itself, if no epoch ever beat the
starting point) is what actually gets saved, not whatever the LAST epoch
happened to land on -- with only num_rollout_samples training pairs and no
other regularization, later-epoch overfitting to them is a real risk.

Usage: same checkpoint/config args as eval_dit_vrmse.py / diagnose_latent_scale.py.
    python scripts/finetune_decoder_on_rollout.py configs/dit/inference_dit.yaml \\
        --preprocessed_data_root dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \\
        --checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \\
        --scalar_checkpoint dit_outputs/lundquist/dit_sds_untrained/checkpoints/scalar_embedding_step_08033.safetensors \\
        --vae_checkpoint finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \\
        --num_rollout_samples 300 --epochs 15 --out_dir decoder_ft_out/sds_untrained
"""

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.config import ScalarConditioningConfig
from windinet.inference.model_loader import load_ltxv_components, select_vae_env
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.losses import SSIMLoss, reconstruction_losses, vrms_loss
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.shockwave_data import (
    CHANNEL_NAMES, ShockWaveDataset, build_shockwave_video, load_channel_normalization, normalize_fields,
)
from windinet.utils import get_default_device
from windinet.vae_adapter import latent_space_fingerprint

DTYPE = torch.bfloat16  # DiT rollout dtype (frozen, no_grad) -- decoder trains in fp32, see main().


# ----------------------------------------------------------------------
# Copied verbatim from scripts/eval_dit_vrmse.py / scripts/diagnose_latent_scale.py
# / scripts/diagnose_frame_drift.py -- scripts/ has no __init__.py, so these
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


def get_decoder(vae: torch.nn.Module) -> torch.nn.Module:
    """Same resolution order as windinet.training.vae_trainer.VaeTrainer._get_decoder --
    inflate-mode checkpoints (format ltx-inflated-io-v1, what every arm in
    this project uses) leave `vae` as the plain (inflated) VAE with
    `.decoder` directly; the AdaptedVAE branch is here only for parity with
    that function, not expected to trigger for these checkpoints."""
    from windinet.vae_adapter import AdaptedVAE
    if isinstance(vae, AdaptedVAE):
        return vae.vae.decoder
    return vae.decoder


def decode_latents(vae, latents: torch.Tensor, default_temb: float) -> torch.Tensor:
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    z = latents * std / sf + mean
    temb = torch.full((z.shape[0],), default_temb, device=z.device, dtype=z.dtype)
    return vae.decode(z, temb=temb, return_dict=True).sample


def trim_frames(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[2], b.shape[2])
    return a[:, :, :n], b[:, :, :n]


@torch.no_grad()
def rollout_dit(pipe, scalar_emb, scalar_cfg, sample, stats, device, num_frames,
                 num_inference_steps, guidance_scale, image_cond_noise_scale, seed) -> torch.Tensor:
    """Frozen DiT forward -- one sample -> its predicted latent, [1,C,F,H,W]."""
    H, W = sample["density"].shape[-2:]
    cond_video = build_initial_condition(sample, stats, device)
    scalar_values = [sample["meta"][n] for n in scalar_cfg.scalar_names]
    scalars = torch.tensor([scalar_values], device=device, dtype=DTYPE)
    prompt_embeds = scalar_emb(scalars)
    prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)
    g = torch.Generator(device=device).manual_seed(seed)
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
    return out.frames.float()


def build_dataset(dataset, ids, pipe, scalar_emb, scalar_cfg, stats, device, cfg, seed_base, label):
    """Roll out the frozen DiT once per id, cache (pred_latent, gt_pixels) on CPU."""
    pairs = []
    for i, sid in enumerate(ids):
        idx = dataset.ids.index(sid)
        sample = dataset[idx]
        orig_F = sample["density"].shape[0]
        gt_video = build_shockwave_video(
            sample, device=device, channel_mean=stats["channel_mean"],
            channel_std=stats["channel_std"], normalization_clip=stats["normalization_clip"],
        )
        target = gt_video[:, :, :orig_F].float().cpu()
        pred_latent = rollout_dit(
            pipe, scalar_emb, scalar_cfg, sample, stats, device,
            cfg.get("num_frames", 105), cfg.get("num_inference_steps", 2), cfg.get("guidance_scale", 1.0),
            cfg.get("image_cond_noise_scale", 0.0), seed_base + i,
        ).cpu()
        pairs.append((pred_latent, target))
        if (i + 1) % 25 == 0 or i == len(ids) - 1:
            print(f"[{label}] rolled out {i+1}/{len(ids)}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pairs


def eval_pixel_vrmse(vae, pairs, default_temb) -> float:
    """Decode every cached pred_latent with the CURRENT decoder, average pixel vrmse."""
    total = 0.0
    with torch.no_grad():
        for pred_latent, target in pairs:
            pred_latent, target = pred_latent.to(next(vae.parameters()).device), target.to(next(vae.parameters()).device)
            dit_pred = decode_latents(vae, pred_latent.to(next(vae.parameters()).dtype), default_temb).float()
            dit_pred, target_cmp = trim_frames(dit_pred, target)
            total += float(vrms_loss(dit_pred, target_cmp).item())
    return total / len(pairs)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--preprocessed_data_root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--scalar_checkpoint", type=Path, default=None)
    ap.add_argument("--vae_checkpoint", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, default=None)
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--default_temb", type=float, default=0.0)
    ap.add_argument("--num_rollout_samples", type=int, default=300,
                     help="TRAIN-split sims to roll out ONCE and cache for decoder finetuning")
    ap.add_argument("--num_eval_samples", type=int, default=20,
                     help="VAL-split sims (disjoint from finetuning data) for the before/after check")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--rmse_weight", type=float, default=1.0)
    ap.add_argument("--h1_weight", type=float, default=50.0)
    ap.add_argument("--ssim_weight", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("decoder_ft_out"))
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


def main():
    args = parse_args()
    cfg = load_config(args)
    device = get_default_device()
    random.seed(args.seed)

    manifest = json.loads((args.preprocessed_data_root / "split_manifest.json").read_text())
    h5_path = manifest["data_root"]
    train_ids = manifest["train_ids"][:]
    random.shuffle(train_ids)
    train_ids = train_ids[: args.num_rollout_samples]
    val_ids = manifest["val_ids"][: args.num_eval_samples]
    print(f"Finetuning on {len(train_ids)} TRAIN-split sim(s), checking on "
          f"{len(val_ids)} disjoint VAL-split sim(s), from {h5_path}")

    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = ensure_checkpoint(cfg["scalar_checkpoint"])
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

    # --- Phase 1: cache DiT rollouts (frozen, no_grad, one-time cost) ---
    train_pairs = build_dataset(dataset, train_ids, pipe, scalar_emb, scalar_cfg, stats, device, cfg,
                                 seed_base=cfg.get("seed", 42), label="train")
    # Same seed_base convention as eval_dit_vrmse.py's own val rollout (seed + i,
    # no offset) -- val_ids here are that script's own first --num_eval_samples,
    # so this reproduces (up to sampling floating-point nondeterminism) the
    # same predicted latents eval_dit_vrmse.py would compute for those ids,
    # making this script's "before" number directly comparable to a
    # same-sample-count slice of the established eval_dit_vrmse.py results
    # instead of using an unrelated noise draw.
    val_pairs = build_dataset(dataset, val_ids, pipe, scalar_emb, scalar_cfg, stats, device, cfg,
                               seed_base=cfg.get("seed", 42), label="val")

    # --- Decoder to fp32 for stable small-scale finetuning (rollout above
    # stays bf16/frozen; only the decoder we're about to train switches). ---
    pipe.vae.to(dtype=torch.float32)
    decoder = get_decoder(pipe.vae)
    decoder.requires_grad_(True)
    decoder.train()
    print(f"Decoder trainable params: {sum(p.numel() for p in decoder.parameters()):,}")

    pixel_vrmse_before = eval_pixel_vrmse(pipe.vae, val_pairs, args.default_temb)
    print(f"\nBEFORE decoder finetune: val pixel vrmse = {pixel_vrmse_before:.5f} (n={len(val_pairs)})\n")

    ssim_module = SSIMLoss(channels=4, window_size=11, sigma=1.5).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_vrmse = pixel_vrmse_before
    best_epoch = 0
    best_decoder_sd = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    print(f"Epoch 0 (pre-finetune) is the initial best (val pixel vrmse = {best_val_vrmse:.5f})")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_pairs)
        epoch_losses = {"rmse": 0.0, "h1": 0.0, "ssim": 0.0, "vrms": 0.0, "total": 0.0}
        n_batches = 0
        for start in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[start: start + args.batch_size]
            optimizer.zero_grad()
            batch_totals = {"rmse": 0.0, "h1": 0.0, "ssim": 0.0, "vrms": 0.0, "total": 0.0}
            for pred_latent, target in batch:
                # pred_latent is in LATENT-space frame count (from the DiT rollout,
                # fixed at cfg's num_frames after the VAE's temporal downsampling),
                # target is in PIXEL-space frame count (orig_F, varies per sim) --
                # the two are NOT directly comparable until AFTER decode, unlike
                # diagnose_frame_drift.py's latent-vs-latent trim. Decode first,
                # trim in pixel space (matches eval_dit_vrmse.py's own convention).
                pred_latent = pred_latent.to(device)
                target = target.to(device)
                dit_pred = decode_latents(pipe.vae, pred_latent, args.default_temb)
                n_px = min(dit_pred.shape[2], target.shape[2])
                dit_pred_cmp, target_cmp = dit_pred[:, :, :n_px], target[:, :, :n_px]

                losses = reconstruction_losses(dit_pred_cmp, target_cmp, ssim_module=ssim_module, compute_mlw=False)
                sample_loss = (
                    args.rmse_weight * losses["rmse"] + args.h1_weight * losses["h1"] + args.ssim_weight * losses["ssim"]
                )
                # Per-sample backward scaled by 1/len(batch), accumulated into the
                # decoder's .grad before a single optimizer.step() below -- avoids
                # concatenating tensors of different per-sim frame counts into one
                # batch dim (see the comment above).
                (sample_loss / len(batch)).backward()

                # reconstruction_losses already computes "vrms" (same formula as
                # vrms_loss) unconditionally -- reuse it instead of a second pass.
                v = losses["vrms"]
                batch_totals["rmse"] += float(losses["rmse"].item())
                batch_totals["h1"] += float(losses["h1"].item())
                batch_totals["ssim"] += float(losses["ssim"].item())
                batch_totals["vrms"] += float(v.item())
                batch_totals["total"] += float(sample_loss.item())
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += batch_totals[k] / len(batch)
            n_batches += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Per-epoch validation -- val_pairs are NEVER used for a gradient
        # update, only for this check, so this stays a true held-out signal.
        decoder.eval()
        val_vrmse_epoch = eval_pixel_vrmse(pipe.vae, val_pairs, args.default_temb)
        decoder.train()
        improved = val_vrmse_epoch < best_val_vrmse
        if improved:
            best_val_vrmse, best_epoch = val_vrmse_epoch, epoch
            best_decoder_sd = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}

        history.append({"epoch": epoch, **epoch_losses, "val_pixel_vrmse": val_vrmse_epoch})
        print(f"epoch {epoch}/{args.epochs}  total={epoch_losses['total']:.5f}  "
              f"rmse={epoch_losses['rmse']:.5f}  h1={epoch_losses['h1']:.5f}  "
              f"ssim={epoch_losses['ssim']:.5f}  train_vrms={epoch_losses['vrms']:.5f}  "
              f"val_pixel_vrmse={val_vrmse_epoch:.5f}{'  <- best so far' if improved else ''}")

    # Restore the best-val-epoch weights (possibly epoch 0 / pre-finetune, if
    # no epoch ever beat the starting point) rather than blindly keeping
    # whatever the LAST epoch happened to land on -- with only
    # num_rollout_samples training pairs and no other regularization, later
    # epochs overfitting to them is a real risk, not a hypothetical one.
    decoder.load_state_dict(best_decoder_sd)
    decoder.eval()
    print(f"\nRestored best epoch: {best_epoch} (val pixel vrmse = {best_val_vrmse:.5f}, "
          f"vs. epoch {args.epochs}'s {history[-1]['val_pixel_vrmse']:.5f})")

    pixel_vrmse_after = best_val_vrmse
    print(f"\nBEFORE decoder finetune: val pixel vrmse = {pixel_vrmse_before:.5f} (n={len(val_pairs)})")
    print(f"AFTER  decoder finetune: val pixel vrmse = {pixel_vrmse_after:.5f} (n={len(val_pairs)}, epoch {best_epoch})")
    print(f"Delta: {pixel_vrmse_after - pixel_vrmse_before:+.5f} "
          f"({(pixel_vrmse_after - pixel_vrmse_before) / pixel_vrmse_before * 100:+.1f}%)")

    # --- Write a new checkpoint: original file, decoder.* tensors replaced ---
    with safe_open(str(vae_ckpt), framework="pt", device="cpu") as f:
        orig_metadata = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("decoder.")}
    tensors.update({f"decoder.{k}": v.contiguous() for k, v in best_decoder_sd.items()})
    out_ckpt = args.out_dir / "vae_shockwave_decoder_ft.safetensors"
    save_file(tensors, out_ckpt, metadata=orig_metadata)
    print(f"\nSaved decoder-finetuned checkpoint (same format/metadata, drop-in compatible): {out_ckpt}")

    summary = {
        "source_vae_checkpoint": str(vae_ckpt),
        "dit_checkpoint": str(checkpoint),
        "num_rollout_samples": len(train_pairs),
        "num_eval_samples": len(val_pairs),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "pixel_vrmse_before": pixel_vrmse_before,
        "pixel_vrmse_after": pixel_vrmse_after,
        "history": history,
        "out_checkpoint": str(out_ckpt),
    }
    (args.out_dir / "finetune_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {args.out_dir / 'finetune_summary.json'}")


if __name__ == "__main__":
    main()
