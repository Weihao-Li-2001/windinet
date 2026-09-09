#!/bin/bash
# One-off eval_dit_vrmse submission for the three sng_pvc DiT arms that
# finished their full 8000-step run on 2026-09-08/09 (jobs 534979/535063/
# 535065): cosine_kl1e5, cosine_anchor, and ep20_baseline_std1p5. Their
# train-loss curves aren't directly comparable to each other (different VAE
# latent spaces / cosine_anchor and std1p5 aren't even the same VAE arm as
# kl1e5), and cosine_anchor + std1p5 both plateaued in train loss from
# step ~5000 on while kl1e5 kept dropping -- this submits the real
# vae-only-vs-vae+dit vrmse comparison (scripts/eval_dit_vrmse.py, same
# formula as the committed VAE val_vrmse numbers) to find out whether that
# plateau actually means anything or is just a latent-space-scale artifact.
#
# Checkpoint/preprocessed-root arguments below were reconstructed from
# logs/sng_pvc/INDEX.tsv + each arm's config output_dir + the last
# "Saved checkpoint step_NNNNN" line in that job's .out log (checkpoints.
# keep_last_n=2, so only the last two step checkpoints survive on disk --
# CHECK BEFORE SUBMITTING that model_weights_step_08025/08033.safetensors
# still exist under each arm's checkpoints/ dir; if a later resume run or
# manual cleanup rotated them out, substitute the actual surviving max step).
#
# NUM_SAMPLES left at the script's full-eval precedent (675, all held-out
# sims -- see jobs/sng_pvc/eval_dit_vrmse.sbatch's own calibration comment,
# ~2h wall time per arm at that count) rather than the 50-sample sanity
# default, since we want numbers directly comparable to the already-done
# eval_dit_vrmse runs (529873/529874/529978/529979 etc.), not a quick check.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet).
set -euo pipefail

SCRATCH_ROOT=/hppfs/scratch/0D/go76fuz2/windinet

# --- cosine_kl1e5 (job 534979, final checkpoint step_08025) ---
sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch \
    "${SCRATCH_ROOT}/dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_256res_cosine_kl1e5" \
    "${SCRATCH_ROOT}/outputs/shockwave_dit_cosine_kl1e5/checkpoints/model_weights_step_08025.safetensors" \
    "${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep20_256res_cosine_kl1e5/checkpoints/vae_shockwave_best.safetensors" \
    675

# --- cosine_anchor (job 535063, final checkpoint step_08033) ---
sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch \
    "${SCRATCH_ROOT}/dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_256res_cosine_anchor" \
    "${SCRATCH_ROOT}/outputs/shockwave_dit_cosine_anchor/checkpoints/model_weights_step_08033.safetensors" \
    "${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep20_256res_cosine_anchor/checkpoints/vae_shockwave_best.safetensors" \
    675

# --- ep20_baseline_std1p5 (job 535065, final checkpoint step_08033) ---
# Same VAE arm as the plain ep20 baseline (std1p5 only varies
# flow_matching.timestep_sampling_params.std), so this points at the
# baseline VAE checkpoint, not a std1p5-specific one.
sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch \
    "${SCRATCH_ROOT}/dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_256res" \
    "${SCRATCH_ROOT}/outputs/shockwave_dit_ep20_baseline_std1p5/checkpoints/model_weights_step_08033.safetensors" \
    "${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep20_256res/checkpoints/vae_shockwave_best.safetensors" \
    675
