#!/bin/bash
# One-off eval_dit_vrmse submission for the SDS (score-distillation) VAE
# arm's DiT (job 535840, shockwave_dit_sds), which finished its full
# 8000-step run on 2026-09-11 (8033/8000 steps, 1145.9 min, clean exit --
# see logs/sng_pvc/535840-shockwave_dit.out). This is the arm flagged as
# "training, no eval numbers yet" in the 2026-09-10 KL-sweep update -- see
# configs/dit/train_dit_sng_pvc_sds.yaml's own header for how this arm's
# VAE/DiT recipe relates to the cosine/KL sweep family.
#
# Checkpoint/preprocessed-root arguments below come from logs/sng_pvc/
# INDEX.tsv (jobs 535151/535182 VAE finetune, 535374 preprocess, 535840
# DiT train) plus the last "saved" line in 535840's own log
# (model_weights_step_08033.safetensors; checkpoints.keep_last_n=2, so only
# 07920 and 08033 survive on disk -- CHECK BEFORE SUBMITTING that
# model_weights_step_08033.safetensors still exists under
# shockwave_dit_sds/checkpoints/, in case a later resume or manual cleanup
# rotated it out).
#
# NUM_SAMPLES left at the full-eval precedent (675, all held-out sims) to
# get a number directly comparable to the existing ep20_baseline/cosine_kl*/
# cosine_anchor eval_dit_vrmse runs (see jobs/sng_pvc/
# eval_finished_dit_arms.sh), not a quick sanity check.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet).
set -euo pipefail

SCRATCH_ROOT=/hppfs/scratch/0D/go76fuz2/windinet

sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch \
    "${SCRATCH_ROOT}/dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_256res_sds" \
    "${SCRATCH_ROOT}/outputs/shockwave_dit_sds/checkpoints/model_weights_step_08033.safetensors" \
    "${SCRATCH_ROOT}/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep20_256res_sds/checkpoints/vae_shockwave_best.safetensors" \
    675
