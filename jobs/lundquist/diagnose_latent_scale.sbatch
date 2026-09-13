#!/bin/bash
#SBATCH --job-name=diagnose_latent_scale
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Single-GPU launcher for scripts/diagnose_latent_scale.py -- checks whether
# an arm's low latent_vrmse (e.g. dit_sds_untrained's ~0.073 vs
# dit_cosine_kl0's ~0.61-0.64, a ~9x gap that does NOT show up in
# pixel_vrmse) is a genuine forecast-accuracy win or an artifact of that
# arm's own latent having lower internal variance (see the script's own
# docstring for the full reasoning). Mirrors jobs/lundquist/eval_dit_vrmse.sbatch's
# conventions (same PYTHON path, cd, single GPU, forward-only) but on a
# small default sample count (20) since this is a diagnostic, not a final
# eval -- ~30 min budget is generous headroom, not an expected runtime.
#
# Usage:
#   sbatch jobs/lundquist/diagnose_latent_scale.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [REFERENCE_STD]
#
# Same PREPROCESSED_DATA_ROOT/DIT_CHECKPOINT/VAE_CHECKPOINT conventions as
# jobs/lundquist/eval_dit_vrmse.sbatch. Run once for the baseline arm
# (REFERENCE_STD omitted) to get its target_std_mean from the printed
# summary / <OUT_DIR>/scale_diagnostic.json, then again for the arm in
# question passing that value as REFERENCE_STD.
#
# Example -- run for cosine_kl0 first to get its target_std_mean, then
# for dit_sds_untrained with that value as REFERENCE_STD:
#   sbatch jobs/lundquist/diagnose_latent_scale.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_kl0 \
#     dit_outputs/lundquist/dit_cosine_kl0/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_kl0/checkpoints/vae_shockwave_best.safetensors
#   # then, using that run's target_std_mean as REFERENCE_STD:
#   sbatch jobs/lundquist/diagnose_latent_scale.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors \
#     20 0.61234

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/diagnose_latent_scale.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [REFERENCE_STD]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-20}
REFERENCE_STD=${5:-}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/diagnose_latent_scale/${SLURM_JOB_ID}"

# Sibling file, same step (same convention as eval_dit_vrmse.sbatch).
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "REFERENCE_STD: ${REFERENCE_STD:-<none -- this is the baseline run>}"
echo "OUT_DIR: ${OUT_DIR}"

EXTRA_ARGS=()
if [ -n "${REFERENCE_STD}" ]; then
    EXTRA_ARGS+=(--reference_std "${REFERENCE_STD}")
fi

"${PYTHON}" scripts/diagnose_latent_scale.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
