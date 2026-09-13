#!/bin/bash
#SBATCH --job-name=diagnose_frame_drift
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Single-GPU launcher for scripts/diagnose_frame_drift.py -- checks WHERE in
# the rollout (which frame index) sds_untrained's pixel-space error comes
# from, to distinguish decoder sensitivity from compounding rollout drift
# as the reason a real ~10x latent-space accuracy win (see
# diagnose_latent_scale.py) only produces a ~7% pixel-space win. See the
# script's own docstring for the full reasoning. Mirrors
# jobs/lundquist/diagnose_latent_scale.sbatch's conventions.
#
# Usage:
#   sbatch jobs/lundquist/diagnose_frame_drift.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES]
#
# Same PREPROCESSED_DATA_ROOT/DIT_CHECKPOINT/VAE_CHECKPOINT conventions as
# jobs/lundquist/eval_dit_vrmse.sbatch. Run once per arm you want the
# frame-index breakdown for (sds_untrained is the interesting one; kl0 as
# a comparison point is useful too).
#
# Example:
#   sbatch jobs/lundquist/diagnose_frame_drift.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/diagnose_frame_drift.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-20}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/diagnose_frame_drift/${SLURM_JOB_ID}"
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "OUT_DIR: ${OUT_DIR}"

"${PYTHON}" scripts/diagnose_frame_drift.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --out_dir "${OUT_DIR}"
