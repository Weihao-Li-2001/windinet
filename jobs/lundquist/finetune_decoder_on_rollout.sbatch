#!/bin/bash
#SBATCH --job-name=finetune_decoder_on_rollout
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Single-GPU launcher for scripts/finetune_decoder_on_rollout.py -- fine-tunes
# ONLY the VAE decoder against the DiT's own real rollout latents (instead
# of the encoder's clean posterior-mean latents it has only ever seen
# before), to test whether this closes any of the gap between DiT latent
# accuracy and decoded pixel accuracy diagnosed by diagnose_frame_drift.py
# (see that script's docstring and EXPERIMENTS.md's "Why does a real ~10x
# latent-space accuracy win only produce a ~7% pixel-space win" open
# question). Output is a drop-in-compatible VAE checkpoint (same format,
# same encoder/adapter weights, same latent-space fingerprint) -- no
# re-encoding or DiT retraining needed to use it.
#
# --time=04:00:00 is an UNCALIBRATED GUESS: 300 rollout samples (~similar
# cost per-sample to eval_dit_vrmse.py, which does ~10.7s/sample on
# sng_pvc) plus 20 eval rollouts plus 15 decoder-only epochs (cheap, no
# DiT forward pass) over the cached set. Bump --time if a larger
# NUM_ROLLOUT_SAMPLES/EPOCHS combination times out.
#
# Usage:
#   sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [EPOCHS]
#
# Same PREPROCESSED_DATA_ROOT/DIT_CHECKPOINT/VAE_CHECKPOINT conventions as
# jobs/lundquist/eval_dit_vrmse.sbatch. NUM_ROLLOUT_SAMPLES defaults to 300,
# EPOCHS to 15.
#
# Example:
#   sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch \
#     dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_sds_untrained \
#     dit_outputs/lundquist/dit_sds_untrained/checkpoints/model_weights_step_08033.safetensors \
#     finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep20_sds_untrained/checkpoints/vae_shockwave_best.safetensors

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/finetune_decoder_on_rollout.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_ROLLOUT_SAMPLES] [EPOCHS]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_ROLLOUT_SAMPLES=${4:-300}
EPOCHS=${5:-15}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/finetune_decoder_on_rollout/${SLURM_JOB_ID}"
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_ROLLOUT_SAMPLES: ${NUM_ROLLOUT_SAMPLES}"
echo "EPOCHS: ${EPOCHS}"
echo "OUT_DIR: ${OUT_DIR}"

"${PYTHON}" scripts/finetune_decoder_on_rollout.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_rollout_samples "${NUM_ROLLOUT_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --out_dir "${OUT_DIR}"
