#!/bin/bash
#SBATCH --job-name=dit_preprocess
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Encode step for the lundquist DiT pipeline. Companion to
# train_dit_2gpu.sbatch -- run this first to turn a VAE checkpoint into
# latents, then that job trains on the output. 1 GPU: scripts/preprocess_
# dataset.py is a single-process encode loop (no DDP/accelerate inside it),
# same reasoning as jobs/lrz_ai/preprocess_dit_data.job.
#
# VAE_CHECKPOINT is REQUIRED (env var), e.g. a checkpoint produced by
# jobs/lundquist/finetune_vae_2gpu.sbatch under
# finetune_vae_outputs/lundquist/<run>/checkpoints/vae_shockwave_best.safetensors.
#
# $1 (optional) overrides OUTPUT_NAME -- the latents land in
# dit_preprocessed/<OUTPUT_NAME>/. Defaults to VAE_CHECKPOINT's own run
# directory name, so pointing at a different checkpoint naturally lands in a
# different output directory instead of silently overwriting a previous
# encode. $2 (optional) overrides DATA_ROOT (default: lundquist's 128x128_ds
# train split, see windinet/cluster_config.py CLUSTER_DEFAULTS).
#
# MAX_SAMPLES/EVAL_SIMS/SPLIT_SEED are optional env vars. Default is 0/0/42
# -- encode everything, no held-out split. --max-samples and --eval-sims
# cannot both be set (preprocess_dataset.py rejects it -- max-samples shifts
# the split permutation, see its own docstring), so pick one:
#   MAX_SAMPLES=24 for a fast smoke-test encode (~1-2 min for 24 samples)
#   EVAL_SIMS=675 SPLIT_SEED=42 to reproduce the VAE run's own held-out split
#   for a real (non-smoke-test) run whose DiT val_loss should be comparable
#   to that VAE's val_vrmse.
#
# Submit with:
#   VAE_CHECKPOINT=finetune_vae_outputs/lundquist/<run>/checkpoints/vae_shockwave_best.safetensors \
#     sbatch jobs/lundquist/preprocess_dit_data.sbatch [OUTPUT_NAME] [DATA_ROOT]
#
#   # smoke test, 24 samples:
#   VAE_CHECKPOINT=finetune_vae_outputs/lundquist/finetune_vae_dit_smoketest_source/checkpoints/vae_shockwave_best.safetensors \
#     MAX_SAMPLES=24 sbatch jobs/lundquist/preprocess_dit_data.sbatch smoketest

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [[ -z "${VAE_CHECKPOINT:-}" ]]; then
    echo "error: VAE_CHECKPOINT is required, e.g." >&2
    echo "  finetune_vae_outputs/lundquist/<run>/checkpoints/vae_shockwave_best.safetensors" >&2
    exit 1
fi

DEFAULT_OUTPUT_NAME=$(basename "$(dirname "$(dirname "${VAE_CHECKPOINT}")")")
OUTPUT_NAME=${1:-${DEFAULT_OUTPUT_NAME}}
DATA_ROOT=${2:-/local/disk/hramachandran/work/wh_work/windinet/euler_mq_dataset/128x128_ds/train.h5}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_SIMS=${EVAL_SIMS:-0}
SPLIT_SEED=${SPLIT_SEED:-42}
OUT_DIR="dit_preprocessed/${OUTPUT_NAME}"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_NAME: ${OUTPUT_NAME} -> ${OUT_DIR}"

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${VAE_CHECKPOINT}" "${OUT_DIR}" >> logs/lundquist/INDEX.tsv

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
    EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ "${EVAL_SIMS}" != "0" ]]; then
    EXTRA_ARGS+=(--eval-sims "${EVAL_SIMS}" --split-seed "${SPLIT_SEED}")
fi

"${PYTHON}" scripts/preprocess_dataset.py \
    "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --inflate-checkpoint "${VAE_CHECKPOINT}" \
    "${EXTRA_ARGS[@]}"

echo "Latents written to ${OUT_DIR}"
