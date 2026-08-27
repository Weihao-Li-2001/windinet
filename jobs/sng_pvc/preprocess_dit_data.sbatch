#!/bin/bash
#SBATCH --job-name=wh_preprocess_dit
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Encode step for the sng_pvc DiT pipeline (Intel Data Center GPU Max 1550 /
# XPU). Companion to train_dit.sbatch -- run this first to turn a VAE
# checkpoint into latents, then that job trains on the output. 1 tile:
# scripts/preprocess_dataset.py is a single-process encode loop (no DDP/
# accelerate inside it -- windinet/utils.py's get_default_device() picks XPU
# automatically), same reasoning as jobs/lrz_ai/preprocess_dit_data.job and
# jobs/lundquist/preprocess_dit_data.sbatch.
#
# VAE_CHECKPOINT is REQUIRED (env var), e.g. a checkpoint produced by
# jobs/sng_pvc/finetune_vae.sbatch under
# ${SCRATCH}/windinet/finetune_vae_outputs_sng_pvc/<run>/checkpoints/vae_shockwave_best.safetensors
# (see windinet/cluster_config.py CLUSTER_DEFAULTS['sng_pvc']['output_root']).
#
# $1 (optional) overrides OUTPUT_NAME -- latents land in
# ${SCRATCH}/windinet/dit_preprocessed/<OUTPUT_NAME>/. Defaults to
# VAE_CHECKPOINT's own run directory name. $2 (optional) overrides DATA_ROOT
# (default: 256x256_ds train split -- every VAE arm this pipeline targets is
# 256res, unlike CLUSTER_DEFAULTS['sng_pvc']'s 128x128 default for VAE
# finetuning itself).
#
# EVAL_SIMS/SPLIT_SEED are optional env vars, default 675/42 -- reproduces
# the VAE runs' own held-out split so DiT val_loss is comparable to that
# VAE's val_vrmse (see jobs/lrz_ai/preprocess_dit_data.job's own header for
# why). Set EVAL_SIMS=0 to disable the split; MAX_SAMPLES for a fast
# smoke-test encode instead (cannot combine with EVAL_SIMS, same restriction
# as preprocess_dataset.py itself).
#
# Submit with:
#   VAE_CHECKPOINT=${SCRATCH}/windinet/finetune_vae_outputs_sng_pvc/<run>/checkpoints/vae_shockwave_best.safetensors \
#     sbatch jobs/sng_pvc/preprocess_dit_data.sbatch [OUTPUT_NAME] [DATA_ROOT]

# No `-u`, same reasoning as jobs/sng_pvc/finetune_vae.sbatch: `module load
# micromamba` itself references unset-by-default variables.
set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

if [[ -z "${VAE_CHECKPOINT:-}" ]]; then
    echo "error: VAE_CHECKPOINT is required, e.g." >&2
    echo "  \${SCRATCH}/windinet/finetune_vae_outputs_sng_pvc/<run>/checkpoints/vae_shockwave_best.safetensors" >&2
    exit 1
fi

DEFAULT_OUTPUT_NAME=$(basename "$(dirname "$(dirname "${VAE_CHECKPOINT}")")")
OUTPUT_NAME=${1:-${DEFAULT_OUTPUT_NAME}}
DATA_ROOT=${2:-"${SCRATCH}/windinet/euler_mq_dataset/256x256_ds/train.h5"}
MAX_SAMPLES=${MAX_SAMPLES:-0}
EVAL_SIMS=${EVAL_SIMS:-675}
SPLIT_SEED=${SPLIT_SEED:-42}
OUT_DIR="${SCRATCH}/windinet/dit_preprocessed/${OUTPUT_NAME}"

echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUTPUT_NAME: ${OUTPUT_NAME} -> ${OUT_DIR}"

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${VAE_CHECKPOINT}" "${OUT_DIR}" >> log_finetuning_vae/sng_pvc/INDEX.tsv

EXTRA_ARGS=()
if [[ "${MAX_SAMPLES}" != "0" ]]; then
    EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
elif [[ "${EVAL_SIMS}" != "0" ]]; then
    EXTRA_ARGS+=(--eval-sims "${EVAL_SIMS}" --split-seed "${SPLIT_SEED}")
fi

python scripts/preprocess_dataset.py \
    "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --inflate-checkpoint "${VAE_CHECKPOINT}" \
    "${EXTRA_ARGS[@]}"

# Mirror just the manifests (small, git-trackable) back onto the repo
# checkout -- NOT latents/ or val/latents/ (the bulk of the output, stays on
# $SCRATCH and is read by train_dit.sbatch from the same path).
LOCAL_DIR="finetune_vae_outputs/sng_pvc/dit_preprocessed/${OUTPUT_NAME}"
mkdir -p "${LOCAL_DIR}"
cp "${OUT_DIR}/normalization.json" "${LOCAL_DIR}/"
[ -f "${OUT_DIR}/split_manifest.json" ] && cp "${OUT_DIR}/split_manifest.json" "${LOCAL_DIR}/"
echo "Mirrored manifests to ${LOCAL_DIR}"

echo "END TIME: $(date)"
