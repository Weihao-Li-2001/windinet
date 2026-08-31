#!/bin/bash
#SBATCH --job-name=wh_eval_dit_latent_vrmse
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err
# Same broken-scratch-mount node jobs/sng_pvc/latent_stats.sbatch excludes
# (see that file's own note). Remove once LRZ fixes the mount, not before.
#SBATCH --exclude=i20r02c04s06

# Single-tile launcher for scripts/eval_dit_latent_vrmse.py -- companion to
# jobs/sng_pvc/eval_dit_vrmse.sbatch, same checkpoint-pair arguments, but
# compares the DiT's rolled-out latent directly against the VAE encoder's
# ground-truth latent (no decode step on the predicted side) instead of
# comparing decoded pixel fields. Answers "how good is the DiT's own
# forecast" without the VAE decoder's reconstruction fidelity mixed in --
# see that script's own docstring for the full motivation (the DiT's
# training loss, a flow-matching loss at random noise levels, does not
# directly report this).
#
# --time=04:00:00: same uncalibrated guess as eval_dit_vrmse.sbatch (this
# does the exact same rollout, just one fewer decode on the ground-truth
# side already computed either way, and one guaranteed decode instead of a
# conditional one -- expect near-identical per-sample cost). Start with a
# small NUM_SAMPLES to get a real per-sample time before committing to 675.
#
# Usage:
#   sbatch jobs/sng_pvc/eval_dit_latent_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [NUM_INFERENCE_STEPS]
#
# Arguments: identical meaning to eval_dit_vrmse.sbatch's own (see that
# file's header) minus SAVE_VIS_SAMPLES, which this script does not support
# -- it has no visualization output, only the latent/pixel VRMSE numbers.

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/sng_pvc/eval_dit_latent_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [NUM_INFERENCE_STEPS]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-50}
NUM_INFERENCE_STEPS=${5:-}

# Same "model_weights_" / "scalar_embedding_" substitution as
# eval_dit_vrmse.sbatch -- covers both model_weights_step_NNNNN.safetensors
# and model_weights_best.safetensors.
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}
NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/sng_pvc/eval_dit_latent_vrmse/${SLURM_JOB_ID}"

echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "NUM_INFERENCE_STEPS: ${NUM_INFERENCE_STEPS:-<config default>}"
echo "OUT_DIR: ${OUT_DIR}"

EXTRA_ARGS=()
if [ -n "${NUM_INFERENCE_STEPS}" ]; then
    EXTRA_ARGS+=(--num_inference_steps "${NUM_INFERENCE_STEPS}")
fi

python scripts/eval_dit_latent_vrmse.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "END TIME: $(date)"
