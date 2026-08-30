#!/bin/bash
#SBATCH --job-name=wh_eval_dit_vrmse
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err

# Single-tile launcher for scripts/eval_dit_vrmse.py -- compares VAE-only
# reconstruction vs full VAE+DiT rollout on the SAME held-out sims the VAE
# was validated on, using the SAME val_vrmse formula (normalized-space
# variance-normalized RMSE), so the two are directly comparable to each
# other and to the already-committed VAE val_vrmse numbers. No distributed
# launch needed -- inference, batch_size 1, forward-only, same reasoning as
# jobs/sng_pvc/latent_stats.sbatch.
#
# --time=04:00:00 is an UNCALIBRATED GUESS (2026-08-30, first run of this
# script) -- no full-validation-set DiT rollout has been timed on sng_pvc
# yet. Budget per sample: one VAE encode+decode (fast) + one 20-step DiT
# denoising rollout on a 2B-param transformer at 896 tokens (lrz_ai's own
# inference_dit.job guesses "seconds, not minutes" per step on an H100;
# sng_pvc's XPU tiles are the ones actually training these checkpoints, so
# use that job's own per-step throughput as a better prior than the H100
# guess). Start with a small NUM_SAMPLES (default 50) to get a real
# per-sample time before committing to the full 675.
#
# Checkpoints passed here can be from a run that HAS NOT FINISHED -- this
# script does not require or check a "final" checkpoint, it just evaluates
# whatever step you point it at.
#
# Usage:
#   sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_NPZ_SAMPLES]
#
# PREPROCESSED_DATA_ROOT: e.g.
#   /hppfs/scratch/0D/go76fuz2/windinet/dit_preprocessed/finetune_vae_whole_structure_baseline_256res
#   (gives both the h5 path and the exact held-out val_ids via that dir's
#   split_manifest.json -- see preprocess_dataset.py's own comment on why
#   that file exists).
# DIT_CHECKPOINT: full path to the exact model_weights_step_*.safetensors
#   file (not the checkpoints/ dir). The sibling *.state.pt scalar-embedding
#   checkpoint is derived automatically (same stem, see dit_trainer.py's
#   own save-path convention).
# VAE_CHECKPOINT: full path to the vae_shockwave_best.safetensors this
#   PREPROCESSED_DATA_ROOT was encoded with.
# NUM_SAMPLES: optional, default 50 (of up to 675 held-out sims).
# SAVE_NPZ_SAMPLES: optional, default 3 -- also saves that many VAE+DiT
#   predictions as .npz for scripts/visualize_dit_predictions.py.

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

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_NPZ_SAMPLES]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-50}
SAVE_NPZ_SAMPLES=${5:-3}

# Sibling file, same stem -- see scripts/eval_dit_vrmse.py's own docstring
# and jobs/lrz_ai/inference_dit.job's identical convention.
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/.safetensors/.state.pt}
NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/sng_pvc/eval_dit_vrmse/${SLURM_JOB_ID}"

echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "OUT_DIR: ${OUT_DIR}"

python scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --save_npz_samples "${SAVE_NPZ_SAMPLES}" \
    --out_dir "${OUT_DIR}"

echo "END TIME: $(date)"
