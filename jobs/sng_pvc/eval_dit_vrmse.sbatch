#!/bin/bash
#SBATCH --job-name=wh_eval_dit_vrmse
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err
# Same broken-scratch-mount node jobs/sng_pvc/latent_stats.sbatch excludes
# (see that file's own note) -- this is what killed job 529872 (FileNotFound
# on split_manifest.json, a file that reads fine from every other node).
# Remove once LRZ fixes the mount, not before.
#SBATCH --exclude=i20r02c04s06

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
#   sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_VIS_SAMPLES] [NUM_INFERENCE_STEPS]
#
# PREPROCESSED_DATA_ROOT: e.g.
#   /hppfs/scratch/0D/go76fuz2/windinet/dit_preprocessed/finetune_vae_whole_structure_baseline_256res
#   (gives both the h5 path and the exact held-out val_ids via that dir's
#   split_manifest.json -- see preprocess_dataset.py's own comment on why
#   that file exists).
# DIT_CHECKPOINT: full path to the exact model_weights_step_*.safetensors
#   file (not the checkpoints/ dir). The sibling scalar_embedding_step_*
#   .safetensors checkpoint is derived automatically (same step, see
#   dit_trainer.py's own save-path convention).
# VAE_CHECKPOINT: full path to the vae_shockwave_best.safetensors this
#   PREPROCESSED_DATA_ROOT was encoded with.
# NUM_SAMPLES: optional, default 50 (of up to 675 held-out sims).
# SAVE_VIS_SAMPLES: optional, default 3 -- for that many samples, chosen to
#   span the evaluated set's gamma range (min/median/max for 3, evenly-
#   spaced quantiles otherwise -- see scripts/eval_dit_vrmse.py's own
#   pick_gamma_spread_ids), renders GT/Prediction/Residual panels at
#   frames 25/50/75/100 (same convention as the VAE/DiT trainers' own
#   periodic visualization) for BOTH passes, under
#   <OUT_DIR>/visualizations/{vae_only,vae_dit}/<sample_id>/. These
#   PNGs are gitignored (logs/**/*.png) -- pull them off the cluster
#   directly (scp/rsync) rather than expecting them in the repo.
# NUM_INFERENCE_STEPS: optional, default unset -- falls back to
#   configs/dit/inference_dit.yaml's own num_inference_steps (20). Pass this
#   to test whether a finer ODE discretization at sampling time recovers
#   rollout quality without retraining (exposure-bias / step-accumulation
#   diagnostic) -- same checkpoint, only the sampler's step count changes.

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
    echo "  usage: sbatch jobs/sng_pvc/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_VIS_SAMPLES]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-50}
SAVE_VIS_SAMPLES=${5:-3}
NUM_INFERENCE_STEPS=${6:-}

# Sibling file, same step (or same "best" slot) -- see scripts/eval_dit_vrmse.py's
# own docstring and jobs/lrz_ai/inference_dit.job's identical convention. NOT
# .state.pt (that's optimizer/scheduler/RNG state, saved by
# _save_training_state -- ScalarEmbedding has its own separate safetensors
# file, see dit_trainer.py's _save_checkpoint). Substitutes on "model_weights_"
# rather than "model_weights_step_" so this also covers the
# model_weights_best.safetensors / scalar_embedding_best.safetensors pair
# dit_trainer.py now saves (see its "New best checkpoint" log line) -- the
# old model_weights_step_-only pattern silently no-ops on a _best path,
# pointing SCALAR_CHECKPOINT at the DiT weights file instead of the scalar
# embedding one.
SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}
NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/sng_pvc/eval_dit_vrmse/${SLURM_JOB_ID}"

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

python scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \
    --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
    --checkpoint "${DIT_CHECKPOINT}" \
    --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
    --vae_checkpoint "${VAE_CHECKPOINT}" \
    --normalization "${NORMALIZATION}" \
    --num_samples "${NUM_SAMPLES}" \
    --save_vis_samples "${SAVE_VIS_SAMPLES}" \
    --out_dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "END TIME: $(date)"
