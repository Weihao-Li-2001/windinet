#!/bin/bash
#SBATCH --job-name=eval_dit_vrmse
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Single-GPU launcher for scripts/eval_dit_vrmse.py on lundquist -- compares
# VAE-only reconstruction vs full VAE+DiT rollout on the SAME held-out sims
# the VAE was validated on, using the SAME val_vrmse formula (normalized-
# space variance-normalized RMSE), so the two are directly comparable to
# each other and to the already-committed VAE val_vrmse numbers. Reports
# BOTH decoded pixel-space VRMSE and raw latent-space VRMSE/RMSE. See
# scripts/eval_dit_vrmse.py's own docstring for the exact methodology.
# Mirrors jobs/sng_pvc/eval_dit_vrmse.sbatch (same script, same CLI) with
# this cluster's plain-conda conventions (PYTHON path, cd, no micromamba/
# module load, no distributed launch -- inference, batch_size 1, forward-
# only, same reasoning as jobs/lundquist/preprocess_dit_data.sbatch).
#
# --time=02:00:00 is an UNCALIBRATED GUESS -- no eval_dit_vrmse run has
# been timed on lundquist yet. sng_pvc measured ~10.7s/sample (675 samples
# in ~2h), so the default 50 samples below should be well under 15 min once
# the ~2-5 min model-load overhead is added; the 2h budget is headroom for
# an unknown lundquist GPU being slower, not an expected runtime. Bump
# --time if a larger NUM_SAMPLES run times out.
#
# Checkpoint can be from a run that HAS NOT FINISHED -- this script does
# not require or check a "final" checkpoint, it just evaluates whatever
# step you point it at.
#
# Usage:
#   sbatch jobs/lundquist/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_VIS_SAMPLES] [NUM_INFERENCE_STEPS]
#
# PREPROCESSED_DATA_ROOT: e.g. dit_preprocessed/finetune_vae_whole_structure_baseline_ep20_kl0
#   (gives both the h5 path and the exact held-out val_ids via that dir's
#   split_manifest.json).
# DIT_CHECKPOINT: full path to the exact model_weights_step_*.safetensors
#   file (not the checkpoints/ dir). The sibling scalar_embedding_step_*
#   checkpoint is derived automatically (same step). Pass the literal
#   string 'untrained' for the DiT-never-finetuned control
#   (scripts/eval_dit_vrmse.py --untrained_dit) -- stock pretrained
#   transformer, freshly random-initialized ScalarEmbedding, no checkpoint
#   files needed.
# VAE_CHECKPOINT: full path to the vae_shockwave_best.safetensors this
#   PREPROCESSED_DATA_ROOT was encoded with -- read it off the DiT run's
#   own latent_provenance.json ("vae_checkpoint" field) rather than
#   guessing, to avoid a silent latent-space mismatch.
# NUM_SAMPLES: optional, default 50 (of up to however many held-out sims
#   the manifest records).
# SAVE_VIS_SAMPLES: optional, default 3 -- renders GT/Prediction/Residual
#   panels for that many samples (span the gamma range) under
#   <OUT_DIR>/visualizations/{vae_only,vae_dit}/<sample_id>/. Gitignored
#   (logs/**/*.png) -- pull off the cluster directly if you need them.
# NUM_INFERENCE_STEPS: optional, default unset -- falls back to
#   configs/dit/inference_dit.yaml's own num_inference_steps (20).

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "error: PREPROCESSED_DATA_ROOT, DIT_CHECKPOINT, and VAE_CHECKPOINT are all required" >&2
    echo "  usage: sbatch jobs/lundquist/eval_dit_vrmse.sbatch <PREPROCESSED_DATA_ROOT> <DIT_CHECKPOINT> <VAE_CHECKPOINT> [NUM_SAMPLES] [SAVE_VIS_SAMPLES] [NUM_INFERENCE_STEPS]" >&2
    exit 1
fi
PREPROCESSED_DATA_ROOT=$1
DIT_CHECKPOINT=$2
VAE_CHECKPOINT=$3
NUM_SAMPLES=${4:-50}
SAVE_VIS_SAMPLES=${5:-3}
NUM_INFERENCE_STEPS=${6:-}

NORMALIZATION=$(dirname "$(dirname "${VAE_CHECKPOINT}")")/training_config.yaml
OUT_DIR="logs/lundquist/eval_dit_vrmse/${SLURM_JOB_ID}"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_DATA_ROOT: ${PREPROCESSED_DATA_ROOT}"
echo "DIT_CHECKPOINT: ${DIT_CHECKPOINT}"
echo "VAE_CHECKPOINT: ${VAE_CHECKPOINT}"
echo "NORMALIZATION: ${NORMALIZATION}"
echo "NUM_SAMPLES: ${NUM_SAMPLES}"
echo "NUM_INFERENCE_STEPS: ${NUM_INFERENCE_STEPS:-<config default>}"
echo "OUT_DIR: ${OUT_DIR}"

EXTRA_ARGS=()
if [ -n "${NUM_INFERENCE_STEPS}" ]; then
    EXTRA_ARGS+=(--num_inference_steps "${NUM_INFERENCE_STEPS}")
fi

if [ "${DIT_CHECKPOINT}" = "untrained" ]; then
    echo "DIT_CHECKPOINT=untrained: DiT-never-finetuned control, skipping checkpoint load"
    EXTRA_ARGS+=(--untrained_dit)

    "${PYTHON}" scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \
        --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
        --vae_checkpoint "${VAE_CHECKPOINT}" \
        --normalization "${NORMALIZATION}" \
        --num_samples "${NUM_SAMPLES}" \
        --save_vis_samples "${SAVE_VIS_SAMPLES}" \
        --out_dir "${OUT_DIR}" \
        "${EXTRA_ARGS[@]}"
else
    # Sibling file, same step -- substitutes on "model_weights_" (not
    # "model_weights_step_") so this also covers a model_weights_best.safetensors
    # / scalar_embedding_best.safetensors pair, same convention as
    # jobs/sng_pvc/eval_dit_vrmse.sbatch.
    SCALAR_CHECKPOINT=${DIT_CHECKPOINT/model_weights_/scalar_embedding_}
    echo "SCALAR_CHECKPOINT: ${SCALAR_CHECKPOINT}"

    "${PYTHON}" scripts/eval_dit_vrmse.py configs/dit/inference_dit.yaml \
        --preprocessed_data_root "${PREPROCESSED_DATA_ROOT}" \
        --checkpoint "${DIT_CHECKPOINT}" \
        --scalar_checkpoint "${SCALAR_CHECKPOINT}" \
        --vae_checkpoint "${VAE_CHECKPOINT}" \
        --normalization "${NORMALIZATION}" \
        --num_samples "${NUM_SAMPLES}" \
        --save_vis_samples "${SAVE_VIS_SAMPLES}" \
        --out_dir "${OUT_DIR}" \
        "${EXTRA_ARGS[@]}"
fi
