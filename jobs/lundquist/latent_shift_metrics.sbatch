#!/bin/bash
#SBATCH --job-name=wh_latent_shift
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# 1-GPU launcher for scripts/latent_shift_metrics.py -- paired z_before
# (zero-init, freshly-inflated) vs z_after (finetuned) latent-shift diagnostic,
# see that script's own docstring and latent_space_shift_measure.md for the
# full protocol (PhD-advisor-sourced). Encode-only, no gradients, no
# accelerate launch needed -- same reasoning as
# jobs/sng_pvc/latent_stats.sbatch, which this mirrors.
#
# No cluster-config patching needed here (unlike the sng_pvc version of this
# job): CONFIG defaults to a lundquist run's own training_config.yaml, whose
# data.data_root is already a lundquist-native /local/disk/... path.
#
# UNVERIFIED (2026-08-17): this assumes
# /local/disk/hramachandran/work/wh_work/windinet/euler_mq_dataset/128x128_ds/test.h5
# exists as a sibling of that dir's train.h5 -- confirm with `ls` on the
# cluster before submitting. If it's missing, the job dies immediately on
# ShockWaveDataset(Path(test_h5)) with no partial output.
#
# Usage (all args optional):
#   sbatch jobs/lundquist/latent_shift_metrics.sbatch [config_yaml] [checkpoint] [test_h5] [num_samples]
#
# Submit with:   sbatch jobs/lundquist/latent_shift_metrics.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/wh_latent_shift_<jobid>.log
# Cancel with:   scancel <jobid>

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

CONFIG=${1:-finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep40_kl1e7/training_config.yaml}
CHECKPOINT=${2:-finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_ep40_kl1e7/checkpoints/vae_shockwave_best.safetensors}
TEST_H5=${3:-euler_mq_dataset/128x128_ds/test.h5}
NUM_SAMPLES=${4:-64}
OUTPUT_JSON="log_finetuning_vae/lundquist/${SLURM_JOB_ID}-latent_shift_metrics.json"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "config=${CONFIG} checkpoint=${CHECKPOINT} test_h5=${TEST_H5} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON}"

"${PYTHON}" scripts/latent_shift_metrics.py "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --test-h5 "${TEST_H5}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
