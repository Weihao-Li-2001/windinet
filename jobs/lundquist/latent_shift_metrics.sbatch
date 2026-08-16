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
# job): CONFIG defaults to the checked-in SOURCE config
# (configs/finetune_vae/...), not the run's resolved training_config.yaml --
# data.data_root there is already a lundquist-native /local/disk/... path, and
# every value latent_shift_metrics.py actually reads (adapter/data/loss) is
# unaffected by the launcher's accum-steps patch, so the two are equivalent
# for this purpose. Pass the resolved training_config.yaml explicitly as $1
# instead if you want the exact-provenance artifact -- but only once it was
# written by a trainer build that includes the model_dump(mode="json") /
# normalization_stats_file fixes (see below); job 21992's own
# training_config.yaml predates those and will still fail to parse.
#
# BUG FIXED (2026-08-17): the first attempt at this job (21994) crashed
# reading finetune_vae_whole_structure_baseline_ep40_kl1e7/training_config.yaml
# with yaml.safe_load -- two bugs in VaeTrainer._save_config/VaeDataConfig,
# both now fixed in windinet/config.py and windinet/training/vae_trainer.py:
#   1. model_dump() (python mode) left model.model_source as a LtxvModelVersion
#      enum instance; plain yaml.dump wrote that as a
#      "!!python/object/apply:..." tag that yaml.safe_load can't construct.
#      Fixed by dumping with model_dump(mode="json") instead.
#   2. VaeDataConfig's normalization_stats_file -> channel_mean/channel_std
#      resolution left BOTH fields set afterward, so reloading a dumped
#      config re-triggered the "set either ... not both" validator error.
#      Fixed by clearing normalization_stats_file once resolved.
# Only new runs get a training_config.yaml that reloads cleanly -- existing
# ones (like job 21992's) are unaffected by this fix, hence defaulting to the
# source config above instead of trying to patch that file in place.
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

CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep40_kl1e7.yaml}
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
