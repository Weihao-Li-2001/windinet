#!/bin/bash
#SBATCH --job-name=shockwave_dit_smoketest
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# Throwaway smoke test: preprocess a handful of samples with WHATEVER VAE
# checkpoint is on disk, then run a few DiT training steps under 2-GPU DDP
# with gradient checkpointing on. Only question this answers: does the
# training loop start and run without crashing (regression check for the
# 2026-08-27 dit_trainer.py fix -- enable_gradient_checkpointing() was being
# called after accelerate wrapped the transformer in DistributedDataParallel,
# which doesn't have that method). Not a real experiment -- output is not
# meant to be kept, val_vrmse is not comparable to anything in EXPERIMENTS.md.
#
# Submit with:   sbatch jobs/lundquist/dit_smoketest.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/shockwave_dit_smoketest_<jobid>.log

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

export HF_HUB_CACHE="${PWD}/pretrained/hub"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

CKPT=$(find finetune_vae_outputs/lundquist -name "*.safetensors" 2>/dev/null | grep -v scalar_embedding | head -1)
if [[ -z "${CKPT}" ]]; then
    echo "No VAE checkpoint found under finetune_vae_outputs/lundquist -- nothing to preprocess with." >&2
    exit 1
fi
echo "Using VAE checkpoint: ${CKPT}"

rm -rf dit_smoketest_preprocessed dit_smoketest_output

"${PYTHON}" scripts/preprocess_dataset.py \
    euler_mq_dataset/128x128_ds/train.h5 \
    --output-dir dit_smoketest_preprocessed \
    --inflate-checkpoint "${CKPT}" \
    --max-samples 24

"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/train.py configs/shockwavenet_lundquist_smoketest.yaml

echo "Smoke test finished without a crash."
