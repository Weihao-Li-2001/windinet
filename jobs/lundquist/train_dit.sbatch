#!/bin/bash
#SBATCH --job-name=shockwave_dit
#SBATCH --partition=debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# Batch submission for the ShockWaveNet DiT run.
#
# Submit with:   sbatch jobs/lundquist/train_dit.sbatch
# Watch with:    tail -f logs/lundquist/train_dit_<jobid>.log
# Cancel with:   scancel <jobid>
#
# Why sbatch rather than running accelerate directly: this is a multi-hour run,
# and a direct launch dies with the shell that started it. That is exactly how
# the VAE run in outputs/vae_inflate4_resume2 ended -- SIGTERM part-way through
# epoch 7, which is why a resume3 config had to exist at all.
#
# NOTE: the partition's 12h limit is a hard kill. Check the step rate in the
# first few minutes; if 10000 steps will not fit, lower optimization.steps
# rather than relying on a resume -- LtxvTrainer._load_checkpoint restores the
# transformer weights only, not the optimizer or LR scheduler state, so a
# resumed run restarts the cosine schedule from the top.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

# See the note in jobs/lundquist/finetune_vae_4gpu.sbatch -- keeps logs/lundquist/ self-describing.
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "configs/shockwavenet.yaml" "-" >> logs/lundquist/INDEX.tsv

# --num_processes must match --gres=gpu:N above, and the config's
# checkpoints.interval assumes it (steps_per_epoch = 3825 / (4 * 4 * N)).
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train.py configs/shockwavenet.yaml
