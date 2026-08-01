#!/bin/bash
#SBATCH --job-name=wh_finetune_vae_6gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# Batch submission for the VAE inflate-mode finetune with the 3x LR tail.
#
# Submit with:   sbatch jobs/lundquist/finetune_vae_6gpu.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/vae_6gpu_<jobid>.log
# Cancel with:   scancel <jobid>
#
# 2- and 4-GPU siblings: jobs/lundquist/finetune_vae_{2,4}gpu.sbatch. All three derive
# gradient_accumulation_steps at submit time instead of relying on whatever the
# config happens to ship, so the GPU count and the accumulation can never drift
# apart in the file that owns them.
#
# Why sbatch rather than running accelerate directly: this is a ~2h run, and a
# direct launch dies with the shell that started it -- that is exactly how the
# 6-epoch resume2 run ended (SIGTERM part-way through epoch 7).
#
# THE COUPLING (see vae_trainer.py:499). batch_size is 1 and the prepared
# loader is sharded, so each rank sees ceil(3825/6) = 638 batches:
#
#     effective batch  = 6 ranks x 1 x accum 5            = 30
#     steps_per_epoch  = ceil(638 / 5)                    = 128
#     total_opt_steps  = 10 epochs x 128                  = 1280
#
# Confirmed against job 21618, which logged
#     LR schedule 'cosine': 1280 steps (50 warmup, ...)
#
# NOTE: 30 / 1280 does NOT match the 4-GPU baselines (32 / 1200), because 32 is
# not divisible by 6. If the point of a run is a direct val_vrmse comparison
# against outputs/vae_inflate4_more (0.10489) or vae_meaninit (0.10177), use the
# 2- or 4-GPU script -- both hit 32 / 1200 exactly. Use this one for throughput.
#
# --cpus-per-task=32 covers 6 ranks x num_dataloader_workers: 4 = 24 loader
# workers plus the ranks themselves.
#
# TIME: measured on job 21618 -- 12 min/epoch at 638 samples/rank, so ~2h for 10
# epochs, well inside the partition's 12h hard kill. (The old header here quoted
# 29 min/epoch from outputs/vae_inflate4_more; that run was on slower hardware
# and overestimates this box by ~2x.)

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# Defaults to the baseline config (EXPERIMENTS.md row 3); pass any other
# config as $1, e.g.
#   sbatch jobs/lundquist/finetune_vae_6gpu.sbatch configs/finetune_vae/<your-config>.yaml
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_6gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
# effective_batch=30, not the cluster default 32: 32 isn't divisible by 6
# ranks. Empty output_suffix like the 2-/4-GPU scripts: all lundquist
# variants intentionally share one output_dir (see cluster_config.py), so
# resubmitting with a different GPU count overwrites the previous run on
# purpose (clean_output_dir: true) rather than keeping per-variant results.
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=6, output_suffix="", effective_batch=30)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"6 GPUs x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 30; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

# Make log_finetuning_vae/lundquist/ self-describing: one row per job mapping the log file to
# the config it ran and the output_dir it wrote. Without this, recovering
# "which log belongs to which run" means grepping every log by hand.
OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/lundquist/INDEX.tsv

# Mixed precision is NOT passed here on purpose: VaeTrainer builds its own
# Accelerator from acceleration.mixed_precision_mode (bf16) in the config.
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 6 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
