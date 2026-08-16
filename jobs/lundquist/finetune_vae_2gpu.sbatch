#!/bin/bash
#SBATCH --job-name=wh_finetune_vae_2gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# 2-GPU launcher for this cluster.
#
# Submit with:   sbatch jobs/lundquist/finetune_vae_2gpu.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/vae_2gpu_<jobid>.log
#
# THE COUPLING (see vae_trainer.py:499). batch_size is 16 (accum 1, see the
# "batch_size 16" note further down) and the prepared loader is sharded, so
# each rank sees ceil(3825/2) = 1913 samples:
#
#     effective batch  = 2 ranks x 16 x accum 1            = 32
#     steps_per_epoch  = ceil(1913 / 16)                   = 120
#     total_opt_steps  = 10 epochs x 120                   = 1200
#
# steps_per_epoch/total_opt_steps depend only on batch_size x accum (16
# either way), not on how that 16 splits between them -- same 32/1200 as
# the 4-GPU reference runs and as batch_size=1/accum=16 would give, so the
# optimization trajectory matches regardless of the split; only wall clock
# differs (see the "batch_size 16" note below for how much).
#
# The baseline config ships gradient_accumulation_steps: 8; this script's
# patch step overrides both batch_size and (via patch_config_for_cluster)
# gradient_accumulation_steps rather than forking another checked-in YAML.
#
# --cpus-per-task=12 covers 2 ranks x num_dataloader_workers: 4 = 8 loader
# workers plus the ranks themselves.
#
# TIME: measured from the live 6-GPU run (job 21618) -- 12 min/epoch at 638
# samples/rank. The training half scales with per-rank samples, the rank-0-only
# eval over 675 sims does not, so 1913/rank here (3x) lands at ~32-36 min/epoch,
# i.e. ~5.5-6h for 10 epochs. That fits the partition's 12h hard kill with
# roughly 2x margin.
#
#   STILL WORTH CHECKING AFTER EPOCH 1. The estimate assumes the eval fraction
#   is small; if an epoch comes in above ~70 min, 10 epochs will not finish --
#   drop optimization.epochs to 8 or move to 4 GPUs, because a 12h SIGKILL
#   mid-epoch loses that epoch entirely.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

# Defaults to the current 128-res baseline (finetune_vae_whole_structure_
# baseline_ep30.yaml, 30ep/1.0x-encoder-LR, val_vrmse 0.078662, job 524322)
# -- was finetune_vae_baseline.yaml (frozen-trunk, superseded), now archived
# to configs/finetune_vae/archive/done/, see EXPERIMENTS.md. Pass any other
# config as $1, e.g.
#   sbatch jobs/lundquist/finetune_vae_2gpu.sbatch configs/finetune_vae/<your-config>.yaml
#
# batch_size 16 (2026-08-16): the largest batch_size that still divides
# effective_batch=32 across 2 GPUs (accum=1) -- adopted as lundquist's
# production default per Open Question 23's lundquist arms (job 21991,
# batch_size=16, val_vrmse 0.084825 vs batch_size=1's 0.083545, ~1.53%
# apart -- above the ~1.1% single-seed noise floor but inside the 2.2%
# "real effect" bar) and a real ~18.7% wall-clock win (8.19h -> 6.66h for
# 18 epochs, job 21989 vs 21991). See EXPERIMENTS.md's "Open Question 23"
# section for the full readout.
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_2gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = 16  # production default, see header comment above
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=2, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"2 GPUs x batch_size {cfg['optimization']['batch_size']} x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 32; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

# Make log_finetuning_vae/lundquist/ self-describing: one row per job mapping
# the log file to the config it ran and the output_dir it wrote. Without this,
# recovering "which log belongs to which run" means grepping every log by hand.
OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/lundquist/INDEX.tsv

# Mixed precision is NOT passed here on purpose: VaeTrainer builds its own
# Accelerator from acceleration.mixed_precision_mode (bf16) in the config.
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 2 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
