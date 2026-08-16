#!/bin/bash
#SBATCH --job-name=wh_finetune_vae_4gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# 4-GPU launcher for this cluster.
#
# Submit with:   sbatch jobs/lundquist/finetune_vae_4gpu.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/wh_finetune_vae_<jobid>.log
#
# THE COUPLING (see vae_trainer.py:499). batch_size is 8 and the prepared
# loader is sharded, so each rank sees ceil(3825/4) = 957 samples:
#
#     effective batch  = 4 ranks x 8 x accum 1             = 32
#     steps_per_epoch  = ceil(957 / 8)                     = 120
#     total_opt_steps  = 10 epochs x 120                   = 1200
#
# Same step count as the old batch_size=1/accum=8 split (steps_per_epoch
# depends only on batch_size x accum = 8 either way, see
# finetune_vae_2gpu.sbatch's own note on this) -- only wall clock should
# differ, from fewer/larger batched forward-backward calls instead of 8
# small accumulation steps.
#
# batch_size 8 (2026-08-17): mirrors finetune_vae_2gpu.sbatch's batch_size=16
# change (2026-08-16) -- the largest batch_size that still divides
# effective_batch=32 across 4 GPUs (accum=1). UNVALIDATED AT 4 GPUS: Open
# Question 23 (EXPERIMENTS.md) only measured batch_size=1 vs 16 at 2 GPUs
# (job 21989 vs 21991, ~18.7% faster, val_vrmse ~1.53% apart, judged inside
# the "real effect" bar there) -- applied here on the same reasoning, not on
# a 4-GPU measurement of its own. Watch val_vrmse against the batch_size=1
# baseline (job 21992, 0.076314) the same way that comparison was made.
#
# The baseline config ships gradient_accumulation_steps: 8 and batch_size: 1;
# this script's patch step overrides both (batch_size explicitly, accum via
# patch_config_for_cluster) rather than forking another checked-in YAML. The
# trainer writes the resolved config to output_dir/training_config.yaml, so
# provenance is preserved.
#
# --cpus-per-task=24 covers 4 ranks x num_dataloader_workers: 4 = 16 loader
# workers plus the ranks themselves.
#
# TIME: measured from the live 6-GPU run (job 21618) -- 12 min/epoch at 638
# samples/rank. The training half scales with per-rank samples, the rank-0-only
# eval over 675 sims does not, so 957/rank here lands at ~16-18 min/epoch, i.e.
# ~3h for 10 epochs. (Ignore the 29 min/epoch figure in the older sbatch header;
# that run was on a slower box and overestimates this hardware by ~2x.)
# --time above is the `debug` partition's 12h hard kill (confirm with
# `scontrol show partition debug`), not a fitted budget -- deliberate slack
# for a 10-epoch run, but a longer-epoch config (e.g. the ep40 configs) can
# still eat most or all of it; check the actual per-epoch rate after epoch
# 1-2 before trusting a long run to finish inside 12h.

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
#   sbatch jobs/lundquist/finetune_vae_4gpu.sbatch configs/finetune_vae/<your-config>.yaml
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_4gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = 8  # production default, see header comment above
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=4, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"4 GPUs x batch_size {cfg['optimization']['batch_size']} x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 32; output_dir {cfg['output_dir']}")
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
    --num_processes 4 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
