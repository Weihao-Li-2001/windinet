#!/bin/bash
#SBATCH --job-name=wh_bs_sweep
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log
#SBATCH --error=log_finetuning_vae/lundquist/%x_%j.log

# Batch-size sweep, Open Question 23 (EXPERIMENTS.md). Tests whether
# optimization.batch_size (per-GPU micro-batch) changes results or only
# wall-clock/OOM feasibility, at a FIXED effective_batch=32 -- same
# optimizer-step count and LR schedule as every other run in this ledger,
# see windinet/cluster_config.py's derivation. 2-GPU variant of
# finetune_vae_2gpu.sbatch, parameterized by BATCH_SIZE instead of
# hard-coding batch_size=1 / accum=16.
#
# Submit with:   BATCH_SIZE=<1|2|4|8|16> sbatch jobs/lundquist/finetune_vae_batchsize.sbatch
# Watch with:    tail -f log_finetuning_vae/lundquist/wh_bs_sweep_<jobid>.log
#
# gradient_accumulation_steps is re-derived by patch_config_for_cluster so
# 2 ranks x BATCH_SIZE x accum always lands on effective_batch 32 -- e.g.
# BATCH_SIZE=2 -> accum 8, BATCH_SIZE=16 -> accum 1. Total optimizer steps
# (~1200 over 10 epochs, ~2160 over this config's 18) and the LR schedule
# are unchanged across arms, so any val_vrmse difference is a real finding,
# not a step-count confound.
#
# THIS IS A REAL RUN, NOT A THROWAWAY DIAGNOSTIC: full 18 epochs, normal
# checkpointing/visualization, output committed to
# finetune_vae_outputs/lundquist/finetune_vae_whole_structure_baseline_bs<N>/
# per protocol. BATCH_SIZE=1 also fills in lundquist's own missing
# reference point for this config -- previously only run on sng_pvc (job
# 523577, val_vrmse 0.08360).
#
# IF A GIVEN BATCH_SIZE OOMs, THAT IS THE ANSWER FOR THAT ARM, NOT A BUG --
# see Open Question 23's kill criterion. Slurm/CUDA will kill the job; check
# the .log for "CUDA out of memory" and record which sizes fail here before
# running the same sweep on sng_pvc/lrz_ai (different GPU memory budgets).
#
# TIME: 10h matches finetune_vae_2gpu.sbatch's own budget for this config
# at batch_size=1 (measured ~6.5h for 18 epochs on sng_pvc, lundquist
# historically a bit slower per epoch). Larger batch_size may finish faster
# or slower -- not assumed either way, hence no per-arm time override.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

BATCH_SIZE=${BATCH_SIZE:?"set BATCH_SIZE (1, 2, 4, 8, or 16), e.g. BATCH_SIZE=4 sbatch jobs/lundquist/finetune_vae_batchsize.sbatch"}

# Defaults to the ep18 whole-structure baseline (Open Question 23's test
# subject); pass any other config as $1.
BASE_CONFIG=${1:-configs/finetune_vae/archive/done/finetune_vae_whole_structure_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_bs_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${BATCH_SIZE}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst, batch_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = batch_size
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=2, output_suffix=f"_bs{batch_size}")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"2 GPUs x batch_size {batch_size} x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 32; output_dir {cfg['output_dir']}"
)
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES} BATCH_SIZE=${BATCH_SIZE}"

# Same INDEX.tsv convention as the other lundquist launchers, tagged with
# BATCH_SIZE so grepping the index alone tells arms apart.
OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}_bs${BATCH_SIZE}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/lundquist/INDEX.tsv

# Mixed precision is NOT passed here on purpose: VaeTrainer builds its own
# Accelerator from acceleration.mixed_precision_mode (bf16) in the config.
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 2 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
