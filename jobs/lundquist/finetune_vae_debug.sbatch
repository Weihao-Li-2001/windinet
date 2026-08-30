#!/bin/bash
#SBATCH --job-name=wh_debug
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# 1-GPU debug launcher, default config configs/finetune_vae/finetune_vae_overfit.yaml
# (the overfit-8 capacity diagnostic -- see that file's header and
# EXPERIMENTS.md Open Questions #2). Not for real training runs: 1 GPU is
# deliberate there (memorization needs optimizer-step COUNT, not batch size),
# and this launcher is only meant to run on the debug partition.
#
# Submit with:   sbatch jobs/lundquist/finetune_vae_debug.sbatch
# Watch with:    tail -f logs/lundquist/wh_debug_<jobid>.log
#
# EFFECTIVE BATCH IS PATCHED TO 1, NOT THE CLUSTER DEFAULT (32). The other
# lundquist scripts call patch_config_for_cluster() with its default
# effective_batch, deriving gradient_accumulation_steps to keep runs
# comparable to each other. finetune_vae_overfit.yaml relies on
# gradient_accumulation_steps: 1 ("one optimizer step per sim", see its
# header) to hit its optimizer-step-count target; letting the default kick in
# would force accum=32 and silently break the diagnostic. Pass
# effective_batch=1 here so 1 rank x batch_size 1 x accum 1 stays 1.
#
# TIME: ~32 min measured for 25 epochs / 1000 steps. Attempt 2 of the overfit
# diagnostic runs 50 epochs / 2000 steps at a 4x lower LR, so ~64 min -- hence
# 2h here, not the 1h that fitted attempt 1.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

# Defaults to the overfit-8 diagnostic; pass any other config as $1, e.g.
#   sbatch jobs/lundquist/finetune_vae_debug.sbatch configs/finetune_vae/<your-config>.yaml
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_overfit.yaml}
RUN_CONFIG=$(mktemp /tmp/wh_debug_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=1, output_suffix="", effective_batch=1)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"1 GPU x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 1; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

# Make logs/lundquist/ self-describing: one row per job mapping
# the log file to the config it ran and the output_dir it wrote. Without this,
# recovering "which log belongs to which run" means grepping every log by hand.
OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/lundquist/INDEX.tsv

# Mixed precision is NOT passed here on purpose: VaeTrainer builds its own
# Accelerator from acceleration.mixed_precision_mode (bf16) in the config.
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 1 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
