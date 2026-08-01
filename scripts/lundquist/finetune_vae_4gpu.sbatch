#!/bin/bash
#SBATCH --job-name=wh_finetune_vae
#SBATCH --partition=debug
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --output=log_lundquist/%x_%j.log
#SBATCH --error=log_lundquist/%x_%j.log

# 4-GPU variant of scripts/lundquist/finetune_vae_6gpu.sbatch.
#
# Submit with:   sbatch scripts/lundquist/finetune_vae_4gpu.sbatch
# Watch with:    tail -f log_lundquist/wh_finetune_vae_<jobid>.log
#
# THE COUPLING (see vae_trainer.py:499). batch_size is 1 and the prepared
# loader is sharded, so each rank sees ceil(3825/4) = 957 batches:
#
#     effective batch  = 4 ranks x 1 x accum 8            = 32
#     steps_per_epoch  = ceil(957 / 8)                    = 120
#     total_opt_steps  = 10 epochs x 120                  = 1200
#
# That is byte-identical to the outputs/vae_inflate4_more and vae_meaninit
# runs, so val_vrmse from this job is directly comparable to their 0.10489 /
# 0.10177. The 6-GPU job is not (accum 5 -> batch 30, 1280 steps).
#
# The baseline config ships gradient_accumulation_steps: 8, so this
# script patches it to 8 rather than forking another checked-in YAML. The
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
# 8h is deliberate slack, not a fitted budget.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# Defaults to the baseline config (EXPERIMENTS.md row 3); pass any other
# config as $1, e.g.
#   sbatch scripts/lundquist/finetune_vae_4gpu.sbatch configs/<your-config>.yaml
BASE_CONFIG=${1:-configs/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_4gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys, yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="lundquist", num_processes=4, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"4 GPUs x accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch 32; output_dir {cfg['output_dir']}")
PY

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"

# Make log_lundquist/ self-describing: one row per job mapping the log file to
# the config it ran and the output_dir it wrote. Without this, recovering
# "which log belongs to which run" means grepping every log by hand.
OUT_DIR=$("${PYTHON}" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_lundquist/INDEX.tsv

# Mixed precision is NOT passed here on purpose: VaeTrainer builds its own
# Accelerator from acceleration.mixed_precision_mode (bf16) in the config.
"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 4 \
    scripts/finetune_vae.py "${RUN_CONFIG}"
