#!/bin/bash
#SBATCH --job-name=dit_train_1gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# DiT training launcher, 1-GPU. Companion to train_dit_2gpu.sbatch (same
# structure, --gres/--cpus-per-task/--num_processes/job-name differ) --
# written for the batch_size throughput sweep (2026-08-30):
# configs/shockwavenet_lundquist_bs_sweep_bs{1,2,4,8}.yaml, each submitted
# with this script to see how many optimizer steps a fixed 2h window buys
# at each batch_size. --time is hardcoded to 02:00:00 (not left overridable
# like the sng_pvc/lrz_ai queue-wait test jobs) since a fixed wall-clock
# budget is the whole point of this comparison -- if you reuse this script
# for something else, override with `sbatch --time=<HH:MM:SS> ...`.
#
# --cpus-per-task=6 is NOT a confirmed-safe value the way lrz_ai's 48/92
# are (see jobs/lrz_ai/finetune_vae_2gpu.job's header for how those were
# derived from real sinfo/scontrol output) -- just a reasonable guess
# between train_dit_2gpu.sbatch's 12 (6/GPU) and train_dit.sbatch's 16
# (4/GPU). Lower it if submission fails with a CPU-count error.
#
# PREPROCESSED_NAME is REQUIRED, no default -- same convention as
# train_dit_2gpu.sbatch. $2 (optional) overrides BASE_CONFIG, default
# configs/shockwavenet_lundquist_smoketest.yaml.
#
# Submit with:
#   sbatch jobs/lundquist/train_dit_1gpu.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]
#
#   # batch_size throughput sweep (the reason this script exists):
#   sbatch jobs/lundquist/train_dit_1gpu.sbatch smoketest configs/shockwavenet_lundquist_bs_sweep_bs1.yaml
#   sbatch jobs/lundquist/train_dit_1gpu.sbatch smoketest configs/shockwavenet_lundquist_bs_sweep_bs2.yaml
#   sbatch jobs/lundquist/train_dit_1gpu.sbatch smoketest configs/shockwavenet_lundquist_bs_sweep_bs4.yaml
#   sbatch jobs/lundquist/train_dit_1gpu.sbatch smoketest configs/shockwavenet_lundquist_bs_sweep_bs8.yaml
#
# Read the step count each arm reached from its own log
# (logs/lundquist/dit_train_1gpu_<jobid>.log) -- dit_trainer.py
# logs "[train] step={global_step} loss=... lr=..." every step, so:
#   grep -oP '\[train\] step=\K[0-9]+' logs/lundquist/dit_train_1gpu_<jobid>.log | tail -1
# gives the last completed step -- see
# configs/shockwavenet_lundquist_bs_sweep_bs1.yaml's header for why
# checkpoints.interval was raised, so this is read from the log, not from
# how many checkpoints got written.

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [[ -z "${1:-}" ]]; then
    echo "error: PREPROCESSED_NAME is required -- the <name> preprocess_dit_data.sbatch wrote to" >&2
    echo "  usage: sbatch jobs/lundquist/train_dit_1gpu.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]" >&2
    exit 1
fi
PREPROCESSED_NAME=$1
BASE_CONFIG=${2:-configs/shockwavenet_lundquist_smoketest.yaml}
RUN_CONFIG=$(mktemp /tmp/dit_train_1gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_NAME: ${PREPROCESSED_NAME}"
echo "BASE_CONFIG: ${BASE_CONFIG}"

# Substitute the __PREPROCESSED_ROOT__ placeholder for the real path, same
# pattern as train_dit_2gpu.sbatch.
"${PYTHON}" - "${BASE_CONFIG}" "${RUN_CONFIG}" "${PREPROCESSED_NAME}" <<'PY'
import sys, yaml
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src))
root = f"dit_preprocessed/{name}"
cfg["data"]["preprocessed_data_root"] = cfg["data"]["preprocessed_data_root"].replace("__PREPROCESSED_ROOT__", root)
if cfg.get("validation", {}).get("data_root"):
    cfg["validation"]["data_root"] = cfg["validation"]["data_root"].replace("__PREPROCESSED_ROOT__", root)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"preprocessed_data_root={cfg['data']['preprocessed_data_root']} output_dir={cfg['output_dir']}")
PY

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${PREPROCESSED_NAME}" >> logs/lundquist/INDEX.tsv

"${PYTHON}" -m accelerate.commands.launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    scripts/train.py "${RUN_CONFIG}"
