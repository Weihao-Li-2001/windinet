#!/bin/bash
#SBATCH --job-name=dit_train_2gpu
#SBATCH --partition=debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=logs/lundquist/%x_%j.log
#SBATCH --error=logs/lundquist/%x_%j.log

# DiT training launcher, 2-GPU. Companion to preprocess_dit_data.sbatch --
# run that first, then this job trains on its output. Separate from the
# existing train_dit.sbatch (4-GPU, hardcoded to configs/shockwavenet.yaml +
# euler_mq_preprocessed/ -- the real production run): this one is
# parameterized so it can point at ANY dit_preprocessed/<name> the encode
# job produced, e.g. the smoke-test encode that regression-tests the
# 2026-08-27 DDP-wrap-order fix in dit_trainer.py (see
# configs/shockwavenet_lundquist_smoketest.yaml's header).
#
# PREPROCESSED_NAME is REQUIRED, no default: must match the <name>
# preprocess_dit_data.sbatch wrote under dit_preprocessed/.
# $2 (optional) overrides BASE_CONFIG, default
# configs/shockwavenet_lundquist_smoketest.yaml -- pass
# configs/shockwavenet.yaml (or any other config using the
# __PREPROCESSED_ROOT__ placeholder in data.preprocessed_data_root /
# validation.data_root) for a real run instead of the smoke test.
#
# Submit with:
#   sbatch jobs/lundquist/train_dit_2gpu.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]
#
#   # smoke test:
#   sbatch jobs/lundquist/train_dit_2gpu.sbatch smoketest

set -euo pipefail

PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python
cd /local/disk/hramachandran/work/wh_work/windinet

# LTX base weights ship with the repo (pretrained/hub), not ~/.cache/huggingface.
export HF_HUB_CACHE="${PWD}/pretrained/hub"

if [[ -z "${1:-}" ]]; then
    echo "error: PREPROCESSED_NAME is required -- the <name> preprocess_dit_data.sbatch wrote to" >&2
    echo "  usage: sbatch jobs/lundquist/train_dit_2gpu.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]" >&2
    exit 1
fi
PREPROCESSED_NAME=$1
BASE_CONFIG=${2:-configs/shockwavenet_lundquist_smoketest.yaml}
RUN_CONFIG=$(mktemp /tmp/dit_train_2gpu_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

echo "job=${SLURM_JOB_ID} node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES}"
echo "PREPROCESSED_NAME: ${PREPROCESSED_NAME}"
echo "BASE_CONFIG: ${BASE_CONFIG}"

# Substitute the __PREPROCESSED_ROOT__ placeholder for the real path, same
# pattern as jobs/lrz_ai/train_dit.job (but a plain repo-relative dir here,
# not a /mnt/result container mount).
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
    --num_processes 2 \
    --mixed_precision bf16 \
    scripts/train.py "${RUN_CONFIG}"
