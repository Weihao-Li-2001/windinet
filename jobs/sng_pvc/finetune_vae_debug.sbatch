#!/bin/bash
#SBATCH --job-name=wh_debug_vae
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # NOT "test" (30 min cap) -- 50 epochs won't fit
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# 1-XPU-tile debug launcher for the overfit-8 capacity diagnostic (see
# EXPERIMENTS.md Open Question 2 and Open Question 12), the sng_pvc
# counterpart of jobs/lundquist/finetune_vae_debug.sbatch. Does NOT exist as
# of writing this -- created because the first Open Question 12 attempt
# (`sbatch jobs/lundquist/finetune_vae_debug.sbatch ...`) hit
# `sbatch: error: Invalid generic resource (gres) specification` on this
# account/login node (the script's `--gres=gpu:1` doesn't resolve here), and
# because sng_pvc is where this project's real runs already live -- see
# jobs/sng_pvc/finetune_vae.sbatch.
#
# WHY THIS SCRIPT EXISTS SEPARATELY FROM finetune_vae.sbatch /
# finetune_vae_diag.sbatch: both of those hard-code NUM_XPUS=8 and pass
# num_processes=8 into patch_config_for_cluster(), which derives
# gradient_accumulation_steps to hit the cluster's effective_batch=32
# default (windinet/cluster_config.py). The overfit-8 diagnostic's entire
# design depends on ONE optimizer step per sim (gradient_accumulation_steps
# staying at 1, see finetune_vae_overfit.yaml's own header) -- submitting it
# through either of those scripts silently turns that into effective batch
# 32 against only 40 samples/epoch (8 sims x overfit_repeat 5), collapsing
# 40 intended optimizer steps/epoch down to ~1. This is what actually
# invalidated the known-bad fullenc/head*/tail2/tail3 sng_pvc diagnostic
# runs (see EXPERIMENTS.md "Capacity diagnostic re-run under whole-structure
# unfreeze", Open Question 12) -- their `overfit_repeat` was always 5, not
# the 1 that archive/known-bad/README.md blames; the resolved
# training_config.yaml for those runs shows gradient_accumulation_steps
# patched 1 -> 4 against 8 processes instead. This script uses 1 tile and
# passes effective_batch=1 explicitly so gradient_accumulation_steps is left
# at whatever the source config already sets (1), exactly mirroring what
# jobs/lundquist/finetune_vae_debug.sbatch does for lundquist.
#
# ONE TILE IS DELIBERATE, not a smaller version of the 8-tile scripts:
# memorization needs optimizer-step COUNT, not batch size (same rationale as
# finetune_vae_overfit.yaml's header). A multi-tile variant would need MORE
# steps for the same wall-clock, not fewer.
#
# TIME BUDGET IS AN ESTIMATE, not measured on this hardware: lundquist's
# equivalent (1x A6000, same 50 epochs / 2000 steps at batch 1 + grad
# checkpointing) took ~64 min. sng_pvc's single-XPU-tile per-step cost for
# this exact workload hasn't been timed -- 03:00:00 leaves generous headroom
# (finetune_vae.sbatch's 8-tile full-data runs are asked for 12:00:00 for a
# much larger workload); watch the first job's epoch timing lines and shrink
# this if resubmitting.
#
# Defaults to the overfit-8 diagnostic; pass any other config as $1, e.g.
#   sbatch jobs/sng_pvc/finetune_vae_debug.sbatch configs/finetune_vae/finetune_vae_overfit_lr5e5_wsd_wholestruct.yaml
#
# Submit with:   sbatch jobs/sng_pvc/finetune_vae_debug.sbatch <config>
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_debug_vae.out

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "MASTER_ADDR: $MASTER_ADDR  MASTER_PORT: $MASTER_PORT"

# Single tile -- these collective-comms settings are inert no-ops at
# num_processes=1 but kept for parity with the 8-tile scripts.
export NUM_XPUS=1
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HDF5_USE_FILE_LOCKING=FALSE

BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_overfit.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_debug_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
# effective_batch=1, NOT the cluster default of 32 -- see header. Leaves
# gradient_accumulation_steps at the source config's own value (1) since
# target_batch=1, num_processes=1, batch_size=1.
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=1, output_suffix="", effective_batch=1)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"1 XPU tile x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 1; output_dir {cfg['output_dir']}"
)
PY

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/sng_pvc/INDEX.tsv

srun --mpi=none accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"

# Metrics/panels are the actual payload for a capacity diagnostic (the
# weights are worthless, see the config's own checkpoints.interval: null) --
# mirror them back onto the DSS-backed repo checkout so they're browsable
# and can be committed, same as finetune_vae.sbatch does for real runs.
OUTPUT_ROOT=$(python -c "
from windinet.cluster_config import CLUSTER_DEFAULTS
import os
print(CLUSTER_DEFAULTS['sng_pvc']['output_root'].format(scratch=os.environ.get('SCRATCH', '')))
")
RUN_NAME=${OUT_DIR#"${OUTPUT_ROOT}/"}
LOCAL_DIR="finetune_vae_outputs_sng_pvc/${RUN_NAME}"
mkdir -p "${LOCAL_DIR}"
rsync -a --exclude='checkpoints/' "${OUT_DIR}/" "${LOCAL_DIR}/"
echo "Mirrored (excl. checkpoints/) to ${LOCAL_DIR}"

echo "END TIME: $(date)"
