#!/bin/bash
#SBATCH --job-name=wh_finetune_vae
#SBATCH --time=12:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Batch submission for VAE finetuning on the sng_pvc cluster (Intel Data
# Center GPU Max 1550 / XPU, 8 tiles per node).
#
# Defaults to the shared baseline config (see EXPERIMENTS.md) and patches in
# the sng-specific bits (storage under $SCRATCH, effective-batch-preserving
# gradient_accumulation_steps for 8 tiles, num_dataloader_workers=4) onto a
# temp copy rather than forking another checked-in YAML -- same pattern as
# jobs/lundquist/finetune_vae_4gpu.sbatch. Pass any other config as $1:
#   sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/<your-config>.yaml
#
# $2 (optional) overrides data_root, e.g. to train against a resolution
# other than CLUSTER_DEFAULTS['sng_pvc']'s default 128x128_ds:
#   sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/<your-config>.yaml \
#       "${SCRATCH}/windinet/euler_mq_dataset/256x256_ds/train.h5"
#
# Submit with:   sbatch jobs/sng_pvc/finetune_vae.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_finetune_vae.out
# Cancel with:   scancel <jobid>
#
# Verified working interactively on a single XPU tile first (srun + plain
# `python scripts/finetune_vae.py ...`) before moving to this 8-tile sbatch run.

# No `-u`: `module load micromamba` itself (not just the later shell hook/
# activate calls) references variables like MAMBA_ROOT_PREFIX that the
# module never exports and that aren't given internal defaults -- fatal
# under nounset before our own code even runs. `-e`/pipefail stay on so a
# real failure in our own commands (the yaml patch step, the launch) still
# stops the script.
set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet   # env with an XPU-enabled torch +
                                # intel-extension-for-pytorch installed

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

# Rendezvous settings (only exercised once --nodes/--ntasks-per-node above are
# increased for multi-node; harmless as a single-node no-op otherwise).
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

# Preventive fix carried over from jobs/sng_pvc/finetune_vae_diag_2node.sbatch
# (see its header and EXPERIMENTS.md's H4 section for the full diagnosis):
# job 521396, the first --nodes=2 attempt anywhere in this project, hung in
# rendezvous with a TCPStore that accepted connections but never responded
# -- including on its own node's loopback connection, which points at
# PyTorch's libuv TCPStore backend rather than an inter-node network/
# firewall problem. Harmless no-op at --nodes=1 (today's default above),
# same as the MASTER_ADDR/PORT block itself.
export USE_LIBUV=0

export NUM_XPUS=8
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline   # or WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# 8 ranks all open train.h5 concurrently. HDF5's file locking is unreliable
# on network filesystems like DSS/Lustre and throws "Unable to synchronously
# open object (can't open file)" under this much concurrent access -- disable it.
export HDF5_USE_FILE_LOCKING=FALSE

# Defaults to the current 128-res baseline (finetune_vae_whole_structure_
# baseline_ep30.yaml, 30ep/1.0x-encoder-LR, val_vrmse 0.078662, job 524322)
# -- was finetune_vae_baseline.yaml (frozen-trunk, superseded), now archived
# to configs/finetune_vae/archive/done/, see EXPERIMENTS.md.
#
# batch_size 4 (2026-08-16): the largest batch_size that still divides
# effective_batch=32 across 8 tiles (accum=1) -- adopted as sng_pvc's
# production default per Open Question 23's sng_pvc arms (job 524577,
# batch_size=4, val_vrmse 0.084099 vs batch_size=1's 0.083633, well inside
# the ~1.1% noise floor) and the batch_size x accum grid follow-up. See
# EXPERIMENTS.md's "Open Question 23" section for the full readout.
#
# EFFECTIVE_BATCH (2026-08-30, default 32 -- unchanged behavior): overrides
# the target effective batch patch_config_for_cluster() derives
# gradient_accumulation_steps from, e.g. `EFFECTIVE_BATCH=128 sbatch ...`
# with batch_size still hardcoded to 4 above and 8 XPU tiles gives accum=4
# instead of the default 1 -- written for the accum=1-vs-4 runtime/effect
# comparison in configs/finetune_vae/finetune_vae_whole_structure_baseline_
# ep10_256res_acc{1,4}.yaml. Leave unset for normal production runs.
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-32}
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml}
DATA_ROOT_OVERRIDE=${2:-}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" "${DATA_ROOT_OVERRIDE}" "${EFFECTIVE_BATCH}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst, num_processes, data_root_override, effective_batch = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]),
)
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = 4  # production default, see header comment above
patch_config_for_cluster(
    cfg,
    cluster="sng_pvc",
    num_processes=num_processes,
    output_suffix="",
    effective_batch=effective_batch,
    data_root=data_root_override or None,
)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU tiles x batch_size {cfg['optimization']['batch_size']} x "
    f"accum {cfg['optimization']['gradient_accumulation_steps']} -> effective batch "
    f"{effective_batch}; output_dir {cfg['output_dir']}"
)
PY

# Make log_finetuning_vae/sng_pvc/ self-describing: one row per job mapping the log to the
# config it ran and the output_dir it wrote, same convention as log_finetuning_vae/lundquist/.
OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/sng_pvc/INDEX.tsv

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"

# Mirror the run's visualizations/metrics/config back onto the DSS-backed repo
# checkout so they're browsable without touching $SCRATCH -- but not
# checkpoints/ (weights + optimizer state), which stay scratch-only since
# they're large and already addressed via OUT_DIR in log_finetuning_vae/sng_pvc/INDEX.tsv.
OUTPUT_ROOT=$(python -c "
from windinet.cluster_config import CLUSTER_DEFAULTS
import os
print(CLUSTER_DEFAULTS['sng_pvc']['output_root'].format(scratch=os.environ.get('SCRATCH', '')))
")
RUN_NAME=${OUT_DIR#"${OUTPUT_ROOT}/"}
LOCAL_DIR="finetune_vae_outputs/sng_pvc/${RUN_NAME}"
mkdir -p "${LOCAL_DIR}"
rsync -a --exclude='checkpoints/' "${OUT_DIR}/" "${LOCAL_DIR}/"
echo "Mirrored (excl. checkpoints/) to ${LOCAL_DIR}"

echo "END TIME: $(date)"
