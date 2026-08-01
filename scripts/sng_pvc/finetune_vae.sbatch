#!/bin/bash
#SBATCH --job-name=wh_finetune_vae
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o log_sng_pvc/%j-%x.out
#SBATCH -e log_sng_pvc/%j-%x.err

# Batch submission for VAE finetuning on the sng_pvc cluster (Intel Data
# Center GPU Max 1550 / XPU, 8 tiles per node).
#
# Defaults to the shared baseline config (see EXPERIMENTS.md) and patches in
# the sng-specific bits (storage under $SCRATCH, effective-batch-preserving
# gradient_accumulation_steps for 8 tiles, num_dataloader_workers=0) onto a
# temp copy rather than forking another checked-in YAML -- same pattern as
# scripts/lundquist/finetune_vae_4gpu.sbatch. Pass any other config as $1:
#   sbatch scripts/sng_pvc/finetune_vae.sbatch configs/<your-config>.yaml
#
# Submit with:   sbatch scripts/sng_pvc/finetune_vae.sbatch
# Watch with:    tail -f log_sng_pvc/<jobid>-wh_finetune_vae.out
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

BASE_CONFIG=${1:-configs/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst, num_processes = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=num_processes, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU tiles x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 32; output_dir {cfg['output_dir']}"
)
PY

# Make log_sng_pvc/ self-describing: one row per job mapping the log to the
# config it ran and the output_dir it wrote, same convention as log_lundquist/.
OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_sng_pvc/INDEX.tsv

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"

echo "END TIME: $(date)"
