#!/bin/bash
#SBATCH --job-name=vae_finetune
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o sng_pvc_log/%j-%x.out
#SBATCH -e sng_pvc_log/%j-%x.err

# Batch submission for VAE finetuning on the sng_pvc cluster (Intel Data
# Center GPU Max 1550 / XPU, 8 tiles per node). Takes the training config as
# an argument, so any configs/finetune_vae_*.yaml can be run without editing
# this script.
#
# Submit with:   sbatch scripts/sng_pvc/finetune_vae.sbatch configs/finetune_vae_meaninit.yaml
# Watch with:    tail -f sng_pvc_log/<jobid>-vae_finetune.out
# Cancel with:   scancel <jobid>
#
# Verified working interactively on a single XPU tile first (srun + plain
# `python scripts/finetune_vae.py ...`) before moving to this 8-tile sbatch run.

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/sng_pvc/finetune_vae.sbatch <config.yaml>" >&2
    exit 1
fi
CONFIG="$1"

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

# Effective batch = batch_size * gradient_accumulation_steps * num_processes.
# Configs written for lundquist (4 GPUs) assume num_processes=4; check
# gradient_accumulation_steps in $CONFIG still gives the intended effective
# batch at num_processes=8 (NUM_XPUS below), and adjust if not.
srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "$CONFIG"

echo "END TIME: $(date)"
