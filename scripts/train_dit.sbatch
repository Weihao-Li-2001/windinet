#!/bin/bash
#SBATCH --job-name=shockwave_dit
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o %j-%x.out
#SBATCH -e %j-%x.err

# Batch submission for the ShockWaveNet DiT run on the sng_pvc cluster
# (Intel Data Center GPU Max 1550 / XPU, 8 tiles per node). For the
# NVIDIA/lundquist variant of this script, see the main branch.
#
# Submit with:   sbatch scripts/train_dit.sbatch
# Watch with:    tail -f <jobid>-shockwave_dit.out
# Cancel with:   scancel <jobid>
#
# Adapted from a working sng_pvc sbatch template (micromamba env activation,
# CCL/Level-Zero env vars, torchrun-style rendezvous).
#
# NOTE: the partition's time limit is a hard kill. Check the step rate in the
# first few minutes; if 10000 steps will not fit, lower optimization.steps
# rather than relying on a resume -- LtxvTrainer._load_checkpoint restores the
# transformer weights only, not the optimizer or LR scheduler state, so a
# resumed run restarts the cosine schedule from the top.

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

# --num_processes must match NUM_XPUS * num nodes, and the config's
# checkpoints.interval assumes it (steps_per_epoch = 3825 / (4 * 4 * N)).
srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/train.py configs/shockwavenet.yaml

echo "END TIME: $(date)"
