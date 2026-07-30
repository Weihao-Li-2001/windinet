#!/bin/bash
#SBATCH --job-name=vae_meaninit_test1gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=test              # max 30 min -- quick sanity check only
#SBATCH --account=pn82ku
#SBATCH -o %j-%x.out
#SBATCH -e %j-%x.err

# Single-tile smoke test for the VAE mean-init finetune
# (configs/finetune_vae_meaninit.yaml) on sng_pvc, before burning a full
# 8-tile allocation on scripts/finetune_vae_meaninit.sbatch.
#
# Submit with:   sbatch scripts/finetune_vae_meaninit_test1gpu.sbatch
# Watch with:    tail -f <jobid>-vae_meaninit_test1gpu.out
# Cancel with:   scancel <jobid>
#
# What to look for in the log:
#   - the [ShockWaveDataset DEBUG] lines (h5_path, external targets to
#     prewarm = 10)
#   - a pause right after those lines -- that's shockwave_data.py's startup
#     prewarm resolving each of the 10 gamma shard external links once
#   - training reaching the first logged step with no
#     KeyError: "Unable to synchronously open object" traceback
#
# NUM_XPUS defaults to 1 (a true single-tile run). Bump it to e.g. 2 to also
# exercise multi-rank concurrent access to train.h5 without paying for a full
# 8-tile allocation.
export NUM_XPUS=1

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

export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline   # or WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# Same reasoning as the full 8-tile sbatch: HDF5 file locking is unreliable
# on network filesystems like DSS/Lustre and throws "Unable to synchronously
# open object (can't open file)" under concurrent access -- disable it.
export HDF5_USE_FILE_LOCKING=FALSE

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py configs/finetune_vae_meaninit.yaml

echo "END TIME: $(date)"
