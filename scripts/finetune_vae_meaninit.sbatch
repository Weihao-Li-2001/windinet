#!/bin/bash
#SBATCH --job-name=vae_meaninit
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o %j-%x.out
#SBATCH -e %j-%x.err

# Batch submission for the VAE mean-init finetune (configs/finetune_vae_meaninit.yaml)
# on the sng_pvc cluster (Intel Data Center GPU Max 1550 / XPU, 8 tiles per node).
#
# Submit with:   sbatch scripts/finetune_vae_meaninit.sbatch
# Watch with:    tail -f <jobid>-vae_meaninit.out
# Cancel with:   scancel <jobid>
#
# Verified working interactively on a single XPU tile first (srun + plain
# `python scripts/finetune_vae.py ...`) before moving to this 8-tile sbatch run.

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

# gradient_accumulation_steps=4 in the config is already tuned for
# num_processes=8 (effective batch = batch_size 1 * grad_acc 4 * 8 = 32,
# matching the 4-GPU lundquist baseline). RE-DERIVE IT IF YOU CHANGE NUM_XPUS.

# Stage the HDF5 dataset on node-local scratch instead of reading it from DSS
# for every sample. num_dataloader_workers is already 0 in this config and
# shockwave_data.py already retries opens with backoff -- both added because
# 8 ranks opening the same file on DSS concurrently causes transient "Unable
# to synchronously open object" failures. Those mitigations only shrink the
# collision window; a DSS hiccup that outlasts the retry budget still kills
# the whole job (all 8 ranks failing on the first batch has been observed).
# Copying the file once removes the network filesystem from the hot loop
# entirely. NOTE: this only stages onto the first allocated node -- if
# --nodes is ever raised above 1, this staging step must move inside the
# srun below so it runs on every node.
CONFIG_SRC="configs/finetune_vae_meaninit.yaml"
LOCAL_SCRATCH="${TMPDIR:-/tmp/$SLURM_JOB_ID}"
mkdir -p "$LOCAL_SCRATCH"

REMOTE_H5=$(python - "$CONFIG_SRC" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["data"]["data_root"])
PY
)
LOCAL_H5="$LOCAL_SCRATCH/$(basename "$REMOTE_H5")"
echo "Staging dataset on node-local scratch: $REMOTE_H5 -> $LOCAL_H5"
cp "$REMOTE_H5" "$LOCAL_H5"

LOCAL_CONFIG="$LOCAL_SCRATCH/finetune_vae_meaninit.local.yaml"
python - "$CONFIG_SRC" "$LOCAL_H5" "$LOCAL_CONFIG" <<'PY'
import sys, yaml
src, local_h5, dst = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
cfg["data"]["data_root"] = local_h5
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "$LOCAL_CONFIG"

echo "END TIME: $(date)"
