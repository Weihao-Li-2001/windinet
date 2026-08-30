#!/bin/bash
#SBATCH --job-name=wh_diag_vae_2node
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # NOT "test" (30 min cap) -- see finetune_vae_diag.sbatch's header for why
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err

# Node-count scaling probe: does requesting 2 nodes (16 XPU tiles) actually
# train faster than 1 node (8 tiles), or does the inter-node network
# interconnect eat the extra compute the same way intra-node communication
# already dominates elsewhere in this ledger (see "sng_pvc throughput
# diagnostic" in EXPERIMENTS.md -- H1's I/O fix and H3's I/O-ruled-out both
# left communication/compute, not I/O, as the remaining unknown)?
#
# THE QUESTION: 1-node/8-tile reference (job 520456, workers=4, ZE_FLAT,
# accum 4) trains at ~869s/epoch (epoch-2 train= time). This run doubles to
# 2 nodes x 8 tiles = 16 ranks; patch_config_for_cluster keeps effective_batch
# at 32, so gradient_accumulation_steps halves automatically (4 -> 2). If
# communication (now crossing the node-to-node fabric, not just intra-node
# XeLink/PCIe) is the bottleneck, epoch-2 "train=" will land near 869s again
# (no speedup) or worse (network slower than intra-node links). If compute
# was still the limiter, it should roughly halve.
#
# READ-OUT: compare this run's "Epoch 2 timing: train=...s" to job 520456's
# 869.3s. KILL CRITERION: if train= is >= 90% of 869.3s, 2-node scaling isn't
# worth the extra allocation for this workload -- don't chase 4-node next.
#
# CAVEAT: this is the first multi-node launch attempted on sng_pvc for this
# project. The rendezvous (MASTER_ADDR = first node in $SLURM_NODELIST,
# --num_machines/--machine_rank via srun's per-node SLURM_PROCID) mirrors the
# scaffolding jobs/sng_pvc/finetune_vae.sbatch already carries for this
# exact purpose ("# increase this for multi-node" in its --nodes comment),
# but it has never actually been exercised at nodes=2 -- if the two nodes
# can't reach each other on MASTER_PORT, this will hang at NCCL/CCL init
# rather than fail fast; give it a few minutes past job start before
# assuming it's stuck and checking `scontrol show job $SLURM_JOB_ID`.
#
# UPDATE 2026-08-06, job 521396 FAILED (see EXPERIMENTS.md H4 for the full
# diagnosis) -- root cause NOT confirmed, but the evidence points at a
# specific known class of bug, and the fix below is applied but UNTESTED:
#
#   torch.distributed.DistStoreError: wait timeout after 900000ms, keys:
#   /none/torchelastic/role_info/0, /none/torchelastic/role_info/1
#
# The .err log's c10d socket trace is the key clue: BOTH the cross-node
# connection (i20r02c04s06 -> i20r02c04s04:45361) AND node 0's own LOOPBACK
# connection to its own rendezvous store (i20r02c04s04 -> i20r02c04s04:45361)
# completed their TCP handshake fine, then both timed out identically
# waiting for a response. Loopback hanging the same way as the cross-node
# connection rules out an inter-node firewall/routing problem (loopback
# never leaves the host) and points at the TCPStore SERVER on the master
# never actually responding to ANY client, local or remote -- a known
# failure signature of PyTorch's libuv-backed TCPStore (the default since
# `USE_LIBUV` defaults to "1" in this torch version), which has had
# reported hangs on some HPC/interconnect setups where the older
# non-libuv TCPStore backend works fine. Fix: force it off.
#
# Also present but judged NOT the cause: `slurmstepd: error: Unable to
# create TMPDIR [/hppfs/scratch/0D/go76fuz2/tmp]` (falls back to per-node
# local /tmp). This happens before this script's own commands run (SLURM
# itself emits it), so it can't be fixed from in here -- a one-time
# `mkdir -p /hppfs/scratch/0D/go76fuz2/tmp` on the login node would clear
# it, but it's unrelated to the TCPStore hang above (that's a pure-socket
# protocol issue; local-vs-shared TMPDIR wouldn't explain a loopback
# connection also hanging) and hasn't stopped any other job from running,
# so it's noted rather than chased here.
#
# Weights are throwaway (output_suffix "_diag_2node"), not mirrored to git.

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

# Primary fix for job 521396's rendezvous hang -- see the UPDATE note above.
# Forces PyTorch's TCPStore onto its older, pre-libuv implementation.
# Cheap/safe either way: does nothing if libuv wasn't actually the problem,
# and every earlier 1-node run's rendezvous was a no-op regardless of this
# setting (num_machines=1 never exercises the real multi-node TCPStore
# path), so this can't regress anything already working.
export USE_LIBUV=0

# If USE_LIBUV=0 alone doesn't fix it, these make the NEXT failure far more
# diagnosable than 521396's was: DETAIL logs every collective/store op
# instead of just the final timeout, and unbuffered output means those logs
# actually show up in the .out/.err before a hang, not just after a crash.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

export NUM_XPUS=8   # per node, same as the 1-node baseline
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HDF5_USE_FILE_LOCKING=FALSE

# Same knobs as finetune_vae_diag.sbatch, so this can be re-run with e.g.
# DIAG_WORKERS=4 later without editing the script.
DIAG_TAG=${DIAG_TAG:-2node}
DIAG_WORKERS=${DIAG_WORKERS:-}          # empty = leave cluster default (2)
DIAG_EPOCHS=${DIAG_EPOCHS:-2}            # epoch1=warmup, epoch2=clean reading
echo "DIAG_TAG=${DIAG_TAG} DIAG_WORKERS=${DIAG_WORKERS:-<default>} DIAG_EPOCHS=${DIAG_EPOCHS} nodes=${SLURM_JOB_NUM_NODES}"

BASE_CONFIG=${1:-configs/finetune_vae/archive/done/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_diag2node_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" \
    "${DIAG_TAG}" "${DIAG_WORKERS}" "${DIAG_EPOCHS}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst, num_processes, tag, workers, epochs = sys.argv[1:7]
num_processes = int(num_processes)
epochs = int(epochs)

cfg = yaml.safe_load(open(src))
patch_config_for_cluster(
    cfg, cluster="sng_pvc", num_processes=num_processes, output_suffix=f"_diag_{tag}"
)

if workers != "":
    cfg["data"]["num_dataloader_workers"] = int(workers)

# Diagnostic only: don't care about convergence, only wall clock.
cfg["optimization"]["epochs"] = epochs
cfg["checkpoints"]["interval"] = 1
cfg["checkpoints"]["save_last_state"] = False
cfg["visualization"]["enabled"] = False

yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU ranks across {num_processes // 8} node(s) x accum "
    f"{cfg['optimization']['gradient_accumulation_steps']} -> effective batch 32; "
    f"workers={cfg['data']['num_dataloader_workers']}; epochs={epochs}; "
    f"output_dir {cfg['output_dir']}"
)
PY

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> logs/sng_pvc/INDEX.tsv

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"

echo "END TIME: $(date)"
echo "Diagnostic run -- weights/state left on \$SCRATCH under ${OUT_DIR}, NOT mirrored to git (throwaway)."
