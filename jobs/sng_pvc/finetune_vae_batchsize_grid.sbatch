#!/bin/bash
#SBATCH --job-name=wh_bs_grid
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# batch_size x accum grid at effective_batch > 32, extending Open Question 23
# (jobs/sng_pvc/finetune_vae_batchsize.sbatch) past its deliberate ceiling.
# That script only accepts BATCH_SIZE in {1,2,4} because it holds
# effective_batch fixed at 32 (8 tiles x batch_size x accum = 32) to keep
# every arm's optimizer-step count and LR schedule identical to the rest of
# the ledger -- a clean single-variable test of batch_size alone.
#
# THIS SCRIPT DELIBERATELY BREAKS THAT: BATCH_SIZE and ACCUM are both set
# directly (effective_batch = 8 x BATCH_SIZE x ACCUM, no longer 32), to see
# whether larger micro-batches + larger effective_batch buy more throughput
# (or hit OOM/scaling limits) as batch_size grows past what the 8-tile,
# effective_batch=32 script allows. The seven arms this was designed for
# (see EXPERIMENTS.md's "batch_size x accum grid" entry near Open Question
# 23) share effective_batch across combos on purpose, so speed AND quality
# can be compared two ways: same batch_size at different effective_batch,
# and same effective_batch reached via a different (batch_size, accum) split:
#   BATCH_SIZE=4  ACCUM=2  -> effective_batch 64
#   BATCH_SIZE=4  ACCUM=4  -> effective_batch 128
#   BATCH_SIZE=8  ACCUM=1  -> effective_batch 64
#   BATCH_SIZE=8  ACCUM=2  -> effective_batch 128
#   BATCH_SIZE=8  ACCUM=4  -> effective_batch 256
#   BATCH_SIZE=16 ACCUM=1  -> effective_batch 128
#   BATCH_SIZE=16 ACCUM=2  -> effective_batch 256
#
# WARMUP_STEPS IS RESCALED, NOT LEFT AT THE BASE CONFIG'S VALUE: fewer
# steps/epoch at a bigger effective_batch would otherwise eat an outsized
# share of training in warmup (e.g. unscaled 200-step warmup at
# effective_batch=256 is ~44% of a 30-epoch run's ~450 total steps, vs ~5.6%
# at the effective_batch=32 baseline) -- this would confound "does batch
# size/effective_batch change quality" with "the wsd schedule's shape got
# wrecked". The Python patch step below reads the base config's own
# warmup_steps/effective_batch and rescales
# `warmup_steps' = round(warmup_steps * base_effective_batch / target_effective_batch)`
# so every arm keeps the same warmup *fraction* of its own (shorter) total
# step count, keeping the wsd shape comparable across arms. If this ever
# needs re-deriving by hand: base is 200 warmup @ effective_batch 32.
#
# THIS IS A REAL RUN, NOT A THROWAWAY DIAGNOSTIC: full epoch budget (whatever
# BASE_CONFIG specifies -- 30 for the current baseline), normal
# checkpointing/visualization, output committed to
# finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline_ep30_bs<N>_accum<M>/
# per protocol.
#
# IF A GIVEN (BATCH_SIZE, ACCUM) OOMs, THAT IS THE ANSWER FOR THAT ARM, NOT A
# BUG -- check the .err log for "Unable to allocate memory" / oneAPI OOM
# output and record which combos fail before assuming the same sizes are
# safe on lundquist/lrz_ai (different GPU memory budgets, and lrz_ai's 1-GPU
# launcher hits the same batch_size ceiling from a different direction --
# see jobs/lrz_ai/finetune_vae_batchsize.job).
#
# Submit with:
#   BATCH_SIZE=<4|8|16> ACCUM=<N> sbatch jobs/sng_pvc/finetune_vae_batchsize_grid.sbatch [BASE_CONFIG]
# Watch with:
#   tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_bs_grid.out

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

export USE_LIBUV=0   # see finetune_vae.sbatch's header for why this is here

export NUM_XPUS=8
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline   # or WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# 8 ranks all open train.h5 concurrently, same as finetune_vae.sbatch --
# DSS/Lustre file locking is unreliable under this much concurrent access.
export HDF5_USE_FILE_LOCKING=FALSE

BATCH_SIZE=${BATCH_SIZE:?"set BATCH_SIZE (4, 8, or 16), e.g. BATCH_SIZE=8 ACCUM=2 sbatch jobs/sng_pvc/finetune_vae_batchsize_grid.sbatch"}
ACCUM=${ACCUM:?"set ACCUM (the exact gradient_accumulation_steps for this arm), e.g. BATCH_SIZE=8 ACCUM=2 sbatch jobs/sng_pvc/finetune_vae_batchsize_grid.sbatch"}

# Defaults to the current baseline (30-epoch, whole-structure unfreeze,
# encoder_tail_lr_multiplier retired to hardcoded 1.0x); pass any other
# config as $1.
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_bs_grid_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "${BATCH_SIZE}" "${ACCUM}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" <<'PY'
import sys
import yaml
from windinet.cluster_config import patch_config_for_cluster, CLUSTER_DEFAULTS

src, dst, batch_size, accum, num_processes = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
)
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = batch_size

base_effective_batch = CLUSTER_DEFAULTS["sng_pvc"]["effective_batch"]
base_warmup_steps = cfg["optimization"]["warmup_steps"]
target_effective_batch = num_processes * batch_size * accum
new_warmup_steps = round(base_warmup_steps * base_effective_batch / target_effective_batch)
cfg["optimization"]["warmup_steps"] = new_warmup_steps

patch_config_for_cluster(
    cfg,
    cluster="sng_pvc",
    num_processes=num_processes,
    output_suffix=f"_bs{batch_size}_accum{accum}",
    effective_batch=target_effective_batch,
)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU tiles x batch_size {batch_size} x accum {accum} -> "
    f"effective_batch {target_effective_batch}; warmup_steps {base_warmup_steps} -> "
    f"{new_warmup_steps}; output_dir={cfg['output_dir']}"
)
PY

# Make log_finetuning_vae/sng_pvc/ self-describing: one row per job mapping
# the log to the config it ran and the output_dir it wrote, tagged with
# BATCH_SIZE/ACCUM so grepping the index alone tells arms apart -- same
# convention as finetune_vae_batchsize.sbatch's INDEX.tsv rows.
OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}_bs${BATCH_SIZE}_accum${ACCUM}" \
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
