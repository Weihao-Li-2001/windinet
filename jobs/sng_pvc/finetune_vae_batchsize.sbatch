#!/bin/bash
#SBATCH --job-name=wh_bs_sweep
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Batch-size sweep, Open Question 23 (EXPERIMENTS.md). Tests whether
# optimization.batch_size (per-GPU/tile micro-batch) changes results or only
# wall-clock/OOM feasibility, at a FIXED effective_batch=32 -- same
# optimizer-step count and LR schedule as every other run in this ledger,
# see windinet/cluster_config.py's derivation. Same 8-tile setup as
# finetune_vae.sbatch (the config previously used for sng_pvc, job 523577),
# parameterized by BATCH_SIZE instead of relying on the baseline config's
# hard-coded batch_size=1 / accum=4 -- same relation
# jobs/lundquist/finetune_vae_batchsize.sbatch has to
# jobs/lundquist/finetune_vae_2gpu.sbatch.
#
# CEILING AT 8 TILES: effective_batch is fixed at 32, so
# batch_size x num_processes must divide 32 -- at num_processes=8 that caps
# batch_size at 4 (accum=1, the floor). batch_size 8 or 16 would need
# effective_batch >= 64/128, a different (and not directly comparable)
# experiment, so this script only accepts BATCH_SIZE in {1, 2, 4} --
# patch_config_for_cluster raises ValueError for anything larger, on
# purpose. The 8/16 arms of the "1 to 16" sweep only make sense at a lower
# tile count -- see jobs/lrz_ai/finetune_vae_batchsize.job (1 GPU, covers
# the full range) or jobs/sng_pvc/finetune_vae_debug.sbatch-style single-tile
# launch if a same-cluster full-range read-out is wanted later.
#
# Submit with:   BATCH_SIZE=<1|2|4> sbatch jobs/sng_pvc/finetune_vae_batchsize.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_bs_sweep.out
#
# gradient_accumulation_steps is re-derived by patch_config_for_cluster so
# 8 tiles x BATCH_SIZE x accum always lands on effective_batch 32 -- e.g.
# BATCH_SIZE=1 -> accum 4 (today's baseline, job 523577), BATCH_SIZE=2 ->
# accum 2, BATCH_SIZE=4 -> accum 1. Total optimizer steps and the LR
# schedule are unchanged across arms, so any val_vrmse difference is a real
# finding, not a step-count confound (see EXPERIMENTS.md Open Question 23's
# decision rule for how the read-out turns into a choice).
#
# THIS IS A REAL RUN, NOT A THROWAWAY DIAGNOSTIC: full 18 epochs, normal
# checkpointing/visualization, output committed to
# finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline_bs<N>/
# per protocol. BATCH_SIZE=1 reproduces job 523577's setup exactly (same
# 8 tiles, same accum=4) -- expect it to land at val_vrmse ~0.08360 again
# modulo seed noise, a sanity check on the harness before trusting bs2/bs4.
#
# IF A GIVEN BATCH_SIZE OOMs, THAT IS THE ANSWER FOR THAT ARM, NOT A BUG --
# see Open Question 23's kill criterion. Check the .err log for
# "Unable to allocate memory" / oneAPI OOM output and record which sizes
# fail here before assuming the same sizes are safe on lundquist/lrz_ai
# (different GPU memory budgets). Per-epoch timing is logged by
# windinet/training/vae_trainer.py as "Epoch N timing: ... total=Xs" in the
# .out log.

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

BATCH_SIZE=${BATCH_SIZE:?"set BATCH_SIZE (1, 2, or 4 -- 8 tiles x effective_batch=32 caps it here), e.g. BATCH_SIZE=4 sbatch jobs/sng_pvc/finetune_vae_batchsize.sbatch"}

# Defaults to the ep18 whole-structure baseline (Open Question 23's test
# subject); pass any other config as $1.
BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_bs_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "${BATCH_SIZE}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" <<'PY'
import sys
import yaml
from windinet.cluster_config import patch_config_for_cluster
src, dst, batch_size, num_processes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = yaml.safe_load(open(src))
cfg["optimization"]["batch_size"] = batch_size
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=num_processes, output_suffix=f"_bs{batch_size}")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU tiles x batch_size {batch_size} x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 32; output_dir {cfg['output_dir']}"
)
PY

# Make log_finetuning_vae/sng_pvc/ self-describing: one row per job mapping
# the log to the config it ran and the output_dir it wrote, tagged with
# BATCH_SIZE so grepping the index alone tells arms apart -- same convention
# as jobs/lundquist/finetune_vae_batchsize.sbatch's INDEX.tsv rows.
OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}_bs${BATCH_SIZE}" \
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
