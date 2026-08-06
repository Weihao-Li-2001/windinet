#!/bin/bash
#SBATCH --job-name=wh_diag_vae
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # NOT "test" -- see caveat below, one unmodified epoch alone exceeds test's 30 min cap
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Throughput diagnostic for the sng_pvc slowdown (see EXPERIMENTS.md /
# 2026-08-02 discussion): baseline job 520211 averaged 38.7 min/epoch on
# 8 XPU tiles, roughly the SAME as lundquist's 2-GPU (A6000) estimate of
# 32-36 min/epoch in jobs/lundquist/finetune_vae_2gpu.sbatch. 4x the compute
# should not land at parity -- this script isolates why.
#
# vae_trainer.py now logs a per-epoch phase breakdown
# ("Epoch N timing: train=...s eval=...s viz=...s ckpt=...s total=...s"),
# so a short run already tells you where the time goes -- no need to guess
# from whole-job START/END like job 520211 had to be read. Only 2 epochs are
# run: epoch 1 eats one-time model-load/compile cost, epoch 2 is the clean
# reading. A 3rd epoch would not add information -- every epoch does
# identical work -- so it's not worth the extra ~40 min.
#
# WHY partition=general, NOT "test" (30 min cap): Exp 0 below changes
# nothing, so its epoch takes the same ~38.7 min as job 520211's average.
# 2 epochs of that is ~77 min, already over test's cap before a single
# "Epoch 2 timing" line would print. 02:00:00 leaves headroom for queueing
# variance; if Exp 1/2's fix works, the job will simply finish early and
# free the allocation, no harm in the larger budget.
#
# Three things to try, one variable at a time (see EXPERIMENTS.md protocol
# "one variable per run"), each its own submission:
#
#   sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
#       Exp 0 -- baseline as-is (8 tiles, workers=0, full checkpoint+viz
#       every epoch). Read epoch 2's timing line (skip epoch 1, it eats
#       one-time model-load/compile cost). This alone tells you the
#       train/eval/viz/ckpt split -- run this FIRST, its output decides
#       whether Exp 1 or Exp 2 below is worth running at all.
#
#   DIAG_TAG=workers2 DIAG_WORKERS=2 sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
#       Exp 1 -- H1 (I/O-bound training step): num_dataloader_workers 0->2.
#       cluster_config.py dropped this to 0 because 4 workers x 8 ranks
#       crashed opening train.h5 concurrently on DSS; 0 was never shown to
#       be the fastest safe value, just the one that didn't crash. Compare
#       this run's "train=" time to Exp 0's.
#
#   DIAG_TAG=noio DIAG_SKIP_HEAVY_IO=1 sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
#       Exp 2 -- H3 (checkpoint/viz I/O to the DSS network filesystem):
#       baseline config has checkpoints.interval=1 + save_last_state=true,
#       so EVERY epoch writes a ~1.1GB safetensors decoder checkpoint plus a
#       ~4.4GB optimizer-state torch.save (AdamW m/v in fp32 over 552.9M
#       trainable params) to $SCRATCH (network filesystem), on top of 8
#       visualization PNGs. lundquist writes the equivalent to LOCAL disk.
#       This run disables save_last_state and visualization so "ckpt="/"viz="
#       collapse to ~0 -- compare this run's "total=" to Exp 0's "total=" to
#       see how much of the 38.7 min/epoch that I/O actually was.
#
# Exp 0's breakdown should make it obvious which of Exp 1 / Exp 2 (or both)
# is worth running. Communication-topology (H2: 8 ranks x accum4 vs
# lundquist's 2 ranks x accum16) is NOT tested here -- see
# jobs/sng_pvc/finetune_vae_diag_4rank.sbatch, it needs Exp 0/2 ruled out
# first since heavy per-epoch I/O would confound a communication-only test.
#
# --- Added 2026-08-06, after the eval parallelization fix (EXPERIMENTS.md
# "Eval parallelization fix, NOT YET TESTED") -- two more single-variable
# diagnostics, same "run Exp 0's baseline command with one knob changed"
# pattern as above. The cluster default is now workers=4 (not 0, note Exp
# 0/1's text above predates that), so "Exp 0 as-is" below already means
# workers=4, HDF5_USE_FILE_LOCKING=FALSE, and the new eval code path (it's
# unconditional now, not a flag) -- there is no config knob to run the OLD
# unparallelized eval for comparison; the "before" numbers are the ones
# already in EXPERIMENTS.md's head-vs-tail sweep table and
# finetune_vae_baseline_tail3's per-epoch log (eval ~853s/epoch avg).
#
#   DIAG_TAG=evalfix_confirm sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
#       Exp 3 -- confirm the eval parallelization fix. Nothing to override;
#       this is "Exp 0 as-is" under a distinct tag so its output_dir/log
#       don't collide with the historical Exp 0 (job 520300). Read epoch 2's
#       "eval=" and compare to the ~800-1050s/epoch this repo's committed
#       full-data runs show pre-fix. Also sanity-check val_vrmse against a
#       comparable pre-fix number if this ever runs past 2 epochs (it won't
#       by default -- DIAG_EPOCHS=2) -- the all-reduce is exact, so a
#       multi-epoch value should track pre-fix numbers, not deviate.
#
#   DIAG_TAG=filelock_true DIAG_FILE_LOCKING=TRUE sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
#       Exp 4 -- does concurrent 8-rank *eval* (new, from the fix above) trip
#       the "Unable to synchronously open object (can't open file)" failure
#       that HDF5_USE_FILE_LOCKING=FALSE was set to avoid on the *train*
#       side (see the flag's own comment further down in this script)? Only
#       eval reads concurrently across all 8 ranks now that it's
#       parallelized -- train already did before this fix, so if TRUE
#       crashes here, the crash is coming from eval's newly-concurrent
#       reads, not a train regression. Read: does the job crash or finish
#       clean? If it finishes clean, also compare "eval=" to Exp 3's -- file
#       locking could plausibly add overhead even without crashing.
#
# Weights from these runs are throwaway -- output_dir is suffixed
# "_diag_<tag>" so nothing collides with or wipes the real baseline in
# finetune_vae_outputs_sng_pvc/finetune_vae_baseline/.

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

export NUM_XPUS=8
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# DIAG_FILE_LOCKING=TRUE overrides the FALSE default -- see Exp 3 in the
# header (does concurrent 8-rank *eval* now trip the same DSS locking issue
# train was already exempted from?).
export HDF5_USE_FILE_LOCKING=${DIAG_FILE_LOCKING:-FALSE}

# Diagnostic knobs -- see header for what each combination tests.
DIAG_TAG=${DIAG_TAG:-base}
DIAG_WORKERS=${DIAG_WORKERS:-}          # empty = leave cluster default (4)
DIAG_SKIP_HEAVY_IO=${DIAG_SKIP_HEAVY_IO:-0}
DIAG_EPOCHS=${DIAG_EPOCHS:-2}            # epoch1=warmup, epoch2=clean reading
echo "DIAG_TAG=${DIAG_TAG} DIAG_WORKERS=${DIAG_WORKERS:-<default>} DIAG_SKIP_HEAVY_IO=${DIAG_SKIP_HEAVY_IO} DIAG_EPOCHS=${DIAG_EPOCHS} HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING}"

BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_diag_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" \
    "${DIAG_TAG}" "${DIAG_WORKERS}" "${DIAG_SKIP_HEAVY_IO}" "${DIAG_EPOCHS}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst, num_processes, tag, workers, skip_heavy_io, epochs = sys.argv[1:8]
num_processes = int(num_processes)
skip_heavy_io = skip_heavy_io == "1"
epochs = int(epochs)

cfg = yaml.safe_load(open(src))
patch_config_for_cluster(
    cfg, cluster="sng_pvc", num_processes=num_processes, output_suffix=f"_diag_{tag}"
)

if workers != "":
    cfg["data"]["num_dataloader_workers"] = int(workers)

# Diagnostic only: don't care about convergence, only wall clock.
cfg["optimization"]["epochs"] = epochs
cfg["checkpoints"]["interval"] = 1  # keep triggering _save_checkpoint every epoch...
if skip_heavy_io:
    cfg["checkpoints"]["save_last_state"] = False  # ...but skip the ~4.4GB optimizer-state write
    cfg["visualization"]["enabled"] = False          # and the 8 PNGs/epoch

yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU tiles x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 32; workers={cfg['data']['num_dataloader_workers']}; "
    f"skip_heavy_io={skip_heavy_io}; epochs={epochs}; output_dir {cfg['output_dir']}"
)
PY

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}_${DIAG_TAG}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/sng_pvc/INDEX.tsv

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
