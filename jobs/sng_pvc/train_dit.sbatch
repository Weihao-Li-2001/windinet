#!/bin/bash
#SBATCH --job-name=shockwave_dit
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    # increase this for multi-node
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o logs/sng_pvc/%j-%x.out
#SBATCH -e logs/sng_pvc/%j-%x.err

# Batch submission for the ShockWaveNet DiT run on the sng_pvc cluster
# (Intel Data Center GPU Max 1550 / XPU, 8 tiles per node). Companion to
# preprocess_dit_data.sbatch -- run that first, then this job trains on its
# output. For the NVIDIA/lundquist variant, see jobs/lundquist/train_dit_2gpu.sbatch;
# for lrz_ai (enroot container), see jobs/lrz_ai/train_dit.job.
#
# PARAMETERIZED (2026-08-27): previously hardcoded to configs/shockwavenet.yaml,
# which is lundquist's own config with paths hardcoded to lundquist's local
# disk -- this job had never actually been run against it (no entry in
# logs/sng_pvc/INDEX.tsv), so nothing depended on that default.
# Now takes PREPROCESSED_NAME + BASE_CONFIG like the other clusters' DiT
# launchers, and substitutes a __PREPROCESSED_ROOT__ placeholder the same way
# jobs/lrz_ai/train_dit.job and jobs/lundquist/train_dit_2gpu.sbatch do.
#
# PREPROCESSED_NAME is REQUIRED, no default: must match the <name>
# preprocess_dit_data.sbatch wrote under ${SCRATCH}/windinet/dit_preprocessed/.
# $2 (optional) overrides BASE_CONFIG, default
# configs/shockwavenet_sng_pvc_baseline.yaml -- pass one of the sibling
# kl1e6/kl1e5/anchor_kl1e7 configs for the other VAE arms.
#
# STEPS (optional env var, 2026-08-30): overrides optimization.steps at
# launch instead of editing the config file -- e.g. for a shorter trial run
# without forking a new BASE_CONFIG just to change one number. Leave unset
# to keep BASE_CONFIG's own value. Does NOT touch checkpoints.interval or
# validation.interval, which stay in absolute-step units from BASE_CONFIG --
# for a much shorter STEPS, consider whether those checkpoint/val cadences
# still make sense (e.g. BASE_CONFIG's checkpoints.interval=120 assumes a
# 10000-step horizon; at STEPS=5000 that's still ~41 checkpoints, unchanged
# cadence, just fewer of them).
#
# Submit with:
#   STEPS=<n> sbatch jobs/sng_pvc/train_dit.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]
#
# NOTE: the partition's 24h limit is a hard kill. Check the step rate in the
# first few minutes; if optimization.steps will not fit, be ready to resume
# rather than assume 24h is enough -- LtxvTrainer._load_checkpoint restores
# the transformer weights only, not the optimizer or LR scheduler state, so
# a resumed run restarts the cosine schedule from the top (set
# model.load_checkpoint to the output_dir and resubmit).

# No `-u`, same reasoning as jobs/sng_pvc/finetune_vae.sbatch: `module load
# micromamba` itself references unset-by-default variables.
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

if [[ -z "${1:-}" ]]; then
    echo "error: PREPROCESSED_NAME is required -- the <name> preprocess_dit_data.sbatch wrote to" >&2
    echo "  usage: sbatch jobs/sng_pvc/train_dit.sbatch <PREPROCESSED_NAME> [BASE_CONFIG]" >&2
    exit 1
fi
PREPROCESSED_NAME=$1
BASE_CONFIG=${2:-configs/shockwavenet_sng_pvc_baseline.yaml}
STEPS=${STEPS:-}
RUN_CONFIG=$(mktemp /tmp/dit_train_sng_pvc_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

echo "PREPROCESSED_NAME: ${PREPROCESSED_NAME}"
echo "BASE_CONFIG: ${BASE_CONFIG}"
echo "STEPS override: ${STEPS:-<none, using BASE_CONFIG default>}"

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

# Substitute the __PREPROCESSED_ROOT__ placeholder for the real path, same
# pattern as jobs/lrz_ai/train_dit.job and jobs/lundquist/train_dit_2gpu.sbatch
# (but a plain $SCRATCH dir here, not a /mnt/result container mount).
python - "${BASE_CONFIG}" "${RUN_CONFIG}" "${PREPROCESSED_NAME}" "${STEPS}" <<'PY'
import os, sys, yaml
src, dst, name, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = yaml.safe_load(open(src))
root = os.path.join(os.environ.get("SCRATCH", ""), "windinet", "dit_preprocessed", name)
cfg["data"]["preprocessed_data_root"] = cfg["data"]["preprocessed_data_root"].replace("__PREPROCESSED_ROOT__", root)
if cfg.get("validation", {}).get("data_root"):
    cfg["validation"]["data_root"] = cfg["validation"]["data_root"].replace("__PREPROCESSED_ROOT__", root)
if steps:
    cfg["optimization"]["steps"] = int(steps)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"preprocessed_data_root={cfg['data']['preprocessed_data_root']} output_dir={cfg['output_dir']} steps={cfg['optimization']['steps']}")
PY

printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${PREPROCESSED_NAME}" >> logs/sng_pvc/INDEX.tsv

# --num_processes must match NUM_XPUS * num nodes, and the config's
# checkpoints.interval assumes effective_batch 32 (see the config's own
# comments) -- recompute both if NUM_XPUS changes.
srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/train.py "${RUN_CONFIG}"

echo "END TIME: $(date)"
