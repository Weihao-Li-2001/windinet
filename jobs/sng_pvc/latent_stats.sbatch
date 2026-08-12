#!/bin/bash
#SBATCH --job-name=wh_latent_stats
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err
# i20r02c04s06 has a broken/stale scratch (hppfs) mount as of 2026-08-12:
# jobs 523911/523913/523914 all landed there and died with
# FileNotFoundError on euler_mq_dataset/128x128_ds/train.h5, while the same
# config on i20r02c04s04 (523910, 523912) read it fine. Exclude until LRZ
# fixes the mount; remove this line once that's confirmed.
#SBATCH --exclude=i20r02c04s06

# Batch submission for scripts/latent_stats.py -- encode-only diagnostic
# (no training, no gradients), see that script's own docstring for the
# full motivation (PhD-advisor-requested, 2026-08-10: mean/std shift from
# raw data to latent space, and from the freshly-inflated pretrained
# encoder to the current finetuned checkpoint).
#
# Single tile, plain python (not accelerate launch) -- this is a forward
# pass through vae.encode() only, no distributed training machinery
# needed, same reasoning as jobs/sng_pvc/compute_channel_stats.sbatch.
#
# TIME LIMIT 03:00:00 (raised from the original 01:00:00, 2026-08-12):
# job 523900/523901's own default (--num-samples 32) finished in ~20min
# including two full LTX-2B model loads, but that 32 was later found to be
# an unrepresentative slice of the 675-sim eval split (see EXPERIMENTS.md)
# -- the recommended invocation now passes the full eval_sims count
# (675), ~21x more encode() calls than the run this file's original limit
# was sized for. No checkpointing / partial output (same caveat as
# compute_channel_stats.sbatch) -- getting killed mid-run wastes the
# whole thing, so this errs generous rather than tight.
#
# Usage (all args optional):
#   sbatch jobs/sng_pvc/latent_stats.sbatch [config_yaml] [checkpoint] [num_samples]
#
# Submit with:   sbatch jobs/sng_pvc/latent_stats.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_latent_stats.out
# Cancel with:   scancel <jobid>

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml}
CHECKPOINT=${2:-"${SCRATCH}/windinet/finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline/checkpoints/vae_shockwave_best.safetensors"}
NUM_SAMPLES=${3:-32}
OUTPUT_JSON="log_finetuning_vae/sng_pvc/${SLURM_JOB_ID}-latent_stats.json"

# latent_stats.py reads data.data_root straight from the config file with no
# cluster patching of its own -- every checked-in config's data_root is a
# lundquist-style literal path (/local/disk/hramachandran/...) that doesn't
# exist on sng_pvc, so without this it fails immediately (job 523692:
# FileNotFoundError). Same fix jobs/sng_pvc/finetune_vae.sbatch already
# applies for training runs -- patch a temp copy via patch_config_for_cluster
# rather than editing the checked-in config.
RUN_CONFIG=$(mktemp /tmp/latent_stats_sng_pvc_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT
python - "${CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
# num_processes=1/output_suffix="": this isn't a training launch, so the
# gradient_accumulation_steps and output_dir side effects of the patch are
# irrelevant -- only data.data_root and data.num_dataloader_workers matter
# to latent_stats.py.
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=1, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

echo "config=${CONFIG} (patched -> ${RUN_CONFIG}) checkpoint=${CHECKPOINT} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON}"

python scripts/latent_stats.py "${RUN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
