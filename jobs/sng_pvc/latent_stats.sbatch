#!/bin/bash
#SBATCH --job-name=wh_latent_stats
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

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
CHECKPOINT=${2:-"${SCRATCH}/windinet/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline/checkpoints/vae_shockwave_best.safetensors"}
NUM_SAMPLES=${3:-32}
OUTPUT_JSON="log_finetuning_vae/sng_pvc/${SLURM_JOB_ID}-latent_stats.json"
echo "config=${CONFIG} checkpoint=${CHECKPOINT} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON}"

python scripts/latent_stats.py "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
