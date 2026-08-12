#!/bin/bash
#SBATCH --job-name=wh_weight_drift
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Batch submission for scripts/inflate_weight_drift.py -- CPU-only,
# no-training weight-space diagnostic (no dataset, no forward pass): loads
# a freshly-inflated (pretrained, no finetuning) VAE and a finetuned
# checkpoint, compares the fresh channel's encoder.conv_in/decoder.conv_out
# weight blocks between the two. Same "plain python, not accelerate
# launch" reasoning as jobs/sng_pvc/compute_channel_stats.sbatch and
# jobs/sng_pvc/latent_stats.sbatch -- no distributed training machinery
# needed. `python` is not on PATH on the login node directly (needs the
# micromamba env activated below), same reason those two scripts are run
# via sbatch rather than a bare shell command.
#
# Usage (all args optional):
#   sbatch jobs/sng_pvc/inflate_weight_drift.sbatch [config_yaml] [checkpoint]
#
# Submit with:   sbatch jobs/sng_pvc/inflate_weight_drift.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_weight_drift.out
# Cancel with:   scancel <jobid>

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml}
CHECKPOINT=${2:-"${SCRATCH}/windinet/finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline/checkpoints/vae_shockwave_best.safetensors"}
OUTPUT_JSON="log_finetuning_vae/sng_pvc/${SLURM_JOB_ID}-weight_drift.json"

echo "config=${CONFIG} checkpoint=${CHECKPOINT} output=${OUTPUT_JSON}"

python scripts/inflate_weight_drift.py "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
