#!/bin/bash
#SBATCH --job-name=wh_channel_stats
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # NOT "test" (30 min cap) -- see caveat below
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Batch submission for scripts/compute_channel_stats.py -- streams an entire
# ShockWave HDF5 dataset (density/momentum_x/momentum_y/pressure/
# log_density/log_pressure) and reports mean/std/min/max per channel, e.g.
# to sanity-check configs/finetune_vae/euler_mq_128_only_train.yaml's
# hand-computed data_normalization_stats block against the real data.
#
# WHY partition=general, NOT "test": a full 128x128_ds scan is ~4500
# simulations, ~120GB of raw float32 data, read by a SINGLE process with no
# parallelism (unlike training's 8-rank x 4-worker dataloader) -- unknown
# whether it fits in test's 30 min cap, and the script only writes its
# table/JSON once at the very end (no partial output / checkpointing), so
# getting killed mid-scan wastes the entire run rather than losing a little
# progress. 02:00:00 leaves generous headroom; if it finishes early the
# allocation is simply freed.
#
# WHY sbatch and not a bare `srun` in your terminal: Ctrl-C does not
# reliably stop this script while it is blocked inside a single h5py/
# libhdf5 C read call (Python only checks for pending signals between
# bytecode instructions, and a big read over the network filesystem via an
# external-link shard file can take a while) -- and exiting your local
# shell does NOT cancel the remote job step, it just stops you watching it
# while it keeps running and holding the allocation. sbatch runs
# unattended; watch with `tail -f`, cancel for real with `scancel <jobid>`.
#
# Usage (all three args optional, same order as compute_channel_stats.py's
# own arguments -- pass a different --h5 to reuse this unchanged once
# 256x256_ds / 512x512_orig are downloaded):
#   sbatch jobs/sng_pvc/compute_channel_stats.sbatch \
#       [h5_path] [compare_to_yaml] [output_json]
#
# Submit with:   sbatch jobs/sng_pvc/compute_channel_stats.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_channel_stats.out
# Cancel with:   scancel <jobid>

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"

H5_PATH=${1:-"${SCRATCH}/windinet/euler_mq_dataset/128x128_ds/train.h5"}
COMPARE_TO=${2:-configs/finetune_vae/euler_mq_128_only_train.yaml}
OUTPUT_JSON=${3:-"$(dirname "${H5_PATH}")/channel_stats_recomputed.json"}
echo "h5_path=${H5_PATH} compare_to=${COMPARE_TO} output=${OUTPUT_JSON}"

python scripts/compute_channel_stats.py "${H5_PATH}" \
    --compare-to "${COMPARE_TO}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
