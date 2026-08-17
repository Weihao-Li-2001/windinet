#!/bin/bash
#SBATCH --job-name=wh_latent_shift
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err
# See jobs/sng_pvc/latent_stats.sbatch's exclusion note -- add back
# "--exclude=i20r02c04s06" here too if that node's scratch mount is still
# broken when this is submitted.

# sng_pvc launcher for scripts/latent_shift_metrics.py -- paired z_before
# (zero-init, freshly-inflated) vs z_after (finetuned) latent-shift
# diagnostic. Mirrors jobs/lundquist/latent_shift_metrics.sbatch (same
# script, same protocol -- see latent_space_shift_measure.md), but reuses
# jobs/sng_pvc/latent_stats.sbatch's cluster-config patching, since
# data.data_root in every checked-in config is a lundquist-native
# /local/disk/... path that doesn't exist on sng_pvc.
#
# UNVERIFIED (2026-08-17): assumes
# ${SCRATCH}/windinet/euler_mq_dataset/128x128_ds/test.h5 (and the 256x256_ds
# sibling, if measuring the 256res checkpoint) exists next to train.h5 --
# confirm with `ls` before submitting. If missing, the job dies immediately
# on ShockWaveDataset(Path(test_h5)) with no partial output.
#
# Usage (all args optional):
#   sbatch jobs/sng_pvc/latent_shift_metrics.sbatch [config_yaml] [checkpoint] [test_h5] [num_samples]
#
# Submit with:   sbatch jobs/sng_pvc/latent_shift_metrics.sbatch
# Watch with:    tail -f log_finetuning_vae/sng_pvc/<jobid>-wh_latent_shift.out
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
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG=${1:-configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml}
CHECKPOINT=${2:-"${SCRATCH}/windinet/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep30/checkpoints/vae_shockwave_best.safetensors"}
TEST_H5=${3:-"${SCRATCH}/windinet/euler_mq_dataset/128x128_ds/test.h5"}
NUM_SAMPLES=${4:-64}
OUTPUT_JSON="log_finetuning_vae/sng_pvc/${SLURM_JOB_ID}-latent_shift_metrics.json"

# Same reasoning as jobs/sng_pvc/latent_stats.sbatch: every checked-in
# config's data.data_root is a lundquist-style literal path, so patch a temp
# copy for sng_pvc rather than editing the checked-in config. num_processes=1/
# output_suffix="" -- this isn't a training launch, only data.data_root and
# data.num_dataloader_workers matter to latent_shift_metrics.py.
RUN_CONFIG=$(mktemp /tmp/latent_shift_sng_pvc_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT
python - "${CONFIG}" "${RUN_CONFIG}" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=1, output_suffix="")
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

echo "config=${CONFIG} (patched -> ${RUN_CONFIG}) checkpoint=${CHECKPOINT} test_h5=${TEST_H5} num_samples=${NUM_SAMPLES} output=${OUTPUT_JSON}"

python scripts/latent_shift_metrics.py "${RUN_CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --test-h5 "${TEST_H5}" \
    --num-samples "${NUM_SAMPLES}" \
    --output "${OUTPUT_JSON}"

echo "END TIME: $(date)"
