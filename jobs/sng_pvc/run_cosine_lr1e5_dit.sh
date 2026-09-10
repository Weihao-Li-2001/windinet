#!/bin/bash
# One-off: encode + train the "cosine schedule, KL=0" DiT control arm --
# see configs/dit/train_dit_sng_pvc_cosine_lr1e5.yaml's own header for why
# this arm matters (isolates whether the cosine LR schedule itself is
# implicated in cosine_kl1e5's broken eval result, independent of KL/anchor
# regularization).
#
# VAE checkpoint (job 529953, val_vrmse=0.065861) already exists -- this
# only needs the encode + DiT training steps.
#
# Chains train_dit via --dependency=afterok so it only starts once the
# encode job actually succeeds (not just once it's submitted) -- no need to
# babysit preprocess_dit_data's ~7h runtime and submit train by hand.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet).
set -euo pipefail

PREPROCESS_JOB=$(VAE_CHECKPOINT=/hppfs/scratch/0D/go76fuz2/windinet/finetune_vae_outputs_sng_pvc/finetune_vae_whole_structure_baseline_ep20_256res_cosine_lr1e5/checkpoints/vae_shockwave_best.safetensors \
    sbatch --parsable jobs/sng_pvc/preprocess_dit_data.sbatch)
echo "Submitted preprocess_dit_data: job ${PREPROCESS_JOB}"

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREPROCESS_JOB} \
    jobs/sng_pvc/train_dit.sbatch \
    finetune_vae_whole_structure_baseline_ep20_256res_cosine_lr1e5 \
    configs/dit/train_dit_sng_pvc_cosine_lr1e5.yaml)
echo "Submitted train_dit (waits on ${PREPROCESS_JOB}): job ${TRAIN_JOB}"
