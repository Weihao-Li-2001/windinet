#!/bin/bash
# Submit the lundquist cosine KL sweep's kl0/kl1e7 DiT arms strictly
# sequentially -- never in parallel.
#
# Jobs 22753/22754 and again 22755/22756 (2026-09-12) OOM'd because both
# arms were submitted within a few minutes of each other and landed on the
# SAME node with the SAME `gpus=0,1` -- lundquist's debug partition does
# not appear to give multi-GPU jobs exclusive GPU isolation the way
# sng_pvc/lrz_ai do. The second job's accelerate launch even logged "Port
# `29500` is already in use" (proof the first job was still resident), and
# both then trained double-booked on the same 2 physical GPUs until they
# both OOM'd around step 120. batch_size=4/accum=4 (the fix in commit
# 87ee0b5) is very likely fine in isolation -- the actual repeat failure
# was submitting the second job before the first had exited, not the batch
# size. Manually remembering to check `squeue` before submitting the next
# one has now failed twice, hence this wrapper.
#
# Chains the kl1e7 arm via --dependency=afterany so Slurm will not start it
# until the kl0 job has actually exited (successfully, OOM'd, or
# time-limit-killed -- afterany, not afterok, since even a killed kl0 job
# frees the GPUs kl1e7 needs).
#
# Run from the repo root on lundquist (/local/disk/hramachandran/work/wh_work/windinet):
#   bash jobs/lundquist/run_kl_sweep_sequential.sh
set -euo pipefail

KL0_JOB=$(sbatch --parsable --time=12:00:00 jobs/lundquist/train_dit_2gpu.sbatch \
    finetune_vae_whole_structure_baseline_ep20_kl0 \
    configs/dit/train_dit_lundquist_cosine_kl0.yaml)
echo "Submitted kl0 arm: job ${KL0_JOB}"

KL1E7_JOB=$(sbatch --parsable --time=12:00:00 --dependency=afterany:${KL0_JOB} \
    jobs/lundquist/train_dit_2gpu.sbatch \
    finetune_vae_whole_structure_baseline_ep20_kl1e7 \
    configs/dit/train_dit_lundquist_cosine_kl1e7.yaml)
echo "Submitted kl1e7 arm (waits for ${KL0_JOB} to exit): job ${KL1E7_JOB}"
