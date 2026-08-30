#!/bin/bash
#SBATCH --job-name=wh_queue_wait_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # "test" is max 30 min, "general" is max 24h
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# Minimal job for measuring how sng_pvc's backfill queue wait depends on the
# requested --time, holding everything else (nodes/ntasks/partition/account)
# fixed. --time is intentionally NOT set here -- pass it at submission so
# one script covers all four requested horizons:
#
#   sbatch --time=08:00:00 jobs/sng_pvc/queue_wait_test.sbatch
#   sbatch --time=10:00:00 jobs/sng_pvc/queue_wait_test.sbatch
#   sbatch --time=18:00:00 jobs/sng_pvc/queue_wait_test.sbatch
#   sbatch --time=24:00:00 jobs/sng_pvc/queue_wait_test.sbatch
#
# Submit all four back-to-back (same wave, seconds apart) so they see the
# same queue state -- otherwise cluster load drift between submissions would
# confound the --time effect you're trying to isolate. Queue wait is very
# load-dependent in general, so treat one wave as a single noisy sample --
# repeat at a different time of day if the result looks surprising.
#
# After all four have started (or finished), read off Submit/Start/Wait
# directly from SLURM's own accounting (more reliable than eyeballing squeue):
#   sacct -j <id1>,<id2>,<id3>,<id4> \
#       --format=JobID,ReqTIME%12,Submit,Start,Elapsed,State -X
# "Start - Submit" is the queue wait for each job.

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
python3 -c "print('hello from sng_pvc queue-wait test')"
echo "END TIME: $(date)"
