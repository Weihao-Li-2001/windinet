# Cluster logs

SLURM stdout+stderr for every job run on sng_pvc. The sbatch scripts write
here directly (`#SBATCH -o sng_pvc_log/%j-%x.out`, `-e sng_pvc_log/%j-%x.err`),
so `%j` is the SLURM job id and `%x` the job name -- one `.out`/`.err` pair per
job, unlike lundquist_log/'s single combined `.log` file per job.

**`.gitkeep` must stay tracked.** SLURM does not create the output directory --
if `sng_pvc_log/` is missing when a job starts, the job dies before producing
any output, and on a fresh clone the directory would otherwise not exist
(only the files listed in `.gitignore`'s `!sng_pvc_log/*.err`/`*.out`
exception are tracked, so an empty repo checkout needs something else to keep
the directory itself present).

`INDEX.tsv` maps each job to the config it ran and the output_dir it wrote:

```
timestamp    job    log_or_name    config    output_dir
```

Rows are appended by `scripts/sng_pvc/finetune_vae.sbatch` at job start, same
convention as `lundquist_log/INDEX.tsv`. Rows dated before this file existed
were backfilled from log contents -- best-effort, some fields truncated in
the source log and marked `-`.

Experiment results and rationale live in `../EXPERIMENTS.md`, not here.
