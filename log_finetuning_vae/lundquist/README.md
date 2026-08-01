# Cluster logs

Slurm stdout+stderr for every job run on this cluster. The sbatch scripts write
here directly (`#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log`), so
`%x` is the job name and `%j` the Slurm job id.

**`.gitkeep` must stay tracked.** Slurm does not create the output directory --
if `log_finetuning_vae/lundquist/` is missing when a job starts, the job dies
before producing any output, and on a fresh clone the directory would
otherwise not exist (`*.log` is gitignored, so the logs themselves are not).

`INDEX.tsv` maps each job to the config it ran and the output_dir it wrote:

```
timestamp    job    log_or_name    config    output_dir
```

Rows are appended by the sbatch scripts at job start. Rows dated before
2026-08-01 were backfilled from file mtime and log contents -- the `job` and
`config` columns are best-effort there, and the four `train*.log` files predate
the current naming and could not be attributed.

Experiment results and rationale live in `../../EXPERIMENTS.md`, not here.
