# Cluster logs

SLURM stdout+stderr for every job run on sng_pvc. The sbatch scripts write
here directly (`#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out`,
`-e log_finetuning_vae/sng_pvc/%j-%x.err`), so `%j` is the SLURM job id and
`%x` the job name -- one `.out`/`.err` pair per job, unlike
log_finetuning_vae/lundquist/'s single combined `.log` file per job.

**`.gitkeep` must stay tracked.** SLURM does not create the output directory --
if `log_finetuning_vae/sng_pvc/` is missing when a job starts, the job dies
before producing any output, and on a fresh clone the directory would
otherwise not exist (only the files listed in `.gitignore`'s
`!log_finetuning_vae/sng_pvc/*.err`/`*.out` exception are tracked, so an empty
repo checkout needs something else to keep the directory itself present).

`INDEX.tsv` maps each job to the config it ran and the output_dir it wrote:

```
timestamp    job    log_or_name    config    output_dir
```

Rows are appended by `jobs/sng_pvc/finetune_vae.sbatch` at job start, same
convention as `log_finetuning_vae/lundquist/INDEX.tsv`. Rows dated before this
file existed were backfilled from log contents -- best-effort, some fields
truncated in the source log and marked `-`. Rows from before the 2026-08-01
`configs/`/`jobs/`/`log_sng_pvc/` reorg (e.g. `configs/finetune_vae_baseline.yaml`)
record the paths that were actually current at the time -- a historical log,
not something to rewrite.

Experiment results and rationale live in `../../EXPERIMENTS.md`, not here.
