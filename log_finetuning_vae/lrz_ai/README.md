# Cluster logs

SLURM stdout+stderr for every job run on lrz_ai. `jobs/lrz_ai/*.job` writes
here directly (`#SBATCH -o log_finetuning_vae/lrz_ai/%j-%x.out`,
`-e log_finetuning_vae/lrz_ai/%j-%x.err`), so `%j` is the SLURM job id and
`%x` the job name -- one `.out`/`.err` pair per job, same convention as
`log_finetuning_vae/sng_pvc/`.

**`.gitkeep` must stay tracked.** SLURM does not create the output directory
-- if `log_finetuning_vae/lrz_ai/` is missing when a job starts, the job
dies before producing any output.

`INDEX.tsv` maps each job to the config it ran and the output_dir it wrote:

```
timestamp    job    log_or_name    config    output_dir
```

Rows are appended by the inner script `jobs/lrz_ai/*.job` writes and runs
inside the container (see that script's own comments) at job start.

Note the `output_dir` column here is the *container-internal* path
(`/mnt/result/<run>`, bind-mounted from `$RESULT_DIR` on the host) -- the
git-tracked mirror of that run's metrics/config/visualizations lives under
`finetune_vae_outputs/lrz_ai/<run>/` (checkpoints stay on `$RESULT_DIR`
only, same split as sng_pvc/lundquist).

Experiment results and rationale live in `../../EXPERIMENTS.md`, not here.
