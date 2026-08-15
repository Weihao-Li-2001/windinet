# lrz_ai cluster job scripts

Slurm + enroot/pyxis container launcher for lrz_ai (H100 partition
`lrz-hgx-h100-94x4`). Unlike sng_pvc/lundquist, training does not run
directly on the compute node's own filesystem -- it runs inside a container
(`--container-image=.../windinet1.sqsh`), with three host paths bind-mounted
in:

| host path | container path | what it is |
|---|---|---|
| `$HOME` | `$HOME` (1:1) | the actual repo checkout (`$HOME/work_wh/windinet`), used as the container's workdir |
| `$RESULT_DIR` (a DSS path, hard-coded in the job script header) | `/mnt/result` | where training writes `output_dir` -- **not the git repo** |
| `$DATA_DIR` (a DSS path, hard-coded in the job script header) | `/mnt/data` | the dataset (`train.h5` expected directly under it) |

`windinet/cluster_config.py`'s `CLUSTER_DEFAULTS["lrz_ai"]` uses the
*container-internal* paths (`/mnt/data/train.h5`, `/mnt/result`), since
`patch_config_for_cluster` always runs inside the container.

## Submitting

**Submit from the repo root**, same as sng_pvc:

```bash
sbatch jobs/lrz_ai/finetune_vae_1gpu.job                                    # baseline config
sbatch jobs/lrz_ai/finetune_vae_1gpu.job configs/finetune_vae/other.yaml    # or pass one
sbatch jobs/lrz_ai/finetune_vae_2gpu.job                                    # 2-GPU variant
sbatch jobs/lrz_ai/finetune_vae_4gpu.job                                    # 4-GPU variant
```

`finetune_vae_2gpu.job`/`finetune_vae_4gpu.job` (added 2026-08-16) are the
same script as `finetune_vae_1gpu.job` -- only the `#SBATCH --gres`/
`--cpus-per-task` header differs, since `num_processes`/`--nproc_per_node`
are already derived from `SLURM_GPUS_ON_NODE` at runtime, not hard-coded.
`--cpus-per-task` is scaled proportionally from the 1-GPU script's 48
(96/192), unverified against this partition's actual per-node core count --
if a job stays `PENDING` with a node-configuration-not-available reason,
lower it.

Written to answer a queue-wait/wall-clock question: lrz_ai's 1-GPU queue
wait is under 2h but single-GPU training is slow -- these test whether
requesting 2/4 GPUs queues meaningfully longer, and how much wall-clock that
buys back. To read the queue wait once a job lands:

```bash
sacct -j <jobid> -X -o JobID,Submit,Start,Elapsed,State
```

## Output flow

Because `$RESULT_DIR`/`$DATA_DIR` are DSS paths outside the git repo (same
situation as sng_pvc's `$SCRATCH`), the job script's inner script mirrors
each run's metrics/config/visualizations back onto the repo checkout after
training finishes -- checkpoints stay on `$RESULT_DIR` only (large, and
already addressed via `log_finetuning_vae/lrz_ai/INDEX.tsv`'s `output_dir`
column):

```
/mnt/result/<run>/  (== $RESULT_DIR/<run>/ on the host)
    -> rsync --exclude=checkpoints/ ->
finetune_vae_outputs/lrz_ai/<run>/   (git-tracked)
```

`log_finetuning_vae/lrz_ai/INDEX.tsv` maps each job id -> config ->
container-internal `output_dir`, same convention as
`log_finetuning_vae/{sng_pvc,lundquist}/INDEX.tsv`. See
`../../EXPERIMENTS.md` for results and rationale.

## Why the inner script is a separate file

The config-patch/training/mirror sequence has to run *inside* the container
(that's the only place with the right conda env, GPUs, and `/mnt/data`+
`/mnt/result`), but `$HOME` is bind-mounted 1:1, so the job script writes
that sequence to `$HOME/.windinet_lrz_ai_run_<jobid>.sh` before calling
`srun`, then just runs that file inside the container -- easier to read and
debug than one giant escaped `bash -lc "..."` string, and it's cleaned up on
exit via `trap`.
