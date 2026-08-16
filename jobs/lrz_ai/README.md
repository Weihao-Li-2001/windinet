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
sbatch jobs/lrz_ai/finetune_vae_1gpu.job                                    # 256res baseline (default since 2026-08-16)
sbatch jobs/lrz_ai/finetune_vae_1gpu.job configs/finetune_vae/other.yaml    # or pass one
sbatch jobs/lrz_ai/finetune_vae_2gpu.job                                    # 2-GPU variant
sbatch jobs/lrz_ai/finetune_vae_4gpu.job                                    # 4-GPU variant
```

**Default config/data changed 2026-08-16**: `finetune_vae_1gpu.job`/`2gpu.job`/
`4gpu.job` now default to
`finetune_vae_whole_structure_baseline_ep30_256res.yaml` against
`256x256_ds` (256x256 is the fixed baseline resolution going forward) --
previously the ancient 15-epoch frozen-trunk `finetune_vae_baseline.yaml`
against `128x128_ds`. Their `--time=06:00:00` header has **not** been
updated to match (still sized for the old queue-wait diagnostic default);
override with `sbatch --time=<HH:MM:SS> ...` for a real 256res run until
lrz_ai's actual per-epoch throughput and the partition's `MaxTime` are
known -- see `finetune_vae_whole_structure_baseline_ep30_256res.yaml`'s own
header. `finetune_vae_batchsize.job` is untouched (still defaults to the
128-res whole-structure baseline, tied to Open Question 23's own design).

`finetune_vae_2gpu.job`/`finetune_vae_4gpu.job` (added 2026-08-16) are the
same script as `finetune_vae_1gpu.job` -- only the `#SBATCH --gres`/
`--cpus-per-task` header differs, since `num_processes`/`--nproc_per_node`
are already derived from `SLURM_GPUS_ON_NODE` at runtime, not hard-coded.
`--cpus-per-task` is 48 for 2-GPU (confirmed working, job 5750198) and 92
for 4-GPU. `sinfo` reports 96 cpus/node (raw hardware, `GRES gpu:4(S:0-1)`,
GPUs socket-bound across 2 sockets), but the real allocatable max is lower:
`scontrol show node lrz-hgx-h100-001` shows `CoreSpecCount=2` reserving 2
cores (`CPUSpecList=46-47,94-95` at `ThreadsPerCore=2`) for system overhead,
leaving `CPUEfctv=92` / `CfgTRES cpu=92` -- confirmed partition-wide via
`scontrol show partition lrz-hgx-h100-94x4`'s `TRES=cpu=2760` / 30 nodes =
92 each. Two earlier guesses (96 -- `sinfo`'s raw total; 94 -- reading the
partition name `lrz-hgx-h100-94x4` too literally) both failed submission
outright with "CPU count per node can not be satisfied" before landing on
92. (An even earlier attempt scaled proportionally from the 1-GPU script's
own 48 -- 96/192 -- which also failed: 192 exceeds even the raw 96-core
total.)

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
