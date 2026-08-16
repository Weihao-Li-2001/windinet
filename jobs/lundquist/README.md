# Lundquist cluster job scripts

Slurm launchers for this specific cluster. Live under top-level `jobs/`,
parallel to `scripts/`, because the two are not the same kind of thing: the
Python entry points in `scripts/*.py` run anywhere, these do not. Every file
here hard-codes:

- `#SBATCH --partition=debug` -- this cluster's partition name
- `PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python`
- `cd /local/disk/hramachandran/work/wh_work/windinet`
- `#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log`

Moving the repo to another machine means rewriting all four, not porting them.

## Submitting

**Submit from the repo root.** Slurm resolves `--output` relative to the
*submission* directory, not the script's location, so `log_finetuning_vae/lundquist/`
must exist relative to wherever you run `sbatch`:

```bash
cd /local/disk/hramachandran/work/wh_work/windinet
sbatch jobs/lundquist/finetune_vae_4gpu.sbatch                                    # baseline config
sbatch jobs/lundquist/finetune_vae_4gpu.sbatch configs/finetune_vae/other.yaml    # or pass one
```

The scripts themselves `cd` to an absolute repo path before doing anything, so
every path *inside* them (`scripts/finetune_vae.py`,
`log_finetuning_vae/lundquist/INDEX.tsv`) resolves regardless of where the
script lives.

## What each one does

| script | GPUs | batch_size | accum | effective batch | launches |
|---|---|---|---|---|---|
| `finetune_vae_2gpu.sbatch` | 2 | 16 | 1 | 32 | `scripts/finetune_vae.py` |
| `finetune_vae_4gpu.sbatch` | 4 | 8 | 1 | 32 | `scripts/finetune_vae.py` |
| `train_dit.sbatch` | 4 | - | - | - | `scripts/train.py` |
| `finetune_vae_debug.sbatch` | 1 | 1 | 1 | 1 | `scripts/finetune_vae.py` |

The 2- and 4-GPU variants patch `gradient_accumulation_steps` so both land on
effective batch 32 and 1800 optimizer steps -- results from them are directly
comparable.

**`batch_size=16` on `finetune_vae_2gpu.sbatch` (2026-08-16):** the largest
batch_size that still divides effective_batch=32 at 2 GPUs (accum=1) --
adopted as the production default per `EXPERIMENTS.md` Open Question 23
(closed 2026-08-16): job 21991 (batch_size=16) ran ~18.7% faster than job
21989 (batch_size=1) for 18 epochs, with val_vrmse inside the "real effect"
bar.

**`batch_size=8` on `finetune_vae_4gpu.sbatch` (2026-08-17):** same lever
applied to the 4-GPU launcher -- the largest batch_size that still divides
effective_batch=32 at 4 GPUs (accum=1). UNVALIDATED AT 4 GPUS: Open Question
23 only measured batch_size=1 vs 16 at 2 GPUs; this carries the same
reasoning over rather than a 4-GPU measurement of its own. Watch val_vrmse
against the batch_size=1 baseline (job 21992, 0.076314) the same way.

**Retired (2026-08-15): the 6-GPU variant.** `finetune_vae_6gpu.sbatch` is
gone -- it targeted effective batch 30 (32 isn't divisible by 6 ranks), a
different step count from every other launcher here, so its results were
never directly comparable to the ledger without a caveat. No longer in use;
its historical runs (job 21618 and similar, `vae_inflate4_tail3x` etc. in
`EXPERIMENTS.md`) stand as-is, just not reproducible via a checked-in script
anymore.

**Retired (2026-08-16): `finetune_vae_batchsize.sbatch`.** Swept
`batch_size` in {1, 16} (accum derived to hold effective_batch=32) at a
fixed 2 GPUs -- its results (jobs 21989/21990/21991) are what justified the
`batch_size=16` default above. No further use once Open Question 23 closed;
removed rather than left around as dead config surface, same precedent as
the 6-GPU variant. Its historical runs stand as-is in `EXPERIMENTS.md`.

`finetune_vae_debug.sbatch` is a different kind of script, not a smaller version of the
others: its default config is `finetune_vae_overfit.yaml`, the overfit-8
capacity diagnostic, which measures memorization and needs one optimizer step
per sim, not a comparable effective batch. It patches `effective_batch=1`
explicitly rather than taking the cluster default of 32, and its `val_vrmse`
is not comparable to the ledger at all -- see `EXPERIMENTS.md` Open
Questions #2.

**The 2- and 4-GPU variants write to the same `output_dir`**
(`finetune_vae_outputs/lundquist/<run>`, no per-GPU-count suffix -- see
`windinet/cluster_config.py`), on purpose: there's one current lundquist
result, not one per GPU count. `clean_output_dir: true` wipes that directory
at the start of every run, so submitting a different variant overwrites
whatever the previous one produced. If you need to keep results from two
variants side by side, copy the output_dir out before submitting the next
one.

Each job appends a row to `log_finetuning_vae/lundquist/INDEX.tsv` mapping job
id -> config -> output_dir. See `../../EXPERIMENTS.md` for results and
rationale.
