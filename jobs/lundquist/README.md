# Lundquist cluster job scripts

Slurm launchers for this specific cluster. Live under top-level `jobs/`,
parallel to `scripts/`, because the two are not the same kind of thing: the
Python entry points in `scripts/*.py` run anywhere, these do not. Every file
here hard-codes:

- `#SBATCH --partition=debug` -- this cluster's partition name
- `PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python`
- `cd /local/disk/hramachandran/work/wh_work/windinet`
- `#SBATCH --output=log_finetuning_vae/lundquist/%x_%j.log`

Moving the repo to another machine means rewriting all five, not porting them.

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

| script | GPUs | accum | effective batch | launches |
|---|---|---|---|---|
| `finetune_vae_2gpu.sbatch` | 2 | 16 | 32 | `scripts/finetune_vae.py` |
| `finetune_vae_4gpu.sbatch` | 4 | 8 | 32 | `scripts/finetune_vae.py` |
| `finetune_vae_6gpu.sbatch` | 6 | 5 | 30 | `scripts/finetune_vae.py` |
| `train_dit.sbatch` | 4 | - | - | `scripts/train.py` |
| `finetune_vae_debug.sbatch` | 1 | 1 | 1 | `scripts/finetune_vae.py` |

The 2- and 4-GPU variants patch `gradient_accumulation_steps` so both land on
effective batch 32 and 1800 optimizer steps -- results from them are directly
comparable. **The 6-GPU variant is not** (batch 30, different step count); do
not compare its `val_vrmse` against the ledger without noting this.

`finetune_vae_debug.sbatch` is a different kind of script, not a smaller version of the
others: its default config is `finetune_vae_overfit.yaml`, the overfit-8
capacity diagnostic, which measures memorization and needs one optimizer step
per sim, not a comparable effective batch. It patches `effective_batch=1`
explicitly rather than taking the cluster default of 32, and its `val_vrmse`
is not comparable to the ledger at all -- see `EXPERIMENTS.md` Open
Questions #2.

**All three write to the same `output_dir`** (`finetune_vae_outputs/lundquist/<run>`,
no per-GPU-count suffix -- see `windinet/cluster_config.py`), on purpose:
there's one current lundquist result, not one per GPU count. `clean_output_dir:
true` wipes that directory at the start of every run, so submitting a
different variant overwrites whatever the previous one produced. If you need
to keep results from two variants side by side, copy the output_dir out
before submitting the next one.

Each job appends a row to `log_finetuning_vae/lundquist/INDEX.tsv` mapping job
id -> config -> output_dir. See `../../EXPERIMENTS.md` for results and
rationale.
