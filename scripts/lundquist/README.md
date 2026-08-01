# Lundquist cluster job scripts

Slurm launchers for this specific cluster. Separated from `scripts/*.py`
because the two are not the same kind of thing: the Python entry points run
anywhere, these do not. Every file here hard-codes:

- `#SBATCH --partition=debug` -- this cluster's partition name
- `PYTHON=/local/disk/hramachandran/miniconda/envs/windinet/bin/python`
- `cd /local/disk/hramachandran/work/wh_work/windinet`
- `#SBATCH --output=log_lundquist/%x_%j.log`

Moving the repo to another machine means rewriting all four, not porting them.

## Submitting

**Submit from the repo root.** Slurm resolves `--output` relative to the
*submission* directory, not the script's location, so `log_lundquist/` must
exist relative to wherever you run `sbatch`:

```bash
cd /local/disk/hramachandran/work/wh_work/windinet
sbatch scripts/lundquist/finetune_vae_4gpu.sbatch                       # baseline config
sbatch scripts/lundquist/finetune_vae_4gpu.sbatch configs/other.yaml    # or pass one
```

The scripts themselves `cd` to an absolute repo path before doing anything, so
every path *inside* them (`scripts/finetune_vae.py`, `log_lundquist/INDEX.tsv`)
resolves regardless of where the script lives.

## What each one does

| script | GPUs | accum | effective batch | launches |
|---|---|---|---|---|
| `finetune_vae_2gpu.sbatch` | 2 | 16 | 32 | `scripts/finetune_vae.py` |
| `finetune_vae_4gpu.sbatch` | 4 | 8 | 32 | `scripts/finetune_vae.py` |
| `finetune_vae_6gpu.sbatch` | 6 | 5 | 30 | `scripts/finetune_vae.py` |
| `train_dit.sbatch` | 4 | - | - | `scripts/train.py` |

The 2- and 4-GPU variants patch `gradient_accumulation_steps` so both land on
effective batch 32 and 1800 optimizer steps -- results from them are directly
comparable. **The 6-GPU variant is not** (batch 30, different step count); do
not compare its `val_vrmse` against the ledger without noting this.

Each job appends a row to `log_lundquist/INDEX.tsv` mapping job id -> config ->
output_dir. See `../../EXPERIMENTS.md` for results and rationale.
