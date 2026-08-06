# Closed experiments

Configs in this folder have finished, their verdict is written into
`EXPERIMENTS.md` (repo root), and no further runs against them are planned.
They stay here (rather than getting deleted) because they are the record a
ledger entry points back to, not because anyone expects to relaunch them.

Corresponding outputs (metrics, resolved config) live under
`finetune_vae_outputs_sng_pvc/archive/done/<run>/` and
`finetune_vae_outputs_lundquist/archive/done/<run>/`.

What's here:

- **Capacity diagnostics, attempts 1/3/4/5/6** (`finetune_vae_overfit.yaml`,
  `_lr25e5.yaml`, `_lr5e5_wsd.yaml`, `_lr5e5_wsd_tail.yaml`,
  `_lr5e5_wsd_tail_slowdecay.yaml`) -- see "Capacity diagnostic (Open
  Question 2)" and "Attempt 5/6" in `EXPERIMENTS.md`.
- **Full-data head-vs-tail unfreeze sweep**, all 8 arms
  (`finetune_vae_baseline_{fullenc,head0,head01,head012,head01tail,tail,tail2,tail3}.yaml`)
  plus the schedule-shape check (`finetune_vae_baseline_slowdecay.yaml`) --
  see "Encoder head-vs-tail unfreeze sweep" in `EXPERIMENTS.md`. Winner
  (`fullenc`) was promoted to `configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml`,
  which is where active work continues.

If you want to relaunch one of these as a sanity check, the config still
works as-is (`sbatch <job script> configs/finetune_vae/archive/done/<name>.yaml`)
-- moving it here only means it isn't part of the currently-open question set.
