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
- **Encoder-LR sweep** (Open Question 6), all 4 arms
  (`finetune_vae_whole_structure_baseline_enclr{0p03x,0p3x,1x}.yaml`, plus
  the baseline's own 0.1x reconfirmation run under its old settings, job
  521885 -- no separate archived config for that one since it *was*
  `finetune_vae_whole_structure_baseline.yaml` at the time) -- see "PLANNED,
  not yet run: encoder learning-rate sweep" / the encoder-LR sweep readout
  in `EXPERIMENTS.md`. Winner (`0.3x`, val_vrmse 0.08516, job 521887) was
  promoted into `configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml`
  itself (`encoder_tail_lr_multiplier: 0.1 -> 0.3`, 2026-08-08) -- so
  `finetune_vae_whole_structure_baseline_enclr0p3x.yaml` is archived too,
  even though it "won," because its settings now live in the active
  baseline file under a different name, same treatment `fullenc` got above.
- **Channel-order sweep** (Open Question 8), the 5 non-default arms
  (`finetune_vae_baseline_chorder_{mx_pr_my,my_mx_pr,my_pr_mx,pr_mx_my,pr_my_mx}.yaml`)
  -- see "Channel-order sweep" in `EXPERIMENTS.md`. All 6 arms (including
  the default `mx_my_pr`, i.e. `finetune_vae_baseline.yaml` itself, kept
  active/unarchived since it's still the general frozen-trunk reference
  config) landed within the established mechanism: whichever field sits at
  index 3 (the fresh mean-init slot) sets the outcome, pressure-last best.
  `mx_my_pr` stays the default -- no config change needed, it was already
  in the winning group.
- **Seed noise floor sweep** (Open Question 1)
  (`finetune_vae_baseline_seed1.yaml`, `finetune_vae_baseline_seed2.yaml`)
  -- see "Seed noise floor sweep" in `EXPERIMENTS.md`. Measured the noise
  floor at ~1.1% (val_vrmse 0.09513/0.09519/0.09411 across seeds 42/1/2);
  no config change results from this, it's a calibration number used to
  judge every other sweep in the ledger.

If you want to relaunch one of these as a sanity check, the config still
works as-is (`sbatch <job script> configs/finetune_vae/archive/done/<name>.yaml`)
-- moving it here only means it isn't part of the currently-open question set.
