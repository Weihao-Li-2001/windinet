# Closed experiments

Configs in this folder have finished, their verdict is written into
`EXPERIMENTS.md` (repo root), and no further runs against them are planned.
They stay here (rather than getting deleted) because they are the record a
ledger entry points back to, not because anyone expects to relaunch them.

Corresponding outputs (metrics, resolved config) live under
`finetune_vae_outputs/sng_pvc/archive/done/<run>/` and
`finetune_vae_outputs/lundquist/archive/done/<run>/`.

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
- **Capacity diagnostic re-run under whole-structure unfreeze** (Open
  Question 12) (`finetune_vae_overfit_lr5e5_wsd_wholestruct.yaml`) -- see
  "Capacity diagnostic re-run under whole-structure unfreeze" in
  `EXPERIMENTS.md`. 50-epoch 8-sim overfit ceiling stayed at 0.05867,
  unfreezing the whole trunk did not raise it above the tail-only ceiling.
- **Decoder LR / adapter LR multiplier sweep** (Open Question 9), all 4
  arms (`finetune_vae_whole_structure_baseline_{declr0p5x,declr2x,
  adapterlr0p3x,adapterlr3x}.yaml`) -- see "Decoder LR / adapter LR
  multiplier sweep" in `EXPERIMENTS.md`. Both axes' existing values
  (decoder LR 5e-5, adapter_lr_multiplier 1.0x) confirmed at/near optimum;
  no config change.
- **Copy-init for a physically-paired momentum channel, original arm**
  (Open Question 10) (`finetune_vae_baseline_chorder_pr_my_mx_copyinit.yaml`,
  frozen-trunk lineage) -- see "Copy-init for a physically-paired momentum
  channel" in `EXPERIMENTS.md`. Null result vs mean-init (0.09831 vs
  0.09841). Its mean-init sibling
  (`finetune_vae_baseline_chorder_pr_my_mx.yaml`) was already archived
  above under the channel-order sweep.
- **Log-density experiment** (Open Question 11)
  (`finetune_vae_whole_structure_baseline_logdensity.yaml`) -- see
  "Log-density experiment" in `EXPERIMENTS.md`. Borderline improvement
  (-0.97%), just under the ~1.1% noise floor; no config change.
- **RMSE-only loss ablation** (Open Question 13)
  (`finetune_vae_whole_structure_baseline_rmseonly.yaml`) -- see
  "RMSE-only loss ablation" in `EXPERIMENTS.md`. Zeroing h1/ssim made
  val_vrmse +1.97% worse, a real effect -- confirms both terms are
  net-positive regularizers, not just gradient dilution.
- **18-epoch budget confirmation** (Open Question 7)
  (`finetune_vae_whole_structure_baseline_ep18.yaml`) -- see "Was 15
  epochs actually enough" in `EXPERIMENTS.md`. Confirmed the epoch-18 gain
  (0.08344); its `optimization.epochs: 18` setting was then promoted into
  `finetune_vae_whole_structure_baseline.yaml` itself (2026-08-09), same
  treatment as the encoder-LR sweep's winner above, so this config is now
  redundant with the active baseline.
- **Channel-order sweep v2 + copy-init extension** (Open Questions 8/10
  revisited), all 12 arms (`finetune_vae_whole_structure_baseline_chorder_
  {pr_my_mx,pr_mx_my,mx_pr_my,my_pr_mx,pr_d_mx_my,pr_d_my_mx}.yaml` and
  their `_copyinit` siblings) -- see "Channel-order sweep v2 + copy-init
  extension" in `EXPERIMENTS.md`. Baseline reconfirmed (0.08360, job
  523577); pressure-last mechanism persists under whole-structure unfreeze
  (no arm beats the default); copy-init stays null on 5/6 arms, one
  exception (`pr_d_my_mx`, -2.00%) flagged for a closer look, not enough
  to overturn the null verdict; density's index-0 position confirmed to
  have no special status.
- **Loss-function retest batch** (Open Questions 14/16/17/18/19)
  (`finetune_vae_whole_structure_baseline_{h2,h1x2,ssimx2,mlw,kl}.yaml`)
  -- see "H2 loss term", "Existing-loss weight retests", and "KL
  divergence weight" in `EXPERIMENTS.md`. h1x2/ssimx2/kl land inside the
  ~1.1% noise floor (no effect on val_vrmse; kl additionally confirmed to
  cut the latent space's own KL divergence ~359x, a real regularization
  win worth carrying into the DiT stage). h2 and mlw both land outside the
  floor as real regressions -- neither adopted, both stay at weight 0.
  (GradNorm, the sixth arm of this batch, is filed under `known-bad/`
  instead -- it never converged, see that folder's README.)

If you want to relaunch one of these as a sanity check, the config still
works as-is (`sbatch <job script> configs/finetune_vae/archive/done/<name>.yaml`)
-- moving it here only means it isn't part of the currently-open question set.
