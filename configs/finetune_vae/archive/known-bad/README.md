# Known-bad -- do not reuse these numbers

Configs (and their output data, under
`finetune_vae_outputs_sng_pvc/archive/known-bad/` and
`finetune_vae_outputs_lundquist/archive/known-bad/`) in this folder produced
results `EXPERIMENTS.md` explicitly flags as uninterpretable. They are kept
for provenance, not as a source of numbers to cite or compare against.

## 8-sim diagnostic tier of the head-vs-tail sweep

`finetune_vae_overfit_lr5e5_wsd_{fullenc,head0,head01,head012,head01tail,tail2,tail3}.yaml`

Used `overfit_repeat: 1` (8 opt-relevant samples/epoch) instead of the
capacity-diagnostic attempts' `overfit_repeat: 5`. With `warmup_steps: 50`
and ~2 optimizer steps/epoch, roughly the first 25 of 50 epochs never left
warmup. All seven runs landed at val_vrmse 0.105-0.126 -- a warmup/steps-per-
epoch artifact, not a capacity signal. See "Caveat on the diagnostic (8-sim)
arm of this sweep" in `EXPERIMENTS.md`.

If this diagnostic tier is needed again (e.g. to cheaply screen a value
before spending full-data time), fix `overfit_repeat` or `warmup_steps`
first -- don't just relaunch these as-is.

## `finetune_vae_overfit_lr5e5` (lundquist output only, config already lost)

Capacity-diagnostic attempt 2: constant LR (no decay), oscillated between
0.10-0.30 val_vrmse across all 50 epochs and violated its own kill
criterion. See "Capacity diagnostic (Open Question 2)" attempt 2 in
`EXPERIMENTS.md`. Kept only because the metrics happen to survive on disk;
there is no config to relaunch.
