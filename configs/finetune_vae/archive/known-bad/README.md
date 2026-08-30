# Known-bad -- do not reuse these numbers

**2026-08-16: `adapter.unfreeze_down_blocks`/`unfreeze_encoder_tail` and
`optimization.adapter_lr_multiplier` were removed from the config schema**
(hardcoded to whole-structure-unfreeze / 1.0x, since every current config
already used those values -- see `windinet/training/vae_trainer.py`'s
`_UNFROZEN_DOWN_BLOCKS`). Configs below that varied `unfreeze_down_blocks`/
`unfreeze_encoder_tail` away from `[0,1,2]`/`true` -- the head-vs-tail sweep
arms (`_head0`, `_head01`, `_head012`, `_head01tail`, `_tail2`, `_tail3`) --
**can no longer be loaded as-is**; `VaeTrainerConfig(...)` now rejects the
now-unknown keys (`extra="forbid"`). This is intentional (a loud load
failure, not a silent re-run under the wrong unfreeze set) -- reproducing
one of these exactly would need reverting to the commit before this
removal, not just resubmitting the YAML.

Configs (and their output data, under
`finetune_vae_outputs/sng_pvc/archive/known-bad/` and
`finetune_vae_outputs/lundquist/archive/known-bad/`) in this folder produced
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

## `finetune_vae_whole_structure_baseline_gradnorm.yaml` (Open Question 15)

GradNorm adaptive loss weighting, job 523590. Killed by the 24h time limit
at epoch 14/18 (confirmed ~5x the per-batch cost of every fixed-weight
arm, as predicted before launch), but the real problem is upstream of the
timeout: `val_VRMSE` oscillated between ~0.17-0.23 and ~0.56-1.13 every
other epoch for all 14 completed epochs, never converging, in lockstep
with `mlw`'s adapted weight swinging between ~0.0 and ~3.8-4.0 each epoch.
No output dir here -- the run never reached a committed `metrics.csv`
(only the raw `.out`/`.err` logs, under
`logs/sng_pvc/523590-*`, survive). See "GradNorm loss
weighting" in `EXPERIMENTS.md` for the full per-epoch trace. Do not
relaunch with the current `alpha`/`weight_lr` defaults -- a real retry
would need a smaller `weight_lr` (or a different re-weighting cadence) to
even have a chance at damping the oscillation, and would still carry the
~5x cost.
