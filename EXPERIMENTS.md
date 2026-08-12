# VAE Finetuning Experiment Ledger

Single source of truth for what has been run, what it showed, and what is still
open. Every new run gets a row here **before** it is launched (hypothesis) and a
verdict **after** it finishes.

- **Task**: LTX-Video 0.9.x VAE adapted to 4-channel Euler CFD fields
  (density, momentum_x, momentum_y, pressure), 128x128, inflate mode.
- **Metric**: `val_vrmse` on 675 held-out sims (fixed seed + randperm, identical
  across all runs).
- **Baseline (as of 2026-08-01)**: `finetune_vae_baseline.yaml`, **val_vrmse
  0.095396** (ledger #3). Its output dir no longer exists -- job 21664 re-ran
  into the same `output_dir` with `clean_output_dir: true` and was cancelled at
  step 20, wiping it. The only surviving record of that run is
  `log_finetuning_vae/lundquist/vae_2gpu_21632.log` (21664's own log is gone
  too, and it never got an INDEX.tsv row). Job **21666** is re-running the same
  config at the same seed to restore it.
- **New reference baseline (as of 2026-08-06)**: the encoder head-vs-tail
  sweep (Open Question 3, full results below) shows unfreezing the **entire
  encoder trunk** beats the frozen-trunk baseline by ~9% (`fullenc`,
  val_vrmse **0.08673**, job 521389, sng_pvc, vs 0.095396/0.09513 for the
  frozen-trunk baseline) and is statistically tied with the next-best arms
  (`head012` 0.08686, `tail3` 0.08684 -- see "Full-data follow-ups: full
  head-vs-tail sweep" below). This setup is promoted to a first-class,
  stably-named config, `configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml`
  (identical hyperparameters to `finetune_vae_baseline_fullenc.yaml`, new
  `output_dir`/tags only), so that follow-up sweeps -- starting with the
  learning-rate sweep planned below -- name themselves against a baseline
  tag that won't get confused with one arm of the now-finished unfreeze
  sweep. **Not yet run under its new name** -- next launch should confirm it
  reproduces ~0.0867 before the LR sweep branches off it.
- **New reference baseline (as of 2026-08-08)**: the encoder-LR sweep (Open
  Question 6, full results below) finished all 4 arms and shows
  `encoder_tail_lr_multiplier: 0.3x` beats the inherited `0.1x` default by
  ~1.7% (val_vrmse **0.08516**, job 521887, vs 0.08661 for 0.1x, job
  521885). `configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml`
  updated in place (`0.1x -> 0.3x`) so this is now what "the baseline" means
  going forward -- **every follow-up change should be diffed against 0.3x,
  not 0.1x**. The same pull also answered Open Question 1 (seed noise floor
  measured at **~1.1%**, see below): the 0.3x-vs-1.0x gap (0.08516 vs
  0.08584, ~0.8%) is inside that band, so 0.3x and 1.0x are not reliably
  distinguishable from each other -- 0.3x is adopted only because the
  sweep's read-out criterion picks the lowest number, not because it's
  proven better than 1.0x specifically. The file's own `output_dir` has not
  been re-run under the new value; until it is, job 521887's own directory
  (`finetune_vae_whole_structure_baseline_enclr0p3x`) is the authoritative
  reproduction of what "baseline" now means.

## Fixed setup (identical in every 15-epoch run)

| | |
|---|---|
| Base model | LTXV 2B, `spatial_compression_ratio=32`, `temporal=8`, `latent_channels=128` |
| Latent grid | 128x128 field -> **4x4** latents (each cell covers 32x32 px), ~250:1 |
| Trainable | decoder 552,869,953 + `encoder.conv_in` 221,312 |
| Frozen | encoder trunk 693,877,569 |
| Effective batch | 32 (2-GPU accum 16 and 4-GPU accum 8 both normalized by sbatch) |
| Total opt steps | 1800 (15 ep x 120), warmup 200 |
| Data | 3825 train / 675 eval |
| Seed | 42 (**every run** -- no repeats, see Open Questions) |

## Weight provenance

Both encoder and decoder start from **`Lightricks/LTX-Video-0.9.5`, subfolder
`vae`** -- a natural-video autoencoder. Nothing CFD-specific is in the init.

The resolution chain, which is not obvious from the config:

1. Config says `model_source: "LTXV_2B_0.9.6_DEV"`.
2. `vae_trainer._load_vae` -> `load_vae()` (`windinet/inference/model_loader.py:100`).
3. `LTXV_2B_096_DEV` is in the special-cased list and returns
   `from_pretrained(LTXV_2B_095.hf_repo, subfolder="vae")`. This is deliberate,
   not a bug -- `LTXV_2B_096_DEV.hf_repo` explicitly raises "does not have a
   HuggingFace repo". You write 0.9.6 and get the 0.9.5 VAE.
4. One file: `models--Lightricks--LTX-Video-0.9.5/.../vae/diffusion_pytorch_model.safetensors`
   (2.49 GB, bf16) -- encoder 694,098,881 + decoder 552,814,641 params.
5. `inflate_vae_io_channels(n=4, init="mean")` grows `encoder.conv_in` (3->4 ch,
   221,312 params after growth) and `decoder.conv_out`, seeding the new channel
   with the mean of the pretrained slots at the same patch position.
6. `resume_from: null` in the baseline, so nothing overrides steps 4-5.

**What the 0.9.5-vs-0.9.6 VAE substitution actually costs (measured, not
assumed).** The 0.9.6-dev single-file checkpoint bundles its own VAE
(`vae.*`, 1,246,914,162 params) alongside the DiT. Comparing it tensor-by-tensor
against the 0.9.5 diffusers VAE (both bf16, sha256 over raw bytes):

| part | tensors | byte-identical |
|---|---|---|
| `encoder.*` | 92 / 92 | **92 (all)** |
| `decoder.*` | 132 / 132 | 99 |

The 33 differing decoder tensors are all in the **final up-block and `conv_out`**
(`up_blocks.2.{resnets,time_embedder,upsamplers}` + `conv_out` in diffusers
naming; the same modules appear as `up_blocks.5/6` in the original naming, same
shapes). `per_channel_statistics.*` is present only in the single-file version.

Consequence: **the encoder -- and therefore the latent space -- is bit-identical
between 0.9.5 and 0.9.6-dev**, so pairing a 0.9.5-derived VAE with the 0.9.6-dev
DiT introduces no latent mismatch. Only the last decoder stage starts from
different weights, and the decoder is fully retrained anyway (552,869,953
trainable params), so this is an init difference, not a correctness problem.

### DiT (stage 2, not yet run)

Different repo, different file, different loader -- from the *same*
`model_source` string:

1. `shockwavenet.yaml`: `model_source: "LTXV_2B_0.9.6_DEV"`, `load_checkpoint: null`.
2. `dit_trainer._load_models` -> `load_ltxv_components(model_source)` ->
   `load_transformer()` (`windinet/inference/model_loader.py:226`).
3. `LTXV_2B_096_DEV` is not in the 13B list, so it takes
   `LTXVideoTransformer3DModel.from_single_file(source.safetensors_url)` --
   **not** `from_pretrained`, and **not** the 0.9.5 repo the VAE came from.
4. URL: `https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors`
   (6.34 GB, bf16). The `model.*` prefix in that file is the DiT:
   **1,923,385,472 params**.
5. `load_checkpoint: null` -> `_load_checkpoint()` returns early, nothing
   overrides.

So: **VAE from `LTX-Video-0.9.5` (diffusers layout), DiT from `LTX-Video` root
repo (single-file 0.9.6-dev).** Both are stock pretrained natural-video weights.

Unverified, worth checking before the DiT stage: `load_ltxv_components` also
constructs a VAE inside `dit_trainer`. Since DiT training consumes precomputed
latents from `preprocess_dataset.py` (encoded with the *finetuned inflate* VAE),
that trainer-side VAE is either unused or a 3-channel landmine.

## Ledger

| # | run dir | job | init | ep | sched | admul | h1 | mlw | val_vrmse | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | vae_meaninit_wsd15_h1x2_4gpu | 21627 | mean | 15 | wsd | 10 | 50 | 0 | **0.094739** | best overall |
| 2 | vae_meaninit_wsd15_4gpu | 21623 | mean | 15 | wsd | 10 | 25 | 0 | 0.094886 | h1 25 vs 50 = noise |
| 3 | vae_meaninit_wsd15_h1x2_lr1x_2gpu | 21632 | mean | 15 | wsd | 1 | 50 | 0 | **0.095396** | **BASELINE** |
| 4 | vae_meaninit_wsd15_h1x2_lr1x_mlw_4gpu | 21642 | mean | 15 | wsd | 1 | 50 | 1e-4 | 0.096671 | mlw net-negative, reverted |
| 5 | vae_inflate4_tail3x | 21618 | zeros | 10 | cosine | 10 | 25 | 0 | 0.101276 | superseded |
| 6 | vae_meaninit | - | mean | 10 | cosine | 10 | 25 | 0 | 0.101768 | superseded |
| 7 | vae_inflate4_resume3 | - | zeros | 10 | - | 10 | 0 | 0 | 0.104231 | superseded |
| 8 | vae_inflate4_more | 21547 | zeros | 10 | - | 10 | 25 | 0 | 0.104885 | superseded |
| 9 | vae_inflate4 | - | zeros | 5 | - | 10 | 0 | 0 | 0.117103 | superseded |
| 10 | vae_randinit_4gpu | 21619/21621 | random | 15 | wsd | 10 | 25 | 0 | 0.162905 | **clear failure** |

`admul` = `optimization.adapter_lr_multiplier` (applies to `encoder.conv_in`).

**Artifact retention (2026-08-01).** All ten run directories above have been
deleted to start the next round clean. What survives:

- **In git at `f88eb09`** -- metrics, loss curves, reconstruction panels and
  resolved configs for #1, #2, #4, #6, #8, #9 (plus the config-only remains of
  #3). Retrieve with
  `git show f88eb09:finetune_vae_outputs_lundquist/<run>/metrics/metrics.csv`.
- **Numbers in this table only** -- #5, #7, #10 were already gone before that
  commit; their Slurm logs are in `log_finetuning_vae/lundquist/`, which is
  where every lundquist log now lives (the old `lundquist_log/` and
  `log_lundquist/` directories are gone). `*.log` is gitignored, so those are
  on this disk only.
- **No weights at all.** Every `vae_shockwave_best.safetensors` was deleted,
  including #1's (0.094739), which was never in git -- `checkpoints/` is
  gitignored. There is currently no finetuned VAE, so stage 2 (DiT on
  precomputed latents) is blocked until 21666 finishes.

## In flight

| job | config | output dir | what it answers | read-out |
|---|---|---|---|---|
| 21666 | `finetune_vae_baseline.yaml` | `finetune_vae_baseline` | restores #3 (mean, 15 ep, wsd, admul 1, h1 50, mlw 0, seed 42) | **done: 0.095396**, matches #3 to 6 significant figures -- reproduction confirmed. Same seed, so this is **not** an answer to Open Question 1 -- that still needs seeds 1 and 2. |

### Capacity diagnostic (Open Question 2) -- RESOLVED: partial

Six attempts, all overfitting the same 8 sims (`overfit_sims: 8,
overfit_repeat: 5`, 40 opt steps/epoch, 1 GPU):

| attempt | config | job | lr | sched | best val_vrmse | verdict |
|---|---|---|---|---|---|---|
| 1 | `finetune_vae_overfit.yaml` (orig.) | 21667 | 2e-4 | constant | ~1.0 | diverged, measured nothing |
| 2 | `finetune_vae_overfit_lr5e5.yaml` | 21669/21677 | 5e-5 | constant | 0.0989 (ep48) | ran but untrustworthy: violated its own kill criterion, oscillated 0.10-0.30 all 50 epochs |
| 3 | `finetune_vae_overfit_lr25e5.yaml` | 21684/21686 | 2.5e-5 | constant | 0.0829 (ep46) | still oscillating, no clean landing |
| 4 | `finetune_vae_overfit_lr5e5_wsd.yaml` | 21685 | 5e-5 | wsd | **0.061** (ep46-50, stable) | **clean landing, trustworthy** |
| 5 | `finetune_vae_overfit_lr5e5_wsd_tail.yaml` | 21687 | 5e-5 (tail 5e-6) | wsd, stable_fraction 0.7 | 0.0588 (ep50, stable) | encoder tail unfrozen, only ~3% over #4 -- read as noise, see below |
| 6 | `finetune_vae_overfit_lr5e5_wsd_tail_slowdecay.yaml` | 21690 | 5e-5 (tail 5e-6) | wsd, stable_fraction 0.35 | 0.0608 (ep50) | doubling decay share vs #5 did **not** improve on 0.0588 -- schedule was not the limiting factor, see below |

**Attempt 1** (job 21667): blew up in epoch 2 (train_loss 0.288 -> 2.106),
flat at val_vrmse ~1.0 for 23 epochs -- collapsed, not converging. Cause:
`learning_rate: 2e-4` constant with only 50 warmup steps (4x the 5e-5 every
successful run uses, no decay) plus `adapter_lr_multiplier: 10.0` (effective
2e-3 on `encoder.conv_in`). Also used `inflate_init: zeros`, unlike the
baseline's `mean`. Empty `checkpoints/` is expected, not a bug: `interval: 5`
never lined up with the (epoch-1) best, and `save_last_state` was off --
nothing was written by design. Output dir deleted; metrics/curves/panels are
in git at `2782d67`.

**Attempts 2-4** (this file's earlier revisions had 2 written up as "not yet
launched" -- it has since run). The clean comparison is 2 vs 4: **same peak LR
(5e-5)**, only the schedule differs. Constant (2) never got below ~0.10 across
50 epochs; wsd (4) descended smoothly through the decay phase to a stable
0.061 (epochs 46-50 all in 0.061-0.066). Attempt 3 (lower constant LR alone,
no decay) only reached 0.083 -- confirms it's the *decay*, not just a lower
LR, buying the improvement, so attempt 4's 0.061 reads as genuine convergence
rather than a decay-masking artifact. Full headers/analysis are in
`configs/finetune_vae/finetune_vae_overfit_lr5e5_wsd.yaml` and
`finetune_vae_overfit_lr25e5.yaml`.

**Verdict: 0.061 falls in the 0.05-0.08 "partial" band**, not `>=0.09`. The
information does reach the decoder to a meaningful degree, but there's still
a gap to `<=0.03`. Per the readout table this unblocks Open Question 3
(encoder unfreezing), which was explicitly gated on this answer.

### Attempt 5 -- encoder tail unfreeze, DONE

Tests Open Question 3, scoped down from "unfreeze the whole 694M trunk at
0.1x LR" to just `down_blocks[-1] + mid_block + norm_out + conv_out` (the
tail): these already run on the fully-compressed 4x4 grid, so unfreezing them
cannot change the compression ratio, only what the 512->128 channel
projection keeps -- the axis "partial" points at. `down_blocks[:-1]` (the
actual spatial downsampling, carrying the fragile pretrained basis -- see
`inflate_init: random` below) stays frozen. Also far cheaper: tail activations
are 4x4 vs up to 32x32 for the downsampling blocks.

Code: `windinet/config.py` (`adapter.unfreeze_encoder_tail`,
`optimization.encoder_tail_lr_multiplier`, default 0.1 = 5e-6 absolute,
matching the Q3 proposal below) and `windinet/training/vae_trainer.py`
(`_get_encoder_tail_modules`, freeze/unfreeze in `_load_vae`, the
`_collect_trainable_params` assertion, `_set_trainable_modules_mode`, a third
optimizer param group). `_save_checkpoint` writes `encoder_tail.*` keys, but
`load_inflated_vae_checkpoint` does not restore them yet -- fine for this
diagnostic (`checkpoints.interval: null`, `resume_from: null`), not yet wired
for a real training run.

Single variable vs attempt 4: `adapter.unfreeze_encoder_tail: false -> true`.
Everything else (5e-5 peak, wsd, 50 epochs, same 8 sims) unchanged.

Read-out: val_vrmse materially below 0.061 (toward `<=0.03`) means the tail
was part of the bottleneck, worth carrying into a real run. Landing back at
~0.061 means the tail's channel selection wasn't the limiting factor and
Open Question 4 (latent bandwidth / input resolution) is the next lever, not
unfreezing further into the trunk. Kill criterion unchanged (epoch 2
train_loss above epoch 1's).

**RESULT: 0.0588** (job 21687, epoch 50, stable through epochs 46-50). Only
~3.6% below attempt 4's 0.061 -- read as within noise, not a clean signal
that the tail was the bottleneck. Attempt 6 below checks whether this was a
schedule artifact before trusting that reading.

Launch: `sbatch jobs/lundquist/finetune_vae_debug.sbatch configs/finetune_vae/finetune_vae_overfit_lr5e5_wsd_tail.yaml`

### Attempt 6 -- tail unfreeze + slower decay, DONE

Attempt 5's own metrics.csv showed the stable phase (epoch 2-35, constant LR
under `stable_fraction 0.7`) never plateaus -- val_vrmse wanders 0.13-0.21
with no clear trend at `batch_size 1`, unlike the full-data baseline's much
lower-noise stable phase. The decay phase (epoch 36-50, only 30% of the
post-warmup budget) did essentially all the real convergence (0.127 -> 0.0588)
and was still moving at the last step. That raised the possibility that
0.0588 was decay-length-limited rather than a true floor.

Single variable vs attempt 5: `optimization.stable_fraction: 0.7 -> 0.35`
(same total budget -- 2000 opt steps, 50 epochs, warmup 50 -- decay's share of
the post-warmup schedule roughly doubles).

**RESULT: 0.0608** (job 21690, epoch 50) -- *not* materially below attempt
5's 0.0588 (config: `finetune_vae_overfit_lr5e5_wsd_tail_slowdecay.yaml`).
Doubling the decay share did not improve on attempt 5, so per that config's
own read-out criterion: **the schedule was not the limiting factor**. 0.0588
stands as the real floor for this latent/tail combination, the "partial"
verdict on Open Question 2 holds without a schedule-length caveat, and Open
Question 4 (latent bandwidth) is the next lever -- not further schedule
tuning.

Launch: `sbatch jobs/lundquist/finetune_vae_debug.sbatch configs/finetune_vae/finetune_vae_overfit_lr5e5_wsd_tail_slowdecay.yaml`

### Encoder head-vs-tail unfreeze sweep (Open Question 3, generalized)

Attempt 5 above only let the encoder unfreeze grow from the tail. Rather than
guess whether the head (the actual spatial-downsampling `down_blocks`, which
carry the fragile pretrained patchify basis) or the tail matters more, both
directions are now config-selectable and swept.

Code: `windinet/config.py` `adapter.unfreeze_down_blocks` (list of
`down_blocks[0..2]` indices -- the spatially-downsampling stages -- any
subset/order) alongside the existing `adapter.unfreeze_encoder_tail`
(`down_blocks[3]` + `mid_block` + `norm_out` + `conv_out`, unchanged
semantics/name for backward compatibility with attempt 5's config).
`optimization.encoder_tail_lr_multiplier` now applies to the combined set of
whatever is unfrozen by either field (single shared param group, still 0.1x
decoder LR). `windinet/training/vae_trainer.py`: `_get_encoder_downblock_modules`
+ `_get_encoder_extra_modules` (down_blocks selection unioned with the tail
bundle when both are set), wired into `_load_vae`, `_collect_trainable_params`,
`_set_trainable_modules_mode`, the optimizer param group, and
`_save_checkpoint` (writes `encoder_down_block_{i}.*` keys per selected index,
same known gap as `encoder_tail.*` -- `load_inflated_vae_checkpoint` does not
restore either yet).

Each combo below has a diagnostic config (`finetune_vae_overfit_lr5e5_wsd_<tag>.yaml`,
8-sim capacity screen, same protocol as attempt 5) and a full-data config
(`finetune_vae_baseline_<tag>.yaml`, 15-epoch real run, only launch after the
diagnostic reads out per protocol point 1 below):

| tag | `unfreeze_down_blocks` | `unfreeze_encoder_tail` | modules unfrozen beyond conv_in | status |
|---|---|---|---|---|
| (baseline) | `[]` | `false` | none | done, val_vrmse 0.095396 |
| `tail` | `[]` | `true` | db3+mid+norm+conv_out | done -- diagnostic 0.0588 (attempt 5), confirmed not schedule-limited (attempt 6, 0.0608); full-data 0.09369 (`finetune_vae_baseline_tail.yaml`, job 21692) -- see "Full-data follow-ups" below |
| `tail2` | `[2]` | `true` | db2+db3+mid+norm+conv_out | done -- full-data val_vrmse **0.08742** (job 521394, sng_pvc, 15ep, 8h40m) |
| `tail3` | `[1,2]` | `true` | db1+db2+db3+mid+norm+conv_out | done -- full-data val_vrmse **0.08684** (job 521395, sng_pvc, 15ep, 8h54m) |
| `fullenc` | `[0,1,2]` | `true` | entire encoder trunk | done -- full-data val_vrmse **0.08673** (job 521389, sng_pvc, 15ep, 8h54m) -- **best of the sweep, promoted to `finetune_vae_whole_structure_baseline.yaml`** |
| `head0` | `[0]` | `false` | db0 only | done -- full-data val_vrmse 0.09507 (job 521390, sng_pvc, 15ep, 7h25m) -- essentially no gain over frozen-trunk baseline |
| `head01` | `[0,1]` | `false` | db0+db1 | done -- full-data val_vrmse 0.09086 (job 521391, sng_pvc, 15ep, 7h28m) |
| `head012` | `[0,1,2]` | `false` | db0+db1+db2 | done -- full-data val_vrmse **0.08686** (job 521392, sng_pvc, 15ep, 7h47m) -- ties `fullenc`/`tail3` at ~1h less wall-clock (no tail unfreeze) |
| `head01tail` | `[0,1]` | `true` | db0+db1+db3+mid+norm+conv_out (db2 skipped) | done -- full-data val_vrmse 0.09049 (job 521393, sng_pvc, 15ep, 8h40m) |

All seven full-data runs launched together 2026-08-05 01:06 CEST on sng_pvc
(1 node, 8 XPU tiles, `num_dataloader_workers: 2`, `finetune_vae.sbatch`),
committed in `f627761`. Read-out per combo: same as attempt 5 -- val_vrmse
materially below 0.061 (diagnostic) points at that module set being part of
the bottleneck; ~0.061 means it isn't.

**`head*` vs `tail*` at matched unfrozen-block-count** (the comparison this
sweep was designed for): `head0` (0.09507) vs `tail` (0.09369) -- tail
slightly ahead; `head01` (0.09086) vs `tail2` (0.08742) -- tail ahead;
`head012` (0.08686) vs `tail3` (0.08684) -- **statistical tie**. Reading
across the whole table: gains come almost entirely from unfreezing
`down_blocks[1]` and `[2]` (the deeper spatial-downsampling stages) --
`head0` alone barely moves the needle (0.09507 vs 0.095396 baseline, <0.2%),
while `head01` and `head012` do almost all the work. `unfreeze_encoder_tail`
on top of a given `down_blocks` set adds only a small, roughly-constant
increment (comparable to the `tail` vs baseline gap, ~1.8%) regardless of
how much of the head is already unfrozen -- consistent with the tail's
contribution being close to independent of the head's. Net: **whole-trunk
unfreeze (`fullenc`) is the best single arm, but `head012` (no tail) gets
statistically the same result for ~1h less wall-clock per run** (7h47m vs
8h54m) -- worth keeping in mind as a cheaper alternative if the upcoming LR
sweep needs to multiply run count.

**Caveat on the diagnostic (8-sim) arm of this sweep:** the diagnostic
configs (`finetune_vae_overfit_lr5e5_wsd_<tag>.yaml`) used
`overfit_repeat: 1` (8 opt-relevant samples/epoch) instead of attempts 4-6's
`overfit_repeat: 5` (40 samples/epoch) -- with `warmup_steps: 50` and ~2
optimizer steps/epoch at `overfit_repeat: 1`, roughly the first **25 of 50
epochs are still inside warmup**, well below peak LR. All seven diagnostic
runs land at val_vrmse 0.105-0.126 (vs attempts 4-6's clean 0.059-0.061
floor), which reads as a warmup/steps-per-epoch mismatch, not a real
capacity signal -- **do not use the diagnostic numbers from this sweep for
anything**; the full-data numbers above are the trustworthy read-out. Not
yet fixed; if the diagnostic tier is needed again (e.g. to cheaply screen
LR values before spending full-data time), restore `overfit_repeat: 5` or
cut `warmup_steps` first.

### Full-data follow-ups: tail unfreeze and schedule shape, DONE

Two single-variable full-data (3825-sim, 15-epoch) runs launched off the
`finetune_vae_baseline.yaml` reproduction (job 21666, 0.095396):

| run dir | job | change vs baseline | val_vrmse | verdict |
|---|---|---|---|---|
| `finetune_vae_baseline_tail` | 21692 | `adapter.unfreeze_encoder_tail: false -> true` | **0.09369** | ~1.8% better than baseline -- smaller than hoped, consistent with attempt 5/6's "within noise" diagnostic reading. Encoder tail unfreeze carries a small, real but not decisive gain. |
| `finetune_vae_baseline_slowdecay` | 21689 | `optimization.stable_fraction: 0.7 -> 0.35` | 0.09664 | ~1.3% **worse** than baseline (within the "under ~2%, treat cautiously" noise band the config's own header sets, per Open Question 1 still being unmeasured). Reads as null: `stable_fraction 0.7` was already about right for the full-data regime -- the schedule-length sensitivity seen in the 8-sim diagnostics does not transfer to full data. Do not carry `stable_fraction 0.35` forward. |

Net read: of the two variables tested at full scale so far, only the tail
unfreeze shows a (small) real improvement; the schedule-shape change does
not. `finetune_vae_baseline_tail`'s 0.09369 is the current best full-data
checkpoint and the reference point for the rest of the head-vs-tail sweep
above.

## Established

- **`inflate_init: random` destroys the run.** 0.162905 vs 0.094739 (+72%). The
  pretrained patchify basis must be preserved; `encoder.conv_in` alone (221K
  params) cannot re-learn a basis for a frozen 694M trunk in 1800 steps.
- **The loss-weight axis is exhausted.** Four independent knobs (h1 25->50,
  admul 10->1, wsd schedule, mlw 0->1e-4) all land in 0.0947-0.0967, a 2% band.
- **MLW at weight 1e-4 is net-negative** (#4 vs its own baseline #3, +1.3%).
  Reverted to 0.0. This is separate from the earlier SoftAdapt-driven MLW
  collapse recorded in the old `finetune_vae.yaml` header -- that failure was
  adaptive weighting un-suppressing MLW over time; this one is a fixed weight
  and simply did not help.

## NOT established (previously treated as if it were)

- **`mean` vs `zeros` init has never been cleanly isolated.** The only
  comparable pair is #5 (zeros, 0.101276) vs #6 (mean, 0.101768) -- zeros is
  *slightly better*, and #5 additionally differs by the tail3x schedule. The
  claim "mean beat zeros/random on every prior ablation", repeated in five
  config headers, is supported only for `random`.
- **`wsd` vs `cosine` is confounded with epoch count.** The 6% gain from the
  10-epoch group to the 15-epoch wsd group changed schedule *and* 10->15 epochs
  simultaneously. Neither is attributable.
- **Run-to-run variance is unmeasured.** All ten runs use seed 42, n=1. The
  effect sizes being chased in #1-#4 (0.15%, 0.7%, 1.3%) are almost certainly
  below the seed noise floor.

## The bandwidth ceiling

Measured band-limited VRMSE floors on this dataset (recorded in
`configs/finetune_vae/finetune_vae_overfit.yaml`):

| ideal reconstruction bandwidth | VRMSE floor |
|---|---|
| 32x32 cutoff | 0.165 |
| **64x64 cutoff** | **0.089** |
| current model (0.0947) | behaves like a ~56x56 Fourier truncation |

**Consequence: at most ~6% remains on the loss/schedule axis**, and only if a
perfect recovery of everything below the 64x64 cutoff were achieved. Anything
beyond that requires more latent bandwidth, not a better objective.

Qualitative confirmation from #1's epoch-15 reconstruction panels (now only in
git at `f88eb09`): per-sample RMSE
tracks the sample's high-wavenumber content, not its magnitude. A turbulent
sample (2507, gamma=1.40) gives 4.85e-2 with broadband speckle residual and a
visibly smoothed prediction; a smooth sample with sharp interfaces (1859,
gamma=1.365) gives 1.00e-2 with residual confined to thin lines on the
discontinuities. 5x spread, same model, same epoch.

## Open questions, in priority order

1. ~~What is the seed noise floor?~~ **ANSWERED (2026-08-08): ~1.1%.**
   `finetune_vae_baseline.yaml` at seed 42/1/2 landed val_vrmse 0.09513 /
   0.09519 / 0.09411 -- spread 0.00108, ~1.1% relative. Every "~1-2%, treat
   as noise" caveat elsewhere in this document should now use **~1.1%** as
   the actual threshold. See "Seed noise floor sweep" below for the full
   readout and its implications for other sweeps in this pull (channel-order,
   encoder-LR).
2. ~~Is the bottleneck the latent or the objective?~~ **ANSWERED: partial.**
   See "Capacity diagnostic (Open Question 2)" above -- val_vrmse 0.061 with
   proper convergence (wsd), landing in the 0.05-0.08 band. Neither a clean
   `<=0.03` (pure objective problem) nor `>=0.09` (info never reaches the
   decoder); both the loss/training axis and latent budget remain live.
   Attempt 6 (slower decay on top of the tail unfreeze, 0.0608 vs attempt 5's
   0.0588) confirmed this isn't a truncated-schedule artifact -- the "partial"
   verdict holds without caveat.
3. ~~Does unfreezing (part of) the encoder help, and does it matter which
   part?~~ **ANSWERED: yes, and the whole trunk wins.** Full "Encoder
   head-vs-tail unfreeze sweep" table above, all 7 remaining combos done
   2026-08-05 (sng_pvc, job 521389-521395). `fullenc` (entire trunk
   unfrozen) is best at val_vrmse 0.08673, ~9% better than the frozen-trunk
   baseline, tied with `head012` (0.08686) and `tail3` (0.08684). Gains come
   almost entirely from `down_blocks[1]`/`[2]`; `down_blocks[0]` alone is
   near-inert (0.09507, <0.2% over baseline); `unfreeze_encoder_tail` adds a
   small roughly-constant increment independent of how much of the head is
   already unfrozen. **Promoted `fullenc`'s settings to a new stable
   baseline**, `finetune_vae_whole_structure_baseline.yaml` -- see "New
   reference baseline" note at the top of this file and the LR-sweep plan
   below (Open Question 6).
4. ~~Does more latent bandwidth help?~~ **ANSWERED (2026-08-11): yes, by a
   wide margin -- the largest single-variable gain in this ledger.**
   `finetune_vae_whole_structure_baseline_256res.yaml` (8x8 latent grid, 4x
   capacity vs 128x128's 4x4) landed epoch-18 val_vrmse **0.05771** vs the
   128-resolution 18-epoch baseline's 0.08360 (job 523577) -- **-30.9%**, far
   past the ~1.1% seed-noise floor and past the 64x64-cutoff bandwidth
   ceiling (0.089, "The bandwidth ceiling" section) that section's own text
   already flagged as stale once the encoder unfroze. Gains are even across
   all 4 channels, not concentrated in one. See "256x256 resolution (Open
   Question 4)" below for the full result, the per-channel breakdown, and an
   open caveat on the normalization-stats file plus an unresolved promotion
   question.
5. **Was `stable_fraction 0.7`'s decay length actually right for full data?**
   **ANSWERED: yes.** `finetune_vae_baseline_slowdecay` (job 21689,
   `stable_fraction 0.35`) landed 0.09664, ~1.3% *worse* than the 0.095396
   baseline -- within the noise band Open Question 1 leaves uncalibrated, but
   directionally not an improvement. The schedule-length sensitivity seen on
   the 8-sim diagnostics does not transfer to full data; don't carry
   `stable_fraction 0.35` forward.
6. ~~Encoder learning-rate sweep on the whole-structure baseline.~~
   **ANSWERED (2026-08-08): 0.3x wins.** Full 4-arm grid done (0.03x/0.1x/
   0.3x/1.0x), val_vrmse 0.08847/0.08661/**0.08516**/0.08584. 0.3x adopted
   as the new default (`finetune_vae_whole_structure_baseline.yaml` updated
   in place) -- see "New reference baseline (as of 2026-08-08)" at the top
   and the full readout below. Note 0.3x and 1.0x are within the ~1.1% seed
   noise floor (Open Question 1) of each other. Original framing kept below
   for context: scoped down from "sweep the whole run's LR" to specifically
   the **encoder's** LR: with the entire encoder trunk now unfrozen (Open
   Question 3, answered), `optimization.encoder_tail_lr_multiplier: 0.1`
   (absolute encoder LR 5e-6, decoder LR 5e-5 unchanged) was inherited
   as-is from the original *tail-only* diagnostic (attempt 5, EXPERIMENTS.md
   above) and never re-tuned now that it governs a ~694M-parameter unfrozen
   trunk instead of a small tail submodule. The decoder LR itself
   (`learning_rate: 5e-5`) is **not** part of this sweep -- that is a
   separate, not-yet-opened question; only `encoder_tail_lr_multiplier`
   varies here, one variable per run per protocol point 1.

   **Grid** (log-spaced, ~3.3x steps, centered on the current default):

   | multiplier | absolute encoder LR | config | val_vrmse |
   |---|---|---|---|
   | 0.03x | 1.5e-6 | `finetune_vae_whole_structure_baseline_enclr0p03x.yaml` (job 521886) | 0.08847 |
   | 0.1x | 5e-6 | `finetune_vae_whole_structure_baseline.yaml`'s old setting (job 521885) -- re-confirms the `fullenc` arm of Open Question 3 (0.08673, job 521389) to within seed noise | 0.08661 |
   | **0.3x** | 1.5e-5 | `finetune_vae_whole_structure_baseline_enclr0p3x.yaml` (job 521887) -- **adopted as the new default, see top of file** | **0.08516** |
   | 1.0x | 5e-5 (= decoder LR, no safety margin) | `finetune_vae_whole_structure_baseline_enclr1x.yaml` (job 521888) | 0.08584 |

   Range rationale: `encoder_tail_lr_multiplier`'s default was deliberately
   kept below 1.0 because the unfrozen encoder weights "carry a pretrained
   basis instead of being freshly grown like `encoder.conv_in`"
   (`windinet/config.py` docstring) -- the grid brackets that design intent
   from "much more cautious" (0.03x) to "no caution at all" (1.0x, encoder
   treated the same as the decoder).

   **Read-out criterion:** compare val_vrmse across all 4 arms (0.03x /
   0.1x / 0.3x / 1.0x). Adopt whichever arm is lowest as the new
   `encoder_tail_lr_multiplier` default for the whole-structure baseline
   going forward. A gap under ~2% between arms should be treated as
   inside the still-uncalibrated seed-noise band (Open Question 1) rather
   than a real difference.

   **Kill criterion:** for the 1.0x arm specifically, watch epoch 1-2
   `train_loss` the way attempt 1 (`finetune_vae_overfit.yaml` diverging at
   `learning_rate: 2e-4`, EXPERIMENTS.md "Capacity diagnostic" above) and
   `inflate_init: random` (+72% val_vrmse, "Established" section) were
   caught -- if `train_loss` at epoch 2 is *higher* than epoch 1's, or
   `val_vrmse` is heading toward the 0.15-0.2+ range instead of down from
   the ~0.10-0.11 baseline-adapter-only starting point, that confirms the
   encoder's pretrained basis cannot tolerate decoder-level LR; let it
   finish for the record but don't chase multipliers above 1.0x.

   **DONE (launched 2026-08-06 10:34 CEST, jobs 521885-521888, all 15
   epochs complete as of the 2026-08-08 pull).** Also reconfirms
   `finetune_vae_whole_structure_baseline.yaml` under its own name (job
   521885) alongside the sweep, per protocol point 6 -- though see the
   "New reference baseline (as of 2026-08-08)" note above: that file's
   `encoder_tail_lr_multiplier` has since been bumped to 0.3x, so job
   521885's 0.1x result is no longer what the file reproduces.

   **Timing caveat (still applies):** all four were submitted at 10:34
   CEST, *before* the eval-parallelization fix (commit `f9c8541`, pulled
   ~11:02 CEST) and the `workers: 2 -> 4` default (commit `efa77d1`, ~11:00
   CEST) landed on the cluster -- confirmed by their eval= times staying in
   the 811-1043s/epoch range for all 15 epochs, matching pre-fix numbers,
   not the ~38s the same fix gives on jobs submitted after the pull (see
   "Eval parallelization fix" below). Each of the 4 jobs took ~9.4h
   wall-clock; a same-settings run on the current checkout would be closer
   to ~4h (see chorder/seed sweep below, same eval-fixed checkout, though
   note those arms don't unfreeze the encoder so aren't a clean apples-to
   -apples timing comparison either -- the encoder-LR arms also carry
   ~2.3x more trainable parameters than the frozen-trunk chorder/seed
   arms). Not a correctness problem, val_vrmse numbers below are trustworthy.

   **FINAL RESULT, epoch 15 (schedule fully decayed to `min_learning_rate
   1e-6`):**

   | arm | val_vrmse | vs 0.1x baseline | train_loss |
   |---|---|---|---|
   | 0.03x (job 521886) | 0.08847 | +2.1% (worse) | 0.03442 |
   | 0.1x (baseline, job 521885) | 0.08661 | -- | 0.03368 |
   | **0.3x (job 521887)** | **0.08516** | **-1.7% (best)** | 0.03323 |
   | 1.0x (job 521888) | 0.08584 | -0.9% | 0.03374 |

   **ANSWERED: 0.3x wins, adopted as the new default** (see "New reference
   baseline (as of 2026-08-08)" at the top of this file). Caveats:
   - 0.3x vs 1.0x (0.08516 vs 0.08584, ~0.8% apart) is **inside the ~1.1%
     seed-noise floor** measured by the same pull (Open Question 1, below)
     -- these two are not reliably distinguishable from each other. 0.3x is
     picked only because the sweep's own read-out criterion (lowest
     val_vrmse wins) says so, not because it's proven better than 1.0x.
   - 0.03x vs baseline (+2.1%) and 0.3x vs baseline (-1.7%) both clear the
     noise floor, so "more encoder LR than the inherited 0.1x default helps,
     up to a point" is a real effect, not noise.
   - 1.0x never diverged (matching the kill-criterion check throughout all
     15 epochs, train_loss monotonically decreasing) -- "tolerable but
     slightly behind 0.3x," not a clean failure, consistent with the
     epoch-5 early read.
   - Open Question 7 (was 15 epochs enough?) is still open and unresolved
     for this whole-structure-trunk lineage -- val_vrmse was still falling
     ~0.4-2% per epoch late in all 4 runs' schedules, so the ranking above
     could still shift with more epochs. `finetune_vae_whole_structure_baseline_ep18.yaml`
     exists for this but has not been launched.
7. ~~Was 15 epochs actually enough for the whole-structure baseline?~~
   **ANSWERED (2026-08-09): yes, real (if flattening) gain from 3 more
   epochs.** `finetune_vae_whole_structure_baseline_ep18.yaml` (job 523486)
   reached epoch-18 val_vrmse **0.08344**, ~2.0% better than the 15-epoch
   baseline's 0.08516 (job 521887) -- clears the ~1.1% seed-noise floor
   (Open Question 1). This run's own epoch-15 checkpoint read 0.09039,
   *worse* than the original 15-epoch run's, confirming the predicted
   mechanism: stretching the wsd schedule over 18 epochs leaves LR higher
   at epoch 15 (not yet decayed to the floor), so it needs epochs 16-18 to
   catch up and then pull ahead (0.09039 -> 0.08516 -> 0.08379 -> 0.08344).
   Per-epoch gains are shrinking (-5.8% ep15->16, -1.6% ep16->17, -0.4%
   ep17->18) -- flattening, consistent with genuine (if slowing) convergence rather
   than a schedule artifact, but not yet fully plateaued; a 20-21 epoch
   follow-up would be needed to see where it actually levels off. **Cost
   correction (2026-08-09):** an earlier pass of this write-up compared
   ep18's 6h33m wall-clock to "baseline's ~5h30m" without checking --
   job 521887's actual log shows **9h21m**, because that job (and the
   whole encoder-LR sweep) ran *before* the eval-parallelization fix (see
   "Eval parallelization fix" below); its eval phase cost ~780-980s/epoch,
   not the ~38s every job in this later pull gets. So in raw wall-clock,
   ep18 (post-fix, 18 epochs) actually finished **faster** than the
   pre-fix 15-epoch baseline job, which says nothing about the real
   per-epoch cost of "15 vs 18 epochs" -- that comparison needs two
   same-codebase runs. The honest same-codebase estimate: ep18's own
   steady-state rate is ~1246s/epoch (train+eval), so 3 extra epochs cost
   ~62 extra minutes relative to a 15-epoch run under the *current*
   codebase (~20% more, consistent with 18/15 epochs at a roughly
   constant per-epoch rate) -- not the 6h33m-vs-9h21m comparison a naive
   read of the two jobs' own totals would suggest.
   See "Epoch-budget result (Open Question 7)" below for
   the full per-epoch table. **Promoted 2026-08-09** --
   `finetune_vae_whole_structure_baseline.yaml`'s `optimization.epochs`
   is now `18`, per the file's own header changelog. Not yet re-run under
   its own name to confirm the ~0.0834 reproduction -- see "Channel-order
   sweep v2 + copy-init extension" below, which launches that confirmation
   run alongside a 12-arm follow-up built on top of it.

   Original framing kept for context: all three of `fullenc`/`head012`/`tail3`'s
   per-epoch val_vrmse flatten sharply as the wsd schedule decays to
   `min_learning_rate` (e.g. `fullenc`: -6.5% at epoch 13, -2.2% at epoch
   14, only **-0.4%** at epoch 15, where lr has decayed to 1.00e-6 -- same
   shape for head012 and tail3, see the sweep table above for the full
   curves). Ambiguous which of two things this is: the model genuinely
   converged, or the schedule (sized for 15 epochs) decayed LR to
   near-zero before the model was done improving. Also relevant: `fullenc`'s
   0.08673 is already *below* the 64x64-cutoff bandwidth floor (0.089,
   "The bandwidth ceiling" section) that was computed with the encoder
   frozen -- now that the encoder itself is unfrozen and adapting, that
   ceiling's assumptions may no longer hold, which is mild extra reason to
   suspect there's still real room on the training axis, not just via
   Open Question 4's latent-bandwidth route.

   Config: `finetune_vae_whole_structure_baseline_ep18.yaml`, single
   variable vs `finetune_vae_whole_structure_baseline.yaml`:
   `optimization.epochs: 15 -> 18`. `warmup_steps` (200) stays absolute;
   `stable_fraction` (0.7) is a fraction of the post-warmup budget, so both
   the stable phase and the decay phase stretch proportionally to the
   larger total -- this is exactly "give the schedule more room, let LR
   keep decaying, see if it still improves," no code change needed.

   **Single-variable check (2026-08-09), diffing the two runs' actual
   resolved `training_config.yaml`s:** confirms `optimization.epochs`
   (15 -> 18) is the only *intentional* difference, but the two jobs'
   configs also differ in `data.num_dataloader_workers` (2 for job
   521887, 4 for job 523486) -- this is the same pre-existing
   cluster-default cutover documented in the "Timing caveat" note under
   the encoder-LR sweep above (job 521887 was submitted before the
   `workers: 2 -> 4` default change landed). Not expected to affect
   converged `val_vrmse` (it only changes dataloader prefetch
   parallelism, not any computation or randomness in the training step
   itself), so the epoch-budget conclusion should still hold, but it
   means this comparison isn't a perfectly clean single-variable diff
   against job 521887 specifically -- a re-run of the 15-epoch baseline
   under the current codebase would close this gap if it ever matters.

   **Read-out:** compare this run's epoch-18 val_vrmse to
   `finetune_vae_whole_structure_baseline`'s epoch-15. **Note (2026-08-09):
   this criterion was written when `fullenc` (0.08673, job 521389) was
   still the reference point -- the encoder-LR sweep (Open Question 6)
   has since promoted 0.3x (0.08516, job 521887) as the actual current
   baseline. The result below is read against 0.08516; the direction of
   the conclusion is unchanged (ep18 still wins) but the margin is smaller
   than it would be against the stale 0.08673 number.** Also free from the
   same run without a second job: compare *this* run's own epoch-15 row
   (schedule not yet near the floor at that point, since decay is now
   stretched over 18 epochs) to the original 15-epoch run's epoch-15
   (already at the floor) -- if this run's epoch 15 reads *worse* than the
   original's, that's direct evidence the longer schedule needs the extra
   epochs to catch back up rather than leading throughout, which would
   support "genuinely still improving" over "already converged."

   **Kill criterion:** none beyond the usual (train_loss diverging) --
   only 3 extra epochs (~20% more wall-clock than the baseline), cheap
   enough not to need one.

   **Epoch-budget result (Open Question 7), job 523486, 8 tiles,
   `finetune_vae_whole_structure_baseline_ep18.yaml`, launched 2026-08-08
   10:43 CEST, wall-clock 6h33m:**

   | epoch | learning_rate | val_vrmse | vs prior epoch |
   |---|---|---|---|
   | 13 | 5.00e-5 (still at peak -- stable phase runs longer under the stretched schedule) | 0.10881 | -- |
   | 14 | 4.60e-5 (decay just starting) | 0.09523 | -12.5% |
   | 15 | 3.40e-5 | 0.09039 | -5.1% (vs baseline's *converged* 0.08516 at its own epoch 15: **+6.1% worse**) |
   | 16 | 1.85e-5 | 0.08516 | -5.8% (matches the 15-epoch baseline's final number almost exactly) |
   | 17 | 5.87e-6 | 0.08379 | -1.6% |
   | 18 | 1.00e-6 (floor) | **0.08344** | -0.4% |

   Final verdict: **0.08344 vs baseline 0.08516 = -2.02%, clears the
   ~1.1% noise floor** -- a real, if modest and flattening, improvement.
   15 epochs was not quite enough for the whole-structure baseline;
   `stable_fraction 0.7`'s decay was reaching `min_learning_rate` slightly
   before the model was done improving, matching the hypothesis above.
   Kill criterion never triggered (train_loss monotonically decreasing
   throughout).
8. ~~Does the order of the 4 physical channels matter?~~ **ANSWERED
   (2026-08-08): yes, and it tracks which field lands in index 3 (the fresh
   mean-init slot).** All 6 arms done, val_vrmse ranges 0.0954-0.1014 (~6%
   spread, well clear of the ~1.1% seed noise floor). Grouped by the field
   placed at index 3: pressure-last ~0.0954 (best, ties the existing
   `mx_my_pr` default), momentum_x-last ~0.0994, momentum_y-last ~0.1003
   (worst). Confirms the `inflate_vae_io_channels` mechanism hypothesis
   below -- see "Channel-order sweep" for the full table and per-slot
   breakdown. `mx_my_pr` (density, momentum_x, momentum_y, pressure)
   stays the default: it's already in the best (pressure-last) group and
   is what every other config in this ledger uses.

   Original framing kept for context: with `inflate_vae_io_channels` always
   growing index 3 fresh (mean-init) while indices 0-2 keep the pretrained
   RGB conv weights, which field lands in which slot is a free variable
   nobody had tested -- every run so far used density, momentum_x,
   momentum_y, pressure with no justification beyond "that's the order the
   dataset happened to expose them in." See "Channel-order sweep" below for
   the full 6-arm grid (built on `finetune_vae_baseline.yaml`, density
   fixed at index 0, the other three fields permuted) and required code
   change.
9. ~~Decoder LR and conv_in adapter LR multiplier -- untested on the
   whole-structure baseline.~~ **ANSWERED (2026-08-09): baseline's
   existing values (decoder LR 5e-5, adapter_lr_multiplier 1.0x) are
   already at/near the optimum on both axes -- lowering either hurts (real
   effect), raising either does nothing (noise).** Decoder LR 0.5x: +4.6%
   worse (clears floor). Decoder LR 2.0x: -0.54% (noise). Adapter LR 0.3x:
   +3.2% worse (clears floor). Adapter LR 3.0x: +0.12% (noise). See
   "Decoder LR / adapter LR multiplier sweep" below for the full grids.
   No config change adopted -- this closes the question rather than moving
   the baseline.

   Original framing kept for context: encoder unfreeze extent (Q3) and
   `encoder_tail_lr_multiplier` (Q6) are both answered; the two remaining
   LR knobs on this baseline's trainable weights -- decoder's own
   `optimization.learning_rate` (5e-5, unchanged since the very first
   ledger entry) and `adapter_lr_multiplier` (conv_in's multiplier of
   decoder LR, 1.0x, never independently questioned even though conv_in is
   a mix of pretrained-basis and freshly-grown channels, same ambiguity
   that motivated Q6) -- have never been swept. See "Decoder LR / adapter
   LR multiplier sweep" below for the two 3-arm grids and why a naive
   `learning_rate`-only sweep would confound decoder LR with the
   already-tuned encoder/conv_in LRs.
10. ~~Does copying a physically-paired sibling channel's pretrained
    weights beat `mean`-init for the freshly-grown 4th channel?~~
    **ANSWERED (2026-08-09): no, null result.** `copy` (momentum_y ->
    momentum_x's fresh slot) landed val_vrmse 0.09831 vs `mean`-init's
    0.09841 (`pr_my_mx`, job 522476) -- a 0.10% gap, far inside the ~1.1%
    seed-noise floor. Copying the sibling channel's weights is not
    distinguishable from mean-averaging all three original channels here.
    See "Copy-init for a physically-paired momentum channel" below for the
    full result and a caveat on the per-channel breakdown.

    Original framing kept for context: the channel-order sweep (Q8) picked
    the *arrangement* of which field sits where, but every arm still used
    `inflate_init: mean` for whichever field landed in the fresh slot --
    averaging in two physically unrelated scalar fields (density,
    pressure) alongside whichever field genuinely belongs there. When the
    fresh slot is one half of the momentum vector pair (momentum_x/
    momentum_y), there's a more targeted choice available: copy the
    *other* half's pretrained slot instead of averaging all three
    originals. See "Copy-init for a physically-paired momentum channel"
    below for the full RGB/physical-channel permutation table, the chosen
    arm, and the new `inflate_init: 'copy'` code path this needed.
11. ~~Does log-compressing density before z-scoring lower val_vrmse?~~
    **ANSWERED (2026-08-09): borderline, does not clear the noise floor.**
    `log_transform_channels: ["density"]` landed val_vrmse 0.08433 vs
    baseline 0.08516 -- a 0.97% gap, just under the ~1.1% seed-noise
    floor. Not confidently a real effect. The per-channel `val_vrmse_
    density` breakdown (0.10211) has no baseline counterpart to compare
    against (per-channel VRMSE postdates the baseline run), so whether
    density specifically improved is unverifiable with current data. See
    "Log-density experiment" below for the full result and what a
    follow-up would need.

    Original framing kept for context: density is strictly positive and
    spans a wide dynamic range (real dataset: min 0.024, max 26.1) --
    log(density) is a standard CFD/turbulence preprocessing trick to stop
    the long right tail (shock-compressed high-density regions) from
    dominating a linear-scale loss at the expense of near-vacuum
    resolution. Built on the current whole-structure baseline (0.3x),
    single variable `data.log_transform_channels: [] -> ["density"]`. See
    "Log-density experiment" below for the new `build_shockwave_video`/
    `denormalize_fields` code path this needed and the read-out/kill
    criteria.
12. ~~What is the 8-sim memorization ceiling under the whole-structure
    unfreeze?~~ **ANSWERED (2026-08-09): the ceiling does not move.**
    50-epoch overfit diagnostic with the entire encoder trunk unfrozen
    (job 523504) landed val_vrmse **0.05867**, squarely in the
    "0.05-0.06, no material change from attempts 4-6" bucket of the
    pre-registered read-out table (not the `<=0.03` bucket). Unfreezing
    the whole trunk does not raise the raw memorization capacity above
    what tail-only/conv_in-only unfreezing already achieved -- the
    whole-structure baseline's ~10% full-data gain (Open Questions 3/6)
    is buying something other than more memorization capacity. Open
    Question 4 (latent bandwidth) stays the primary lever for further
    gains. See "Capacity diagnostic re-run under whole-structure unfreeze"
    below for the full result.

    Original framing kept for context: the capacity diagnostic (Open
    Question 2) that established the 0.059-0.061 "partial" floor only
    ever had `encoder.conv_in` (attempt 4) or the tail alone (attempts
    5/6) trainable. The one attempt at the full trunk,
    `archive/known-bad/finetune_vae_overfit_lr5e5_wsd_fullenc.yaml`, is
    flagged known-bad and also predates the encoder-LR sweep (Open
    Question 6) that picked 0.3x over the 0.1x it used. See "Capacity
    diagnostic re-run under whole-structure unfreeze" below for the
    corrected config, a root-cause finding for *why* the known-bad attempt
    failed (which is not what `archive/known-bad/README.md` currently
    says), and the read-out/kill criteria.
13. ~~Does training on RMSE alone (h1 = ssim = 0) change val_vrmse
    relative to the current weighted objective?~~ **ANSWERED (2026-08-09):
    yes, RMSE-only is worse -- h1/ssim are net-positive regularizers, not
    just gradient dilution.** Zeroing h1 and ssim landed val_vrmse 0.08684
    vs baseline 0.08516, +1.97% worse, clearing the ~1.1% noise floor.
    `val_vrmse` is itself RMSE-family, so this is not a metric-alignment
    artifact -- the smoothness/structural terms are doing real work.
    Qualitative panel check (oversmoothing/artifacts at shocks) is still
    outstanding. See "RMSE-only loss ablation" below for the full result.

    Original framing kept for context: "the loss-weight axis is exhausted"
    (see "Established" above) was only ever tested by perturbing h1/mlw's
    *magnitude* -- h1 (weight 50, by far the largest coefficient) and ssim
    (0.15) have never both been zeroed. See "RMSE-only loss ablation"
    below for the single-variable config and the read-out/kill criteria.
14. ~~Does adding `h2` (curvature loss) reduce val_vrmse further on top of
    `h1`?~~ **ANSWERED (2026-08-12): no -- slightly worse, and not the
    targeted fix the motivating panel hoped for.** `h2` weight 25.0 landed
    epoch-18 val_vrmse 0.08449 vs baseline 0.08360, **+1.06% worse**, right
    at the edge of the ~1.1% noise floor. Per-channel breakdown shows the
    regression concentrated in `pressure` (+5.2%), the channel the
    motivating panel's speckle was worst on -- the opposite of the
    read-out criterion's "real, targeted" signature. See "H2 loss term"
    below for the full result.
15. ~~Does GradNorm match or beat the hand-tuned fixed weights, without the
    manual sweep cost?~~ **ANSWERED (2026-08-12): no -- unstable, and the
    predicted 5x cost was confirmed too.** Killed by the 24h time limit at
    epoch 14/18 (job 523590), but the real finding is that it was never
    converging: `val_VRMSE` oscillated between ~0.17-0.23 (even epochs) and
    ~0.56-1.13 (odd epochs) for all 14 completed epochs, in lockstep with
    `mlw`'s adapted weight swinging between ~0.0 and ~3.8-4.0 every single
    epoch -- exactly the "weight runaway" pattern the kill criterion said to
    watch for, just oscillating instead of monotonically diverging. Measured
    cost: ~6032s/epoch train time vs ~1220s for every fixed-weight arm,
    confirming the ~5x prediction. See "GradNorm loss weighting" below for
    the full per-epoch trace.
16. ~~h1 weight 50 -> 100, first retest under the whole-structure +
    18-epoch baseline?~~ **ANSWERED (2026-08-12): no effect.** val_vrmse
    0.08342 vs baseline 0.08360, -0.21%, inside the ~1.1% noise floor.
    Doubling h1 again does not help or hurt. See "Existing-loss weight
    retests" below.
17. ~~ssim weight 0.15 -> 0.3, first independent sweep of ssim's own
    weight?~~ **ANSWERED (2026-08-12): no effect.** val_vrmse 0.08436 vs
    baseline 0.08360, +0.91%, inside the ~1.1% noise floor. See
    "Existing-loss weight retests" below.
18. ~~mlw weight 0.0 -> 1e-4, RETEST under the whole-structure + 18-epoch
    baseline?~~ **ANSWERED (2026-08-12): yes, still net-negative --
    reconfirms the old frozen-trunk finding.** val_vrmse 0.08492 vs
    baseline 0.08360, **+1.58%, clears the ~1.1% noise floor.** mlw stays
    at weight 0.0. See "Existing-loss weight retests" below.
19. ~~KL divergence weight 0.0 -> 1e-7 (on) vs the reconfirmed baseline
    (off)?~~ **ANSWERED (2026-08-12): yes -- regularizes at effectively no
    reconstruction cost.** val_vrmse 0.08335 vs baseline 0.08360, -0.30%,
    inside the ~1.1% noise floor (i.e. indistinguishable from off) --
    exactly the outcome the read-out criterion called a pass ("latent space
    more regularized without costing much val_vrmse"). Root cause of the
    `val_kl`/`train_kl` blip (up to 6.4e8 in epoch 1-2 of some prior runs,
    vs. ~2e5 steady state) confirmed as diffusers' own `logvar` clamp
    (`[-30, 20]`) combined with the freshly-inflated conv_in's uncalibrated
    initial posterior, self-correcting within 1-2 epochs; not a bug. See
    "KL divergence weight" below for the full diagnosis, the weight-choice
    math, and the result.
    Config: `finetune_vae_whole_structure_baseline_kl.yaml`.
20. **PLANNED, configs written, not yet submitted: is training time still
    the bottleneck? Epoch budget 18 -> 30, plus slower decay and a lower
    LR floor, PhD-advisor-requested (2026-08-12).** Direct continuation
    of Open Question 7 (15->18 epochs, -2.02%, still improving not
    plateaued at epoch 18: -1.6% ep16->17, -0.4% ep17->18). Four configs,
    one variable each except (d):
    - (a) `finetune_vae_whole_structure_baseline_ep30.yaml`: epochs only,
      18 -> 30. Isolates "does more schedule room alone still help."
    - (b) `finetune_vae_whole_structure_baseline_ep30_slowdecay.yaml`:
      epochs 30 + `stable_fraction: 0.7 -> 0.5` (decay phase ~67% longer
      in absolute steps at the same total budget).
    - (c) `finetune_vae_whole_structure_baseline_ep30_lowfloor.yaml`:
      epochs 30 + `min_learning_rate: 1e-6 -> 1e-7` (floor 2% -> 0.2% of
      peak).
    - (d) `finetune_vae_whole_structure_baseline_ep30_slowdecay_lowfloor.yaml`:
      all three changes together -- the PhD advisor's literal combined
      ask. THREE variables move at once (protocol point 4 exception,
      explicitly noted in the config's own header) -- not a substitute
      for (a)/(b)/(c), answers "does the combined proposal beat the
      baseline," not which lever did it.
    **Caution on (b) specifically:** `stable_fraction 0.7 -> 0.35` (a
    steeper version of this same change) was already tried twice and did
    NOT help -- Open Question 5 (full data, old frozen-trunk baseline,
    job 21689, 0.09664 vs 0.095396 baseline, ~1.3% worse) and "Attempt 6"
    (8-sim overfit diagnostic, job 21690, 0.0608 vs attempt 5's 0.0588,
    not materially better). Both predate Open Question 3 (whole-trunk
    unfreeze) and Open Question 7 (the first schedule-budget-limited win
    in this ledger, post-unfreeze) -- reason for skepticism on (b), not a
    reason to skip it, since the regime has changed since both prior
    tests.
    **Read-out:** all four vs the 18-epoch baseline's confirmed 0.08360
    (job 523577), bar >2.2% (2x the ~1.1% seed noise floor, protocol
    point 3). (b)/(c)/(d) additionally read against (a)'s own result for
    attribution -- beating the baseline but not (a) means the gain is
    from epoch count alone, not the schedule-shape change.
    **Kill criterion:** usual (epoch 2 train_loss above epoch 1's) for
    all four; ~11h estimated wall-clock each (vs ~6.6h for the 18-epoch
    baseline), cheap enough not to need a stricter one.
21. **PLANNED, config written, not yet submitted: does the fresh channel's
    `mean` init leave a visible "superposition of the other 3 channels"
    artifact, and does `zeros` init remove it? PhD-advisor-motivated
    (2026-08-12).** Observed qualitatively on the whole-structure
    baseline's pressure channel (the fresh, index-3 slot under the
    default channel order): low val_vrmse but reconstructions visually
    resemble a blend of density/momentum_x/momentum_y rather than an
    independent prediction. Revisits the "NOT established" gap flagged
    above ("mean vs zeros init has never been cleanly isolated") with a
    clean single-variable run instead of the old confounded #5-vs-#6
    pair, and is a different question from Open Question 10's copy-init
    (which swapped *which* pretrained slot the fresh channel averages/
    copies from, not whether averaging happens at all -- copy-init came
    back null on val_vrmse but was never checked qualitatively either).
    (a) Config: `finetune_vae_whole_structure_baseline_zerosinit.yaml`,
    single variable vs the confirmed baseline (0.08360, job 523577):
    `adapter.inflate_init: "mean" -> "zeros"`.
    **Complementary no-training diagnostic:** `scripts/
    inflate_weight_drift.py` (new, CPU-only, no data loading) measures
    directly, on the *existing* mean-init checkpoint, whether the fresh
    channel's `decoder.conv_out`/`encoder.conv_in` weights have moved
    away from their `mean`-of-the-other-3 starting point (own-drift:
    cosine sim + relative L2 change vs init) and whether they still
    resemble the average of the other 3 channels' *current* weights
    (fresh-vs-mean-of-others, computed both at init and after
    finetuning). Low own-drift points at "insufficient training" (or too
    low an LR on that slot -- note `encoder_tail_lr_multiplier` does NOT
    scale `decoder.conv_out`, which trains at the full decoder LR); high
    own-drift with fresh-vs-mean-of-others still high after finetuning
    points at a loss/init-landscape effect that more epochs alone won't
    fix. Run this before/alongside the zeros-init training run -- it's
    minutes on a CPU, not a GPU job.
    **Read-out:** two-part (val_vrmse against baseline's 0.08360 /
    per-channel `val_vrmse_pressure` 0.09733, AND a qualitative
    reconstruction-panel comparison of the pressure channel against the
    mean-init baseline's) -- val_vrmse alone cannot answer the motivating
    question, see the config's own header.
    **Kill criterion:** usual, but expect a rougher epoch 1-2 than `mean`
    on the fresh channel specifically (no pretrained-derived starting
    point there) -- don't treat that alone as a kill signal, same caveat
    as every other init-change run in this ledger.
    (b) **Config: `finetune_vae_whole_structure_baseline_ep30_zerosinit.yaml`
    (new, 2026-08-12), epochs 18 -> 30 stacked on top of (a)'s zeros
    init.** Motivation: `zeros` starts the fresh channel colder than
    `mean` (no pretrained-derived prior at all, same qualitative risk
    `inflate_init: random` demonstrates at the extreme) -- 18 epochs may
    not be enough time for it to catch up, which would confound "does
    zeros fix the artifact" with "zeros just needs more steps." Gives
    zeros the same extended schedule Open Question 20 is already testing
    for `mean`, so a zeros-vs-mean read at matched epoch count is
    possible. Two variables move vs the original 18-epoch/mean baseline
    (epochs + init) -- read against (a)'s own epoch-18 zeros result and
    Open Question 20(a)'s epoch-30 mean result for attribution. Same
    two-part (quantitative + qualitative-panel) read-out and kill
    criterion as (a); ~11h estimated wall-clock, same as the other
    `_ep30*` runs.
22. **PLANNED, configs written, not yet submitted: KL divergence weight
    sweep -- where does regularization saturate below 1e-7, and where does
    reconstruction cost actually appear above it?** Direct follow-up to
    Open Question 19, which tested a single point (1e-7: -0.30% val_vrmse,
    ~359x more regularized `val_kl`, 196385 -> 547 by epoch 18) and closed
    with "promoted candidate, not yet adopted as baseline" -- one point is
    not a curve. Four single-variable configs vs
    `finetune_vae_whole_structure_baseline.yaml` (18ep, confirmed 0.08360,
    job 523577), same baseline Q19 swept against (not the still-in-flight
    epoch-30 baseline from Open Question 20):
    - `finetune_vae_whole_structure_baseline_kl1e8.yaml`: `kl` 1e-8 (10x
      below 1e-7). Does regularization basically vanish this far down?
    - `finetune_vae_whole_structure_baseline_kl3e7.yaml`: `kl` 3e-7 (3x
      above 1e-7). First point past the tested one -- does "no cost" hold
      a bit further before jumping an order of magnitude?
    - `finetune_vae_whole_structure_baseline_kl1e6.yaml`: `kl` 1e-6 (10x
      above 1e-7, steady-state contribution ~0.2). First arm expected to
      plausibly cost real val_vrmse -- kl now competes with reconstruction
      terms for gradient budget rather than being a minor add-on.
    - `finetune_vae_whole_structure_baseline_kl1e5.yaml`: `kl` 1e-5 (100x
      above 1e-7, steady-state contribution ~2.0, larger than the rest of
      the weighted objective combined). Deliberate failure-end bracket, not
      expected to work -- gives the sweep two real endpoints instead of one
      working point and unbounded extrapolation past it.
    All four verified loading through `VaeTrainerConfig`.
    **Gap carried over from Open Question 19, still open:** no per-epoch
    latent mean/std/logvar summary-statistic diagnostic exists (pure
    monitoring, no loss/training change) -- would let this sweep read out
    "how regularized" each arm actually is directly, instead of inferring
    it from `train_kl`/`val_kl`'s raw (unnormalized, sum-not-mean) scale.
    Not implemented as part of this sweep; deferred until after these four
    results come back.
    **Read-out:** each arm's val_vrmse vs baseline's 0.08360, bar >2.2%
    (2x the ~1.1% seed-noise floor, protocol point 3), plus epoch-18
    `val_kl` vs baseline's ~196-201k (unweighted, flat) and the 1e-7 arm's
    547, to place all points on one regularization-vs-cost curve.
    **Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) for
    `kl1e8`/`kl3e7`; relaxed to epoch 3 for `kl1e6` and epoch 4 for `kl1e5`
    per each config's own header -- higher weights amplify the same
    epoch-1-2 blip Q19 diagnosed (diffusers' logvar clamp on the
    freshly-inflated conv_in's uncalibrated posterior, bounded by
    `max_grad_norm: 5.0`, not a bug), so a rough start alone is not a kill
    signal at these weights. Genuine kill: `train_kl` still 1e8+ several
    epochs in, or `train_total_loss` diverging upward across multiple
    consecutive epochs with no recovery.
    Launch (one job per arm):
    ```
    sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_kl1e8.yaml
    sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_kl3e7.yaml
    sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_kl1e6.yaml
    sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_kl1e5.yaml
    ```

### Copy-init for a physically-paired momentum channel (Open Question 10, new 2026-08-08)

**THE QUESTION:** of the four physical fields, momentum_x and momentum_y
are the one genuinely symmetric pair (same physical role, different
spatial axis) -- density and pressure are each their own thing. In every
channel-order arm where one of momentum_x/momentum_y lands in the fresh
(index-3, mean-init) slot, `inflate_init: mean` seeds it as the average of
*all three* original slots, silently mixing in density and pressure even
though the physically closest analog (the other momentum component) is
sitting right there in one of those three original slots. Does copying
that sibling's pretrained weights directly, instead of diluting it with
two unrelated scalar fields, give the fresh momentum channel a better
starting point?

**Permutations already run (the channel-order sweep, Q8) and which field
sits in the fresh slot:**

| tag | channel_order (index 0-3) | index-3 (fresh) field | mx/my split across pretrained+fresh? | val_vrmse |
|---|---|---|---|---|
| `mx_my_pr` (default) | density, momentum_x, momentum_y, pressure | pressure | no (both mx, my pretrained) | 0.09540 |
| `my_mx_pr` | density, momentum_y, momentum_x, pressure | pressure | no (both mx, my pretrained) | **0.09536** (best overall) |
| `pr_my_mx` | density, pressure, momentum_y, momentum_x | momentum_x | **yes** -- my pretrained (idx 2), mx fresh (idx 3) | 0.09841 (best of the split arms) |
| `pr_mx_my` | density, pressure, momentum_x, momentum_y | momentum_y | yes -- mx pretrained (idx 2), my fresh (idx 3) | 0.09922 |
| `my_pr_mx` | density, momentum_y, pressure, momentum_x | momentum_x | yes -- my pretrained (idx 1), mx fresh (idx 3) | 0.10043 |
| `mx_pr_my` | density, momentum_x, pressure, momentum_y | momentum_y | yes -- mx pretrained (idx 1), my fresh (idx 3) | 0.10136 (worst overall) |

Four of the six arms already split momentum_x/momentum_y across a
pretrained slot and the fresh slot -- exactly the situation this question
is about. **`pr_my_mx` is the best-performing of those four** (0.09841),
so it's the arm this experiment builds on: momentum_y sits at index 2
(pretrained), momentum_x sits at index 3 (fresh, currently mean-init).

**SINGLE VARIABLE vs `finetune_vae_baseline_chorder_pr_my_mx.yaml`** (the
already-archived, already-measured arm above, val_vrmse 0.09841, job
522476) -- two fields move together since `inflate_init: 'copy'` requires
naming its source, same "multiple YAML fields, one conceptual variable"
pattern as every other sweep in this ledger:

```
adapter.inflate_init:         "mean" -> "copy"
adapter.inflate_copy_channel: (unset) -> "momentum_y"
```

Everything else (channel_order, channel_mean/std, seed 42, 15 epochs,
wsd, frozen encoder trunk) unchanged -- **deliberately still built on the
frozen-trunk `finetune_vae_baseline.yaml` lineage, not the whole-structure
baseline**, for the same reason Q8 itself made that choice: staying a
clean single-variable comparison against an already-measured number
(0.09841) rather than also confounding in the encoder-unfreeze-extent
difference.

Config: `finetune_vae_baseline_chorder_pr_my_mx_copyinit.yaml`.

**Code change required** (this was not a pure-YAML experiment, same as
Q8): `inflate_vae_io_channels` (`windinet/vae_adapter.py`) only supported
`'zeros'`/`'mean'`/`'random'`. Added `'copy'`, which seeds the fresh
channel's `encoder.conv_in`/`decoder.conv_out` slots (weight AND bias)
with an exact copy of one named original slot instead of averaging all of
them -- implemented as a 1-element-list degenerate case of the existing
mean-over-a-set logic (`_fill_new_blocks`'s `src` list has length 1 for
`'copy'`, length `n_orig` for `'mean'`), so no new averaging code path was
needed. `VaeAdapterConfig` gained `inflate_copy_channel: str | None`
(the source channel's name) with a validator requiring it be set (and be
one of the *original* 3 channels, not the new 4th one) iff `inflate_init
== 'copy'`. `VaeTrainer._load_vae` resolves the name to an index via
`adapter_cfg.channels.index(...)` before calling
`inflate_vae_io_channels`; the choice also round-trips through checkpoint
metadata (`inflate_copy_channel` alongside the existing `inflate_init`) so
`load_inflated_vae` (inference reload path) can reconstruct the same
tensor shapes, though by that point the actual values get overwritten by
the loaded checkpoint regardless of which init produced them originally.
Verified with hand-built `nn.Conv3d` layers (small stand-in channel/block
counts): `'copy'` reproduces the source slot's weight AND bias exactly in
the new slot, `'mean'`/`'zeros'` are bit-identical to their pre-refactor
behavior (regression check), and every existing checked-in config still
loads through `VaeTrainerConfig` unchanged. **Not yet run through an
actual training step.**

**Read-out criterion:** compare this run's epoch-15 val_vrmse to
`pr_my_mx`'s 0.09841 against the ~1.1% seed-noise floor (Open Question 1).
If `copy` clears the floor in the improving direction, that's evidence the
fresh channel's init should be picked per-field (copy a real sibling when
one exists) rather than uniformly mean-averaging everything -- worth then
checking whether the same swap on the other three split arms (`pr_mx_my`,
`my_pr_mx`, `mx_pr_my`) also improves before generalizing the claim.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's). Expected
low risk -- `copy` is a strictly gentler intervention than `random` (which
discards all 3 pretrained slots): only the fresh channel's init changes,
and it changes to another real pretrained-derived filter, not zeros or
noise. Not triggered (train_loss monotonically decreasing).

**RESULT, job 523495, 8 tiles, launched 2026-08-08 12:54 CEST, wall-clock
4h00m (faster than the whole-structure sweep's ~1200-1260s/epoch train
time -- this lineage is frozen-trunk, adapter-only, ~854s/epoch train):**

| arm | val_vrmse | vs `pr_my_mx` (0.09841, job 522476) |
|---|---|---|
| `mean` (job 522476) | 0.09841 | -- |
| `copy` (job 523495) | **0.09831** | -0.10% |

**ANSWERED: null result, inside the ~1.1% seed-noise floor.** Copying
momentum_y's pretrained weights into momentum_x's fresh slot is not
distinguishable from `mean`-averaging all three original channels --
`inflate_init: 'copy'` is not adopted. Per-channel breakdown for this run
(`val_vrmse_momentum_x` 0.13743, `val_vrmse_momentum_y` 0.13630,
`val_vrmse_density` 0.11795, `val_vrmse_pressure` 0.11471) cannot be
compared against `pr_my_mx`'s own per-channel numbers, because per-channel
VRMSE (see "Per-channel VRMSE, active by default" below) postdates job
522476 -- so there is no way, with current data, to check whether
momentum_x specifically moved even though the aggregate didn't. A
follow-up would need to re-run `pr_my_mx` under the current codebase to
get a comparable per-channel baseline before this question can be probed
further; not done here since the aggregate result already reads as null
and doesn't motivate chasing a per-channel signal.

### Log-density experiment (Open Question 11, new 2026-08-08)

**THE QUESTION:** density is strictly positive and spans a wide dynamic
range (`euler_mq_128_only_train.yaml`: min 0.024, max 26.1 -- over 3
orders of magnitude). Every run so far z-scores the raw value, which
means shock-compressed high-density regions (the long right tail) and
near-vacuum regions (close to the floor) compete on one linear scale --
the tail can dominate RMSE-family losses while the floor gets
proportionally little resolution. `log(density)` is the standard
CFD/turbulence fix: compresses the tail, expands resolution near the
floor, and turns multiplicative density variations into additive ones
(physically closer to how compression ratios are usually reasoned about
in gas dynamics).

**Mechanism:** `windinet.training.shockwave_data.build_shockwave_video`
gained `log_transform_channels: list[str] | None` -- for each named
channel, the raw physical field is replaced with `log(field.clamp(min=
LOG_TRANSFORM_EPS))` *before* the existing z-score step, not instead of
it. `VaeDataConfig.log_transform_channels` (validated to only accept
`density`/`pressure` -- momentum_x/momentum_y can be negative, `log()` is
undefined there) drives this, and -- since this config already uses
`normalization_stats_file` (see the dedicated ledger entry above) --
`validate_normalization_stats`'s file-lookup automatically switches to
the `log_<name>` entry (e.g. `log_density`, mean -0.288/std 0.604) instead
of `<name>`'s raw entry (mean 0.905/std 0.661) for any channel listed in
`log_transform_channels`, with no second number to hand-edit.
`windinet.training.vae_visualization.denormalize_fields` (used only by
the visualization panels, not by any loss/metric) inverts both steps in
reverse order -- undo the z-score, then `exp()` the named channels -- so
reconstruction panels still show physical density, not log-density.
Training losses and `val_vrmse` (including the per-channel breakdown
above) are computed in the same z-scored space every other channel
already uses -- with this flag on, that space is log-density-z-scored for
the density channel specifically, exactly mirroring how every other
sweep in this ledger already operates entirely in normalized space.

**Single variable vs `finetune_vae_whole_structure_baseline.yaml`** (0.3x
encoder LR, the current baseline):
```
data.log_transform_channels: [] -> ["density"]
```
Config: `finetune_vae_whole_structure_baseline_logdensity.yaml`.

**Known gap, same shape as the channel-order sweep's:** `scripts/
preprocess_dataset.py` and `scripts/inference_shockwave.py` call
`build_shockwave_video`/`denormalize_fields` without passing
`log_transform_channels` (or `channel_order`, the pre-existing gap) --
if this experiment's checkpoint is ever taken into stage-2 (DiT)
preprocessing or inference before those two scripts are updated, density
would be silently treated as untransformed. Training/eval within this
experiment are unaffected.

**Verified (not yet an actual training run):** `VaeDataConfig` rejects
`log_transform_channels: ["momentum_x"]`; with `normalization_stats_file`
set, `channel_mean[0]`/`channel_std[0]` resolve to `log_density`'s -0.288/
0.604 instead of density's raw 0.905/0.661, other three channels
untouched; a hand-built round-trip (`build_shockwave_video` with
`log_transform_channels=["density"]`, then `denormalize_fields` with the
same argument) recovers the original physical density -- and every other
channel -- to float32 precision (~1e-7 max abs diff) with zero elements
needing to saturate against `normalization_clip`; every existing
checked-in config still loads through `VaeTrainerConfig` unchanged
(default `log_transform_channels: []` is a no-op, verified byte-identical
`channel_mean`/`channel_std` to before this feature existed).

**Read-out criterion:** compare epoch-15 `val_vrmse` (and specifically
`val_vrmse_density`, the per-channel breakdown above -- this is the one
channel the change actually touches) to the baseline's 0.08516 (job
521887), against the ~1.1% seed-noise floor (Open Question 1).

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's). No
elevated risk expected -- this changes what density's z-scored values
*mean*, not the model architecture or any LR, and log-space statistics
computed from the same file are already well-formed (finite mean/std,
`euler_mq_128_only_train.yaml`'s `log_density` block has existed since
before this pull). Not triggered (train_loss monotonically decreasing).

**RESULT, job 523502, 8 tiles, launched 2026-08-08 15:34 CEST, wall-clock
5h40m (slowest arm in this pull, ~1246s/epoch train -- the extra
log/exp round-trip in `build_shockwave_video`/`denormalize_fields` is the
likely cause, not confirmed):**

| metric | value | vs baseline (0.08516) |
|---|---|---|
| `val_vrmse` (aggregate) | 0.08433 | -0.97% |
| `val_vrmse_density` | 0.10211 | no comparable baseline (see below) |

**ANSWERED: borderline, does not clear the noise floor.** -0.97% falls
just short of the ~1.1% seed-noise floor -- not confidently a real
improvement. The one number that would actually settle this,
`val_vrmse_density` compared against the baseline's *own* per-channel
density VRMSE, isn't available: job 521887 predates the per-channel VRMSE
feature ("Per-channel VRMSE, active by default" below), so there's no
baseline-density number to diff against. `log_transform_channels:
["density"]` is not adopted on this evidence, but the question is not
cleanly closed either -- a re-run of the baseline under the current
codebase (to get a comparable per-channel number) would be needed to say
more. Same `num_dataloader_workers` (4 vs baseline job 521887's 2) caveat
as Q7/Q9 above also applies here -- doesn't change the borderline
reading.

### Capacity diagnostic re-run under whole-structure unfreeze (Open Question 12, new 2026-08-08)

**THE QUESTION:** does unfreezing the entire encoder trunk (not just the
tail) raise the 8-sim memorization ceiling above attempts 4-6's
~0.059-0.061 floor (see "Capacity diagnostic (Open Question 2)" above)? If
yes, the whole-structure baseline's full-data gain (0.095 -> 0.085, Open
Questions 3/6) is at least partly a real capacity increase, and Open
Question 4 (more latent bandwidth) may buy less until this larger capacity
is itself saturated. If the floor doesn't move, the extra trainable
parameters are buying something other than raw memorization capacity, and
Open Question 4 stays the primary lever toward the <=0.03 band.

**Root-cause finding on the existing known-bad attempt (not previously
documented):** `archive/known-bad/finetune_vae_overfit_lr5e5_wsd_fullenc.yaml`
is the only prior attempt at this exact question, and it's flagged
known-bad -- but `archive/known-bad/README.md`'s stated cause
(`overfit_repeat: 1`, a warmup/steps-per-epoch mismatch) does not match
what's actually checked in. `git log --follow -p` on that file and its 6
siblings (head0/head01/head012/head01tail/tail2/tail3) shows
`overfit_repeat: 5` from their first commit, never 1. What *does* differ:
the resolved `training_config.yaml` under
`finetune_vae_outputs/sng_pvc/archive/known-bad/.../training_config.yaml`
shows `gradient_accumulation_steps: 4` against 8 processes, even though
the source config says `1` -- those 7 runs went through
`jobs/sng_pvc/finetune_vae.sbatch`, whose `patch_config_for_cluster()`
call retargets `gradient_accumulation_steps` to hit sng_pvc's
`effective_batch=32` default (`windinet/cluster_config.py`), silently
turning the diagnostic's "1 optimizer step per sim" design into effective
batch 32 against 40 samples/epoch -- roughly 1 optimizer step/epoch
instead of 40, so `warmup_steps: 50` never comes close to completing in 50
epochs. Same end symptom the README describes, different mechanism.
Whichever write-up is right, the actionable fix is identical: **do not
submit an overfit-diagnostic config through `jobs/sng_pvc/finetune_vae.sbatch`
or `finetune_vae_diag.sbatch`** -- both hard-code 8 tiles and derive
`gradient_accumulation_steps` to hit `effective_batch=32`.
`jobs/sng_pvc/finetune_vae_debug.sbatch` (added alongside this question) is
the sng_pvc counterpart of `jobs/lundquist/finetune_vae_debug.sbatch`: 1
tile, `effective_batch=1` passed explicitly to `patch_config_for_cluster()`,
same "one optimizer step per sim" contract. lundquist's own debug script
was tried first for launching Open Question 12's run and failed at
submission (`sbatch: error: Invalid generic resource (gres) specification`
on this account/login node) -- sng_pvc is where this project's runs
actually live, hence the new script rather than debugging lundquist's gres
setup.

**Single variable vs attempt 4** (`finetune_vae_overfit_lr5e5_wsd.yaml`):
```
adapter.unfreeze_down_blocks:        []  -> [0, 1, 2]
adapter.unfreeze_encoder_tail:    false  -> true
optimization.encoder_tail_lr_multiplier (new)  -> 0.3
```
0.3x matches the *current* whole-structure baseline (post Open Question
6), not the retired 0.1x the known-bad fullenc attempt used. Everything
else (peak LR 5e-5, wsd, `stable_fraction 0.7`, 8 sims x repeat 5, 50
epochs, seed 42) unchanged from attempt 4.

Config: `finetune_vae_overfit_lr5e5_wsd_wholestruct.yaml`. Verified: loads
through `VaeTrainerConfig` unchanged.

**Read-out** (val_vrmse == train reconstruction here, by design):
| result | reading |
|---|---|
| <= 0.03 | capacity ceiling moved a lot -- re-examine Open Question 4's urgency |
| ~0.05-0.06 | no material change from attempts 4-6 -- Open Question 4 (latent bandwidth) stays the primary lever |
| >= 0.08 | worse than the frozen/tail-only attempts -- suspect optimization instability at 1-GPU/accum-1 gradient noise, not a capacity answer; debug before drawing a conclusion |

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's).
**Literally triggered** (epoch 1 `train_loss` 0.18299 -> epoch 2 0.21708),
but read in context, not a real divergence -- see full trajectory below.

**RESULT, job 523504, 1 tile, `finetune_vae_debug.sbatch`
(`effective_batch=1`), launched 2026-08-08 16:05 CEST, wall-clock 1h06m
(50 epochs, ~62s/epoch train + ~3.2s eval -- far cheaper than any
whole-structure full-data run). Full per-epoch trajectory checked (not
just the tail), correcting an earlier pass that only looked at the last
3 rows:**

Epochs 1-35 (warmup + wsd stable phase, `learning_rate` at or near the
5e-5 peak) **oscillate hard**, `val_vrmse` swinging 0.13-0.63 with no
clean trend -- `train_loss` itself rises epoch 1->2 (0.183->0.217) and
spikes again at epoch 4 (0.350) and epoch 10 (0.371), which is exactly
the same shape the original capacity-diagnostic attempts 2/3 (constant
LR, "oscillated 0.10-0.30 all 50 epochs") showed, and exactly what
attempt 4 established the fix for: the wsd schedule's **decay phase**
(epochs 36-50 here, `learning_rate` ramping down from ~5e-5 to the 1e-6
floor) is what actually produces a clean landing, not the stable phase.
That pattern reproduces here almost exactly: `val_vrmse` only becomes
monotonic once decay starts (epoch 36: 0.147 -> epoch 50: 0.0587,
smooth throughout), stable in the last 5 epochs (0.0652/0.0627/0.0607/
0.0592/0.0587). Per-channel breakdown at epoch 50: pressure is the worst
channel throughout the whole run and stays worst at convergence
(val_vrmse_pressure 0.1189 vs density 0.0797/momentum_x 0.1081/
momentum_y 0.0762) -- consistent with pressure's per-channel VRMSE also
reading highest in the log-density and RMSE-only whole-structure runs
above, so this isn't specific to this diagnostic.

**Read on the kill criterion specifically:** the literal "epoch 2
train_loss above epoch 1's" trigger would have called this run dead at
epoch 2, but it recovered and produced the same clean, stable,
low-noise landing the wsd schedule reliably gives once decay starts --
same as how attempts 2/3's early oscillation was a false alarm relative
to attempt 4's eventual clean landing. Worth restating the kill
criterion's actual intent for early oscillation under wsd: watch whether
the **decay-phase** epochs trend down cleanly, not whether every early
epoch individually improves on the last.

val_vrmse (== train reconstruction) landed at **0.05867** (epoch 50,
stable through epochs 46-50: 0.05919/0.05875/0.05867).

**ANSWERED: falls in the "~0.05-0.06, no material change" bucket, not
`<=0.03`.** Unfreezing the entire encoder trunk does not raise the 8-sim
memorization ceiling above what attempts 4-6 (tail-only/conv_in-only)
already reached (0.0588-0.061). Read together with the read-out table's
own framing: the whole-structure baseline's ~10% gain over the
frozen-trunk baseline on full data (Open Questions 3/6) is not explained
by a capacity increase from unfreezing more of the trunk -- something
else about the wider trunk (better features, not more raw memorization
room) is doing the work. Open Question 4 (latent bandwidth / input
resolution) stays the primary lever toward the `<=0.03` band.

### RMSE-only loss ablation (Open Question 13, new 2026-08-08)

**THE QUESTION:** every run in this ledger trains against the fixed
weighted sum `rmse*1.0 + h1*50.0 + ssim*0.15 + mlw*0.0`. "The loss-weight
axis is exhausted" (see "Established" above) only ever perturbed the
*magnitude* of these weights (h1 25->50, mlw 0->1e-4) and landed in a 2%
band each time -- it never tried the boundary case of h1 = ssim = 0, i.e.
training on RMSE alone. h1's weight (50.0) is by far the largest
coefficient in the sum and has never actually been ablated to zero. Does
removing it (and ssim) help, hurt, or leave `val_vrmse` unchanged --
`val_vrmse` is itself an RMSE-family metric (`windinet/losses/vrms.py`),
so a plain-RMSE objective could plausibly score *better* by not spending
gradient on smoothness/structural-similarity terms instead of amplitude
accuracy, or *worse* if those terms are acting as real regularizers
against overfitting the training set's high-frequency noise. No expected
direction going in.

**Single variable vs `finetune_vae_whole_structure_baseline.yaml`** (0.3x
encoder LR, the current baseline):
```
loss_weighting.weights.h1:    50.0 -> 0.0
loss_weighting.weights.ssim:   0.15 -> 0.0
```
(`mlw` was already 0.0.) Treated as one conceptual variable ("which loss
terms are active"), not two separate sweeps -- zeroing only one of h1/ssim
would still leave a compound objective and not answer "is rmse alone
enough." Everything else (`encoder_tail_lr_multiplier` 0.3x, seed 42, 15
epochs, wsd, whole encoder trunk unfrozen, `normalization_stats_file`)
unchanged.

Config: `finetune_vae_whole_structure_baseline_rmseonly.yaml`. Verified:
loads through `VaeTrainerConfig` unchanged.

**Read-out:** compare epoch-15 `val_vrmse` to the baseline's 0.08516 (job
521887), against the ~1.1% seed-noise floor (Open Question 1). Also check
`val_rmse`/`train_rmse` directly (`metrics.csv` logs every component
unconditionally regardless of its weight) and the reconstruction panels
specifically for oversmoothing or spurious high-frequency artifacts at
shock discontinuities -- h1 is what was suppressing exactly that failure
mode (see "The bandwidth ceiling" above), so a `val_vrmse` win here needs
a qualitative check before being read as a strict improvement.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's). No
elevated divergence risk expected -- `max_grad_norm` clipping is unchanged
regardless of which loss terms contribute to the pre-clip gradient. Not
triggered.

**RESULT, job 523503, 8 tiles, launched 2026-08-08 15:57 CEST, wall-clock
5h31m (~1226s/epoch train, in line with the other whole-structure arms):**

| metric | epoch 15 | vs baseline (0.08516) |
|---|---|---|
| `val_vrmse` | 0.08684 | **+1.97% worse** |
| `val_rmse` | 0.01271 | (`train_rmse == train_total_loss` here by construction, h1/ssim contribute 0) |

**ANSWERED: RMSE-only is worse, clears the noise floor -- h1/ssim are
net-positive regularizers, not just gradient dilution.** +1.97% is above
the ~1.1% seed-noise floor, so this reads as a real degradation, not
noise. Since `val_vrmse` is itself in the RMSE family, the naive
"dropping the extra terms should let the optimizer focus purely on
amplitude accuracy" intuition does not hold here -- h1 (curvature/
gradient-matching) and ssim (structural similarity) are evidently
suppressing a failure mode (oversmoothing / high-frequency noise at shock
discontinuities, per "The bandwidth ceiling" above) that otherwise costs
more RMSE than the gradient budget spent on them. **Qualitative panel
check (the read-out criterion's second half) is still outstanding, and
correction (2026-08-09): it isn't currently possible from what's on this
disk, not just "not done yet" as an earlier pass of this write-up implied.**
`visualization.enabled: true` in the resolved config confirms panels were
generated on the cluster during training, but **none of the 9 new run
directories in this pull have a local `visualizations/` folder** -- only
`metrics/` and `training_config.yaml` were pulled/committed this round,
unlike the "Where things live" table's own description of what should be
tracked (`{metrics,visualizations,training_config.yaml}`). Closing this
out needs either pulling `finetune_vae_outputs/sng_pvc/
finetune_vae_whole_structure_baseline_rmseonly/visualizations/` from the
cluster and committing it, or accepting the quantitative `val_vrmse`
result alone. No config change adopted -- the current weighted objective
(`rmse*1.0 + h1*50.0 + ssim*0.15`) stays as-is. Same `num_dataloader_workers` (4 vs
baseline job 521887's 2) caveat as Q7/Q9/Q11 also applies here -- doesn't
change the verdict (+1.97% is a real gap regardless of dataloader
parallelism).

### H2 loss term (Open Question 14, resolved 2026-08-12)

**ANSWERED: no -- slightly worse, and the regression lands exactly where the
motivating panel's speckle was worst, not spread out generically.** `h2`
(`windinet/losses/h2_semi_norm.py`) has been computed and logged
(`train_h2`/`val_h2`) in every whole-structure-lineage run since "New loss
components (H2/PCC/VRMS/KL)" above, but always at implicit weight 0.0 --
this was the first run that actually trained against it.

**THE QUESTION:** a reconstruction panel (epoch 15, sample
2507_gamma1.4, frame 50, shared 2026-08-09) shows residuals as broadband
speckle across all 4 channels, not concentrated only at shock edges --
the same high-wavenumber information loss "The bandwidth ceiling" section
quantified, now visible directly rather than inferred from the aggregate
metric. `h1` (first-derivative/gradient matching) is confirmed
load-bearing (Open Question 13: removing it costs +1.97% val_vrmse). `h2`
is one order past `h1` -- it penalizes curvature (second-derivative)
mismatches specifically, which a pure gradient penalty can still miss
(e.g. a peak whose slopes on both sides match but whose curvature at the
apex doesn't -- an over-smoothed peak, exactly the qualitative failure
mode already on record). Does adding it reduce val_vrmse further beyond
what `h1` already buys, or is the gradient axis already saturated (same
"axis exhausted" pattern seen for h1/mlw magnitude perturbations)?

**Weight choice:** `h2`'s raw magnitude tracks `h1`'s raw magnitude at
roughly 2x throughout training -- checked across all 8 whole-structure-
lineage runs in the 2026-08-08 pull (e.g. `ep18`'s epoch-18 row:
`train_h1` 0.000302, `train_h2` 0.000631, ratio 2.08x; `rmseonly`'s
epoch-15 row: 0.000333 vs 0.000686, ratio 2.06x -- consistent across
otherwise-different arms). `h1`'s weight is 50.0; weighting `h2` at
**25.0** gives it roughly the same *weighted* contribution to the total
loss `h1` currently has -- a "put it on equal footing" starting point,
not a tuned value. If this arm helps, a weight sweep around 25.0 (mirror
of how `h1` itself went 25 -> 50 earlier in this ledger) is the natural
follow-up.

**Single variable vs `finetune_vae_whole_structure_baseline.yaml`** (18
epochs, 0.3x encoder LR): `loss_weighting.weights.h2`: unset (0.0) ->
25.0. Everything else unchanged.

Config: `finetune_vae_whole_structure_baseline_h2.yaml`. Verified: loads
through `VaeTrainerConfig` (resolved `loss_weighting.weights` ==
`{rmse: 1.0, h1: 50.0, ssim: 0.15, mlw: 0.0, h2: 25.0}`).

**RESULT, job 523591 (sng_pvc), 18 epochs, LR decayed to floor:**

| | val_vrmse | vs baseline (0.08360, job 523577) |
|---|---|---|
| baseline (h2 off) | 0.08360 | -- |
| h2 = 25.0 | 0.08449 | **+1.06% worse** |

Right at the edge of the ~1.1% seed-noise floor -- not confidently a real
effect on the aggregate number alone. The per-channel breakdown is more
informative than the read-out criterion expected, though: `val_vrmse_pressure`
0.13360 vs baseline's 0.12699 (**+5.2%**), clearly worse than the other three
channels (density -0.7%, momentum_x +1.0%, momentum_y +0.9%, all inside
noise). Pressure is exactly the channel the motivating panel's speckle was
worst on -- so `h2` did concentrate its effect where the hypothesis expected,
but in the wrong direction: it made that channel's reconstruction worse, not
better. Read against the read-out criterion's own framing, this is the
"generic extra smoothing rather than targeted curvature correction" outcome
flagged as the weaker result -- except even the smoothing didn't pay off net
positive. `h2` stays at weight 0.0; no promotion. Qualitative panel
re-check (does the speckle visually look different) not done as part of
this pull.

**Read-out criterion:** compare epoch-18 val_vrmse to the baseline's
~0.0834 (job 523486, pending reconfirmation under the promoted baseline's
own name per the "Channel-order sweep v2" batch above), against the
~1.1% seed-noise floor. Check the per-channel VRMSE breakdown too -- if
`h2` is doing real work, the biggest per-channel gain should track
whichever channel's residual looked most speckled/unstructured in the
motivating panel, not spread evenly across all 4 (that pattern would
suggest something more generic, like effectively acting as extra
smoothing regularization rather than targeted curvature correction).
Qualitative panel check same as Open Question 13: less broadband speckle
in the residual maps, not just a lower scalar -- a `val_vrmse` win from
`h2` acting as generic extra smoothing (rather than fixing curvature
specifically) would be a weaker result than the hypothesis intends, same
caveat Q13's h1/ssim removal read-out already carries in the other
direction.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's). Per
this pull's own finding (declr2x, adapterlr0.3x, the Q12 overfit
diagnostic, all above), an isolated epoch-to-epoch `train_loss` bump
alone is common and often benign -- confirm `val_vrmse` also regresses
before treating one as real. No elevated divergence risk expected: a
bounded second-derivative penalty at a weight comparable to `h1`'s, which
has run without incident in every prior arm.

**Separate from this run, not tracked here:** the same motivating panel
also shows a repeated bright edge artifact along the left boundary in
multiple channels (density, pressure) -- looks like a systematic
boundary/padding effect (this dataset has open boundaries on all sides,
per the HF dataset card `rha6696/euler_mq`), not random noise. A
loss-weight change is not expected to fix this if it's real; it would
need a separate look at the VAE's conv padding mode / how edge pixels
are handled in the encoder/decoder, independent of which loss terms are
active. Not investigated as part of this pull.

Launch (once ready):
```
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_h2.yaml
```

### GradNorm loss weighting (Open Question 15, resolved 2026-08-12)

**ANSWERED: no -- it never converges, oscillating instead, and the
predicted 5x cost was also confirmed.** Scoped down from a wider
loss-function discussion (2026-08-09): the fuller plan (KL divergence for
latent regularization ahead of the not-yet-run DiT stage, PCC, and
retuning h1/ssim/mlw's weights under the current 18-epoch whole-structure
baseline for the first time) is deferred -- this question is GradNorm
on/off only, for now.

**THE QUESTION:** every run in this ledger uses `loss_weighting.strategy:
"fixed"` -- every weight (rmse 1.0, h1 50.0, ssim 0.15, mlw 0.0) is
hand-picked and hand-swept. `windinet/loss_weighting/gradnorm.py`
(GradNorm, Chen et al. 2018) has existed in the codebase since the
H2/PCC/VRMS/KL pull but has never actually been used -- it adaptively
re-weights each active loss component once per epoch so each one's
*gradient norm* tracks a shared target based on how far behind/ahead its
own training progress is relative to the others. Does it match or beat
the hand-tuned fixed weights without the manual sweep cost?

**Caution already on record:** `strategy: "softadapt"` (the other
adaptive option in this codebase) caused MLW to collapse/blow up in an
earlier, now-archived config (see the old `finetune_vae.yaml` header,
referenced in "Established" above). GradNorm is a different algorithm --
it balances gradient *magnitudes*, not loss *values* like SoftAdapt --
so that specific failure mode isn't guaranteed to recur, but this run
should get at least the same level of scrutiny SoftAdapt's failure was
caught with, not be assumed safe by analogy to the fixed-weight arms'
clean track record.

**Cost warning, found while building this config, NOT YET MEASURED:**
`windinet/loss_weighting/utils.py`'s `compute_grad_norms` calls
`torch.autograd.grad` once per loss component per batch (4 here --
`REQUIRED_LOSS_NAMES` forces `loss_names` to include all of rmse/h1/ssim/
mlw whenever `strategy != "fixed"`, `mlw` cannot be dropped even though
the fixed-weight baseline currently runs it at weight 0.0), each a full
backward pass over every trainable parameter (~1.25B on this
whole-structure baseline: decoder + entire unfrozen encoder trunk +
conv_in), on top of the one combined backward every fixed-weight run
already does -- roughly **5x the backward-pass work per batch**.
`enable_gradient_checkpointing: true` (recomputes activations during
backward) makes each of those 4 extra backward calls costlier still, not
cheaper. This pull's fixed-weight arms ran ~1200-1250s/epoch train time
at 18 epochs; a naive multiplier suggests this arm could easily exceed
the sbatch script's hardcoded `--time=12:00:00` before 18 epochs finish.

**DECISION (2026-08-10): launch directly at full 8-tile scale**, skipping
the originally-recommended cheap 1-tile timing probe. To absorb the
unmeasured overhead without risking the run getting killed mid-epoch-18
by the sbatch script's default `--time=12:00:00`, launch with the
`partition=general` ceiling instead (24h, per the sbatch header
comment -- `--time` passed on the `sbatch` command line overrides the
script's own `#SBATCH --time` directive, standard SLURM precedence):
```
sbatch --time=24:00:00 jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_gradnorm.yaml
```
If it still doesn't finish 18 epochs within 24h, the per-epoch "Epoch N
timing" log lines will at least show the real overhead multiplier after
the fact. Fallback if that happens: the originally-recommended timing
probe --
```
sbatch jobs/sng_pvc/finetune_vae_debug.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_gradnorm.yaml
```
(1 tile, cheap, cancel after 2-3 "Epoch N timing" log lines in
`log_finetuning_vae/sng_pvc/<jobid>-wh_debug_vae.out`) -- won't reproduce
the whole-structure baseline's exact per-epoch train time, but the
*relative* GradNorm-vs-fixed overhead multiplier should transfer
reasonably well to 8 tiles, since the extra backward-pass cost is
per-batch-per-tile, not something parallelism amortizes away.

**Single variable vs `finetune_vae_whole_structure_baseline.yaml`** (18
epochs, 0.3x encoder LR, whole-structure unfreeze):
```
loss_weighting.strategy:    "fixed" -> "gradnorm"
loss_weighting.loss_names:  (unused under fixed) -> ["rmse", "h1", "ssim", "mlw"]
```
`alpha` (1.5) and `weight_lr` (0.025) left at class defaults -- untuned,
not part of this question (this asks "does GradNorm's default beat
hand-tuned fixed weights," not "what is GradNorm's best alpha"). `mlw`
being forced into `loss_names` despite running at weight 0.0 in the
fixed-weight baseline is itself informative -- GradNorm starts every
entry at weight 1.0 and adapts from there, so this run will show whether
it adapts `mlw`'s weight down toward the fixed-weight sweep's own
"net-negative" finding or disagrees with it.

Config: `finetune_vae_whole_structure_baseline_gradnorm.yaml`. Verified:
loads through `VaeTrainerConfig` (`strategy=gradnorm`,
`loss_names=['rmse','h1','ssim','mlw']`, `alpha=1.5`, `weight_lr=0.025`).

**RESULT, job 523590 (sng_pvc), launched 2026-08-09 08:19 CEST -- did not
finish, killed by the 24h time limit partway through epoch 15 (last
completed epoch: 14/18):**

Cost prediction confirmed first: measured train time **~6032-6034s/epoch**
(epochs 13-14) vs ~1216-1250s/epoch for every fixed-weight whole-structure
arm in this ledger -- **~4.9x**, matching the "~5x the backward-pass work"
estimate almost exactly. At that rate 18 epochs would need ~30h, past even
the extended 24h ceiling -- the kill was inevitable once launched, not a
close call.

**But the more important finding is that GradNorm was never converging in
the first place.** `val_VRMSE` for all 14 completed epochs:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val_VRMSE | 0.557 | 0.341 | 1.128 | 0.228 | 0.972 | 0.208 | 0.882 | 0.179 | 0.862 | 0.170 | 0.841 | 0.172 | 0.830 | 0.182 |

Odd epochs (~0.83-1.13) and even epochs (~0.17-0.34) never converge toward
each other -- this is a stable 2-epoch oscillation, not noisy convergence,
and every even-epoch value is already far worse than any fixed-weight arm's
epoch-14 number (~0.09-0.11). The adapted weights explain why: `mlw` swings
between **~0.0000 and ~3.7-4.0 every single epoch**, in lockstep and roughly
anti-phase with `ssim` (~0.01-0.2 vs ~1.8-3.7) -- e.g. epoch 13: `rmse=0.297,
h1=0.002, ssim=3.700, mlw=0.000`; epoch 14: `rmse=0.016, h1=0.0001,
ssim=0.189, mlw=3.795`. This is precisely the kill criterion's own named
failure mode ("if any single weight runs away... stop and inspect before
trusting val_vrmse") -- it just oscillates back and forth instead of
diverging monotonically in one direction, which the epoch-2-vs-epoch-1
`train_loss` check alone would not have caught (each single-epoch
`train_loss` looks locally reasonable; the loss *function itself* is what's
swinging).

**Verdict:** GradNorm does not match the hand-tuned fixed weights here --
even ignoring the time-limit kill, the run gives no indication it would
have stabilized given more epochs, since the oscillation shows no damping
across all 14 observed cycles. Combined with the ~5x cost, not worth
pursuing further with the current `alpha`/`weight_lr` defaults; a real
retry would need either a much smaller `weight_lr` (slower adaptation, may
damp the oscillation) or a different re-weighting cadence, and either way
would still carry the ~5x per-batch cost. No config promoted.

**Read-out criterion:** compare epoch-18 val_vrmse to the reconfirmed
baseline's ~0.0834, against the ~1.1% seed-noise floor. Independent of
whether val_vrmse improves, the 4 weights' trajectory over training is
itself a result worth recording (recoverable from the checkpointed
`loss_weighter` state) -- where GradNorm lands relative to the hand-tuned
(1.0, 50.0, 0.15, 0.0) values, and whether it converges to a stable point
at all or keeps drifting through epoch 18.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) PLUS
watch the per-loss weights themselves for a SoftAdapt-style runaway --
`min_weight`/`max_weight` are 1e-6/1e6, a very wide band, so a genuine
blowup is possible without hitting a hard clamp. If any single weight
runs away, or `mlw`'s climbs instead of adapting toward 0, stop and
inspect before trusting `val_vrmse` from that run.

Launch: see "DECISION" above -- direct full-scale submission with an
extended 24h time limit, not a timing probe first.

### Existing-loss weight retests: h1x2, ssimx2, mlw (Open Questions 16/17/18, resolved 2026-08-12)

**ANSWERED: h1x2 and ssimx2 land inside the noise floor (no effect); mlw
reconfirms its old net-negative finding.** Three single-variable
retests of the currently-active fixed weights (`rmse: 1.0, h1: 50.0,
ssim: 0.15, mlw: 0.0`), each vs `finetune_vae_whole_structure_baseline.
yaml` (18 epochs, 0.3x encoder LR, whole-structure unfreeze) with exactly
one weight changed:

| # | arm | change | why now, not before |
|---|---|---|---|
| 16 | `h1x2` | `h1: 50.0 -> 100.0` | h1's only prior sweep (25->50) predates the whole-structure unfreeze (Q3) and the 18-epoch promotion (Q7); untested direction (higher, not just re-confirming) |
| 17 | `ssimx2` | `ssim: 0.15 -> 0.3` | ssim's weight has **never** been independently swept, on any baseline -- Q13 only zeroed it as part of a compound h1+ssim ablation |
| 18 | `mlw` | `mlw: 0.0 -> 1e-4` | "net-negative at 1e-4" is an old-frozen-trunk-baseline finding ("Established" above), never re-checked post Q3/Q7 -- same weight as the original finding, so this is a clean retest |

Configs: `finetune_vae_whole_structure_baseline_h1x2.yaml`,
`finetune_vae_whole_structure_baseline_ssimx2.yaml`,
`finetune_vae_whole_structure_baseline_mlw.yaml`. All three verified
loading through `VaeTrainerConfig` with exactly the intended single
changed weight.

**RESULT, jobs 523592/523593/523594 (sng_pvc), 18 epochs each, LR decayed
to floor:**

| # | arm | val_vrmse | vs baseline (0.08360, job 523577) | verdict |
|---|---|---|---|---|
| 16 | h1x2 (`h1: 100.0`) | 0.08342 | -0.21% | inside noise floor, no effect |
| 17 | ssimx2 (`ssim: 0.3`) | 0.08436 | +0.91% | inside noise floor, no effect |
| 18 | mlw (`mlw: 1e-4`) | 0.08492 | **+1.58%** | **clears noise floor, real regression** |

h1x2 and ssimx2: neither doubling helps or hurts on the aggregate metric --
both `h1` (50.0) and `ssim` (0.15) stay at their current weights, already
at/near whatever local optimum this axis has (consistent with "the
loss-weight axis is exhausted," see "Established" above). mlw: the old
frozen-trunk-baseline finding ("mlw at 1e-4 is net-negative") reconfirms
cleanly on the whole-structure + 18-epoch lineage, a real effect this time
(+1.58% vs the frozen-trunk retest's own smaller, noise-adjacent gap) --
mlw stays at weight 0.0, no promotion for any of the three arms.

**Read-out:** each vs the reconfirmed baseline's ~0.0834, against the
~1.1% seed-noise floor. Independent single-variable comparisons -- don't
combine into a joint ranking (same caveat as every other multi-arm batch
in this ledger).

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) for all
three. Low risk on all three -- these are all magnitude perturbations of
already-active, already-well-behaved terms, not new components or
architecture changes.

Launch:
```
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_h1x2.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_ssimx2.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_mlw.yaml
```

### KL divergence weight (Open Question 19, resolved 2026-08-12)

**ANSWERED: yes -- the latent space is measurably (~359x) more regularized,
at no reconstruction cost distinguishable from noise.** PhD-advisor-requested,
motivated by the not-yet-run DiT stage ("DiT (stage 2, not yet run)"
above) training on this VAE's precomputed latents -- a latent space
finetuned with zero KL regularization is free to drift into whatever
shape best serves reconstruction alone, not necessarily one a downstream
diffusion model can sample from cleanly. **A win here is "the latent
space is more regularized without costing much val_vrmse," not
necessarily "val_vrmse improves."**

**Blip diagnosis, done before choosing a weight (2026-08-10):**
`val_kl`/`train_kl` were found (in an earlier audit of the 2026-08-08
pull) to spike to 1e8-1e9 scale in epoch 1-2 of some runs -- up to 6.4e8
(`copyinit`, job 523495) -- against a steady-state value of ~1.9e5-2.2e5
common to every run regardless of arm. Inconsistent across otherwise-
similar runs (4 of 8 whole-structure-lineage arms showed no blip at all),
which made "is this safe to weight?" genuinely unclear.

**Root cause, confirmed by reading diffusers' own source** (`site-
packages/diffusers/models/autoencoders/vae.py`,
`DiagonalGaussianDistribution.__init__`), not this repo's code:
```python
self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
```
`exp(20) ~= 4.85e8`. `windinet/losses/kl_divergence.py` **sums** (does
not average) `0.5*(mean^2 + exp(logvar) - logvar - 1)` over every
non-batch dimension of the latent tensor -- a SINGLE element pinned at
the clamp ceiling contributes `0.5*(exp(20)-20-1) ~= 2.43e8` to that sum
on its own, matching the observed blip scale almost exactly (6.4e8 ~=
2-3 saturated elements). The freshly-inflated `conv_in`'s 4th channel
starts with no calibration for what a sane posterior looks like;
ordinary reconstruction-loss gradient descent pulls it back within 1-2
epochs (same "epoch 1-2 rough, then clean" shape already on record for
Q12's overfit diagnostic and others). **This is expected behavior given
a known, deliberate diffusers safety clamp -- not a bug, and not
something needing a code fix**, just a weight choice that respects the
bound it implies.

An attempt to reproduce this end-to-end (load the real pretrained VAE,
inflate it, encode a random tensor, inspect `logvar`'s clamp-ceiling
element count directly) was started but timed out on this environment's
CPU-only inference before finishing -- the diffusers-source-level
mechanism above is confirmed by reading the actual clamp constant, which
is sufficient to act on; the empirical repro would only add a specific
observed count of saturated elements, not change the diagnosis.

**Weight choice: 1e-7.** Two considerations:
- Steady state (`2e5 * 1e-7 = 0.02`): comparable to `rmse`'s own
  ~0.011-0.013 contribution at convergence -- meaningfully regularizing,
  not negligible.
- Worst-case blip (`6.4e8 * 1e-7 ~= 64`): large relative to a typical
  epoch-1 `total_loss` (~0.03-0.11 across this ledger), enough to
  dominate that one epoch's gradient *direction* -- but `optimization.
  max_grad_norm: 5.0` clips the actual gradient *step* regardless of the
  raw loss magnitude, so this cannot cause an unbounded optimizer
  blowup. The realistic downside is a wasted epoch or two, not
  divergence, recoverable over the remaining 16-17 epochs -- this is why
  1e-7 was chosen over the more conservative 1e-9/1e-8 an earlier,
  pre-diagnosis pass of this design was considering.

**Single variable vs `finetune_vae_whole_structure_baseline.yaml`:**
`loss_weighting.weights.kl`: unset (0.0) -> 1e-7. Everything else
unchanged. Config: `finetune_vae_whole_structure_baseline_kl.yaml`.
Verified loading through `VaeTrainerConfig`.

**RESULT, job 523595 (sng_pvc), 18 epochs, LR decayed to floor:**

| | val_vrmse | vs baseline (0.08360, job 523577) |
|---|---|---|
| kl off (baseline) | 0.08360 | -- |
| kl = 1e-7 | 0.08335 | -0.30% (inside noise floor) |

**val_vrmse is indistinguishable from off** -- exactly the framing's own
definition of a pass. The regularization half of the question, flagged as
an unclosed gap when this section was written ("no existing diagnostic
reports per-epoch latent... so this can't be read out yet"), turns out to
be answerable from `train_kl`/`val_kl` themselves, already logged every
epoch regardless of weight:

| | epoch 1 val_kl | epoch 18 val_kl |
|---|---|---|
| baseline (kl off, unweighted passive monitor) | 200145 | 196385 |
| **kl = 1e-7** | 2236 | **547** |

With the weight on, `val_kl` drops from the outset and keeps falling --
**epoch-18 value is ~359x lower than the unweighted baseline's**, which
stays flat around ~196-201k for the entire run (nothing is pulling it down
when its weight is 0). This closes the previously-flagged gap directly: the
latent space is genuinely, substantially more regularized, not just
theoretically so. **Promoted candidate for the DiT-stage motivation** --
`kl: 1e-7` is a clean win against its own stated goal, though not yet
adopted as the new baseline default (that decision affects every future
sweep's comparison point, left for the same kind of explicit call as the
256-resolution promotion question above, not made unilaterally here).

**Read-out:** compare epoch-18 val_vrmse to the reconfirmed baseline's
~0.0834, against the ~1.1% seed-noise floor -- but per the framing above,
val_vrmse holding steady while the latent space regularizes is itself a
pass. **Gap: no existing diagnostic reports per-epoch latent mean/std/
logvar summary statistics**, so the "did the latent space actually get
more regularized" half of the question can't be read out yet -- would
need a pure-monitoring addition (no loss/training change) to close this
properly. Not implemented as part of this config.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) -- BUT
expect epoch 1 to look rough per the diagnosis above; that alone is not a
kill signal. Watch `train_kl` directly instead: 1e5-1e9 at epoch 1
dropping back to 1e5-1e6 by epoch 2-3 is the expected/diagnosed pattern.
Genuine kill signal: `train_kl` still at 1e8+ by epoch 3-4 (blip never
resolving), or `train_total_loss` diverging upward across multiple
consecutive epochs (not just one rough epoch 1).

Launch:
```
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_kl.yaml
```

### 256x256 resolution (Open Question 4, resolved 2026-08-11)

**ANSWERED: yes, and by a wide margin -- the largest single-variable gain in
this ledger.**

**THE QUESTION:** every whole-structure-lineage run (Open Questions 3/6/7)
operates at a fixed 128x128 input resolution, giving a 4x4 latent grid (each
cell covers 32x32px, ~250:1 compression). Feeding the model 256x256 input
instead grows the latent grid to 8x8 -- 4x more latent cells, independent of
which weights are trainable or how they're optimized. Does the extra latent
bandwidth translate into a real val_vrmse win, or is the loss/schedule axis
(assumed ~exhausted, see "The bandwidth ceiling") the actual bottleneck
regardless of latent capacity?

**Single variable vs `finetune_vae_whole_structure_baseline_ep18.yaml`** (18
epochs, 0.3x encoder LR, whole-structure unfreeze -- the most-validated
baseline at the time this config was written): only `data.data_root` and
`data.normalization_stats_file` change, to point at `256x256_ds/train.h5`
and its own channel-stats file (`euler_mq_256_only_train.yaml`) instead of
the 128-resolution dataset/stats. Everything else (LR, schedule, loss
weights, epochs) held fixed. Config:
`finetune_vae_whole_structure_baseline_256res.yaml`.

**RESULT, job 523757 (sng_pvc), launched 2026-08-10 14:08 CEST, wall-clock
~13h55m, full 18 epochs, LR decayed to floor (1e-6):**

| resolution | latent grid | val_vrmse | vs 128-res baseline (0.08360, job 523577) |
|---|---|---|---|
| 128x128 | 4x4 | 0.08360 | -- |
| **256x256** | **8x8** | **0.05771** | **-30.9%** |

Per-channel (`val_vrmse_<channel>`, epoch 18): density 0.0718, momentum_x
0.0727, momentum_y 0.0722, pressure 0.0973 -- pressure (the fresh, mean-init
channel) stays the worst of the four, same ranking as every 128-resolution
run in this ledger, just uniformly lower across all four. The gain isn't
concentrated in (or absent from) any one physical field.

**Far past both the seed-noise floor and the old bandwidth ceiling.** "The
bandwidth ceiling" section's 0.089 floor for a 64x64-cutoff reconstruction
was measured with the encoder frozen at 128 resolution -- that section's own
text already flagged its assumptions as stale once the encoder unfroze (see
the note under Open Question 7). This result confirms it decisively:
0.05771 sits well below that number, so resolution -- not the loss/schedule
axis -- was still the dominant lever available, consistent with "The
bandwidth ceiling"'s own "anything beyond [64x64] requires more latent
bandwidth, not a better objective."

**Cost:** measured per-epoch train=2614-2628s + eval=105s (~45.4 min/epoch)
vs the 128-resolution baseline's ~1220s + 38s (~21 min/epoch) -- roughly
2.2x the per-epoch wall-clock, not a naive 4x.

**Open caveat, carried from the config's own header:** the 256-resolution
`channel_mean`/`channel_std` (`euler_mq_256_only_train.yaml`) have **not**
been independently cross-checked against a full `compute_channel_stats.py`
scan of `256x256_ds/train.h5`, unlike the 128-resolution numbers (see
"Canonical normalization-stats file" above, confirmed to 1e-16 relative
error). Nothing in this run's metrics looks off (smooth monotonic
convergence, no NaN/blowup, sane per-channel spread), so this is not a
reason to distrust the 0.05771 number -- just an unclosed loose end before
leaning on it for a follow-up sweep at this resolution.

**Not yet decided: promote 256x256 to the new default baseline resolution?**
A -30.9% single-variable win this far past the noise floor is a strong
candidate, but every open question answered so far (channel order,
copy-init, decoder/adapter LR, loss weights, KL) was tuned *at 128
resolution* against the 128-resolution baseline -- none of those tunings
have been re-validated at 256. Promoting now would mean either re-running
the already-closed sweeps at 256 (expensive) or carrying 128-tuned
hyperparameters forward on faith. Flagging for a PhD-advisor decision rather
than promoting unilaterally.

Launch (already run, kept for reference):
```
sbatch jobs/sng_pvc/finetune_vae.sbatch \
    configs/finetune_vae/finetune_vae_whole_structure_baseline_256res.yaml \
    "${SCRATCH}/windinet/euler_mq_dataset/256x256_ds/train.h5"
```

### Decoder LR / adapter LR multiplier sweep (Open Question 9, new 2026-08-08)

**THE QUESTION:** two of the whole-structure baseline's LR-bearing
components have never been independently tuned -- the decoder's own
`optimization.learning_rate` (anchors every other LR in the config, since
`adapter_lr_multiplier` and `encoder_tail_lr_multiplier` are *multipliers*
of it, not absolute values) and `adapter_lr_multiplier` (conv_in's
multiplier, always 1.0x so far, meaning the freshly-inflated 4th input
channel trains at exactly the decoder's LR with no justification beyond
"nobody changed it").

**Why not just sweep `learning_rate` directly:** because
`adapter_lr_multiplier`/`encoder_tail_lr_multiplier` are relative to it,
scaling `learning_rate` alone silently rescales the absolute encoder and
conv_in LRs too (both already tuned -- encoder to 0.3x via Q6) -- three
variables move at once instead of one. Each arm below instead moves
`learning_rate` together with compensating changes to the *other two*
fields so their **absolute** LRs stay pinned at the baseline's values
(encoder 1.5e-5, conv_in 5e-5) -- same "multiple YAML fields, one
conceptual variable" pattern the channel-order sweep used for
`channel_order`+`adapter.channels`+`channel_mean`/`channel_std`.

**Grid A -- decoder LR** (baseline anchor 5e-5, encoder/conv_in absolute
LR pinned):

| tag | `learning_rate` | `adapter_lr_multiplier` | `encoder_tail_lr_multiplier` | config | val_vrmse | vs 1.0x |
|---|---|---|---|---|---|---|
| 0.5x | 2.5e-5 | 2.0 (pins conv_in at 5e-5) | 0.6 (pins encoder at 1.5e-5) | `finetune_vae_whole_structure_baseline_declr0p5x.yaml` (job 523487) | 0.08910 | **+4.6% worse** |
| 1.0x | 5e-5 | 1.0 | 0.3 | `finetune_vae_whole_structure_baseline.yaml` (reuse job 521887) | **0.08516** | -- |
| 2.0x | 1e-4 | 0.5 (pins conv_in at 5e-5) | 0.15 (pins encoder at 1.5e-5) | `finetune_vae_whole_structure_baseline_declr2x.yaml` (job 523490) | 0.08470 | -0.54% (noise) |

**Grid B -- conv_in adapter LR multiplier** (decoder LR and
`encoder_tail_lr_multiplier` both pinned at baseline, no compensation
needed since this multiplier doesn't rescale anything else):

| tag | `adapter_lr_multiplier` | config | val_vrmse | vs 1.0x |
|---|---|---|---|---|
| 0.3x | 0.3 (conv_in abs LR 1.5e-5) | `finetune_vae_whole_structure_baseline_adapterlr0p3x.yaml` (job 523488) | 0.08792 | **+3.2% worse** |
| 1.0x | 1.0 (conv_in abs LR 5e-5) | `finetune_vae_whole_structure_baseline.yaml` (reuse job 521887) | **0.08516** | -- |
| 3.0x | 3.0 (conv_in abs LR 1.5e-4) | `finetune_vae_whole_structure_baseline_adapterlr3x.yaml` (job 523489) | 0.08526 | +0.12% (noise) |

Unlike Q6 (where the encoder trunk's pretrained basis made "less LR than
decoder" the a priori expectation), Grid B has no directional prior: 3 of
conv_in's 4 channels carry the pretrained basis (same argument as the
encoder), but the 4th is mean-init and freshly grown, arguably wanting
*more* relative LR, not less -- hence testing both directions.

**Read-out criterion:** compare epoch-15 val_vrmse within each grid
separately against the ~1.1% seed-noise floor (Open Question 1) -- a gap
smaller than that is noise, same standard as every sweep since the floor
was measured. The two grids are independent questions (decoder LR vs
conv_in's relative LR); don't combine them into a single ranking.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) for every
arm. Elevated watch on Grid A's 2.0x arm specifically (decoder LR 1e-4 is
double every decoder LR used elsewhere in this ledger, and the decoder is
552M mostly-pretrained params) -- same failure signature Q6's 1.0x
encoder arm was screened for (train_loss regressing epoch-over-epoch, or
val_vrmse heading toward 0.15-0.2+ instead of down from the
~0.10-0.11 baseline-adapter-only starting point). Grid B's arms carry
lower risk regardless of direction since conv_in is only 221,312 params.

**RESULT (2026-08-09), all 4 new jobs launched 2026-08-08 10:43 CEST, 8
tiles each, ~5.5h wall-clock apiece (declr0.5x 5h35m, declr2x 5h30m,
adapterlr0.3x 5h28m, adapterlr3x 5h34m):**

**ANSWERED: both grids show the same pattern -- lowering either LR from
the baseline hurts (real effect, clears the ~1.1% noise floor), raising
either does nothing distinguishable (inside the noise floor).** The
current baseline values (decoder LR 5e-5, `adapter_lr_multiplier` 1.0x)
are already at or near the optimum on both axes independently. No config
change adopted -- this closes Q9 rather than moving the baseline. (Grid
B's original "no directional prior" framing turned out not to matter in
practice: neither direction helped, only lowering hurt, same as Grid A.)

**Same single-variable caveat as Q7's ep18 check applies to all 4 arms
here** (2026-08-09): all four were submitted in the same 10:43 CEST
2026-08-08 batch as `ep18`, so all four resolved configs carry
`data.num_dataloader_workers: 4` against the reused 1.0x point's
(job 521887) `2` -- not expected to move converged `val_vrmse`, but not a
perfectly clean diff against 521887 specifically. Doesn't change either
grid's verdict (the "lower hurts, higher doesn't help" pattern is
consistent across both LR directions in both grids, which a dataloader
setting wouldn't produce).

**Kill-criterion note, corrected 2026-08-09 after a full per-epoch
re-check (the first pass only looked at final rows).** Two arms, not
one, show a `train_loss` rise that literally triggers "epoch N above
epoch N-1's":
- **declr2x** (the arm flagged above for "elevated watch" -- decoder LR
  1e-4, double every decoder LR used elsewhere): epoch 1->2 rose
  0.10465 -> 0.15253.
- **adapterlr0.3x**, not flagged in advance and a bigger spike: epoch
  2->3 rose 0.07299 -> **0.22332** (~3x), driven almost entirely by
  `train_h1` (0.00046 -> 0.00260) and `train_h2` (0.00083 -> 0.00688)
  jumping for that one epoch. Crucially, `val_total_loss` and `val_vrmse`
  both kept *improving* across the same epoch (0.0815 -> 0.0757, 0.2034
  -> 0.1955) -- this was a noisy train-side epoch (a few outlier
  batches/gradients skewing that epoch's average), not a real regression,
  since held-out performance never wavered.

Every other arm in both grids stayed monotonically decreasing
epoch-over-epoch. Neither flagged run diverged (final val_vrmse: declr2x
0.08470, adapterlr0.3x 0.08792, both finishing in line with their
neighbors) and neither approached the 0.15-0.2+ danger range the original
watch-note called out, so both read as the same benign early-training
noise seen elsewhere in this pull (e.g. the Q12 overfit diagnostic
below), not the instability the watch was guarding against.

### Channel-order sweep (Open Question 8, new 2026-08-06)

**THE QUESTION:** does the *order* in which the 4 physical fields (density,
momentum_x, momentum_y, pressure) are stacked into the VAE's channel
dimension affect val_vrmse? Motivation is mechanical, not superstitious:
`inflate_vae_io_channels` (`windinet/vae_adapter.py:219-319`) always grows
`encoder.conv_in`/`decoder.conv_out` at **index 3** -- indices 0-2 keep the
pretrained LTX-Video RGB conv weights byte-for-byte (before training),
index 3 is seeded with `inflate_init: mean` (the mean of the other three).
Whichever physical field is placed last therefore starts from a different,
averaged initialization than the three fields placed at 0-2, which start
from a real (if domain-mismatched) pretrained basis. "Established" above
already shows init matters enormously (`inflate_init: random` costs +72%
val_vrmse) -- this asks whether a *milder* version of the same effect shows
up just from reassigning which field sits in which of the 4 slots, given
that every slot is fully trainable (`encoder.conv_in` is never frozen in
these runs) but 1800 optimizer steps may not be enough to fully escape a
mediocre starting point.

**Scope:** density is kept at index 0 in every arm (matching every prior
run). Only momentum_x / momentum_y / pressure are permuted across indices
1-3, giving 3! = 6 total arrangements. One of the six is the existing
baseline itself (`finetune_vae_baseline.yaml`, val_vrmse 0.095396, tag
`mx_my_pr` below) -- only the other 5 need a new run.

**Baseline choice:** built on `finetune_vae_baseline.yaml` (frozen encoder
trunk), not `finetune_vae_whole_structure_baseline.yaml`. The whole-structure
baseline is itself still mid-sweep (encoder-LR sweep, Open Question 6, not
yet concluded) -- stacking a second unconfirmed variable on top of it would
confound both reads. `finetune_vae_baseline.yaml` is the most-reproduced
config in this ledger (bit-exact across jobs 21632/21666), so it's the
cleanest single-variable base.

**Code change required** (this was not a pure-YAML experiment): channel
order was previously hardcoded in `build_shockwave_video`'s `torch.stack(...)`
call (`windinet/training/shockwave_data.py`). Added `data.channel_order:
list[str]` to `VaeDataConfig` (`windinet/config.py`), validated as a
permutation of the four field names; `build_shockwave_video` now stacks
according to it (default unchanged: density, momentum_x, momentum_y,
pressure). `adapter.channels` was already a same-shaped list but was **only
metadata** (labels for logging/visualization, silently disconnected from the
real stack order) -- `VaeTrainerConfig` now has a `model_validator` requiring
`data.channel_order == adapter.channels` exactly, so the two can no longer
silently drift apart the way the pre-existing `adapter.channels` field could
have. All 27 pre-existing configs still load unchanged (both fields default
to the same canonical order).

**Grid** (tag = order of momentum_x/momentum_y/pressure after density;
mx/my/pr abbreviate the three):

| tag | channel_order (index 0-3) | config | val_vrmse |
|---|---|---|---|
| `mx_my_pr` | density, momentum_x, momentum_y, pressure | `finetune_vae_baseline.yaml` (reuse) | 0.09540 |
| `mx_pr_my` | density, momentum_x, pressure, momentum_y | `finetune_vae_baseline_chorder_mx_pr_my.yaml` (job 522472) | 0.10136 |
| `my_mx_pr` | density, momentum_y, momentum_x, pressure | `finetune_vae_baseline_chorder_my_mx_pr.yaml` (job 522473) | **0.09536** |
| `my_pr_mx` | density, momentum_y, pressure, momentum_x | `finetune_vae_baseline_chorder_my_pr_mx.yaml` (job 522474) | 0.10043 |
| `pr_mx_my` | density, pressure, momentum_x, momentum_y | `finetune_vae_baseline_chorder_pr_mx_my.yaml` (job 522475) | 0.09922 |
| `pr_my_mx` | density, pressure, momentum_y, momentum_x | `finetune_vae_baseline_chorder_pr_my_mx.yaml` (job 522476) | 0.09841 |

Single variable per arm vs `finetune_vae_baseline.yaml`: `data.channel_order`
+ `adapter.channels` (must move together) + `data.channel_mean`/`channel_std`
(re-paired positionally to the new order -- same four per-field numbers as
the baseline, just reindexed, not re-measured). Everything else (seed 42,
15 epochs, wsd, h1 50, lr 5e-5, adapter_lr_multiplier 1.0) unchanged.

**ANSWERED (2026-08-08): channel order matters, and it tracks index 3.**
6-arm spread is 0.0954-0.1014 (~6.1% relative), well clear of the ~1.1%
seed-noise floor measured by the same pull (Open Question 1) -- this is a
real effect, not noise. Grouped by which field sits in index 3 (the fresh
mean-init slot):

| field at index 3 | arms | mean val_vrmse |
|---|---|---|
| pressure | `mx_my_pr`, `my_mx_pr` | **~0.0954** (best) |
| momentum_x | `my_pr_mx`, `pr_my_mx` | ~0.0994 |
| momentum_y | `mx_pr_my`, `pr_mx_my` | ~0.1003 (worst) |

This confirms the `inflate_vae_io_channels` mechanism hypothesis below:
whichever field is freshly mean-initialized (index 3) has a measurable,
consistent effect on the converged result, and pressure tolerates that
initialization best while momentum_y tolerates it worst. `mx_my_pr`
(density, momentum_x, momentum_y, pressure -- the existing convention
everywhere else in this ledger) stays the default: it's already in the
best-performing (pressure-last) group, so there's no reason to switch.

**Kill criterion:** none triggered (epoch 2 `train_loss` never exceeded
epoch 1's in any arm) -- this doesn't touch LR, capacity, or schedule, so
the risk profile matched the already-reproduced baseline as expected.

**Known gap, deliberately NOT fixed yet (2026-08-06):** `channel_order` is
only threaded through the VAE training/eval loop (`vae_trainer.py`'s three
`build_shockwave_video` call sites). `scripts/preprocess_dataset.py` and
`scripts/inference_shockwave.py` both still hardcode the default order
(`CHANNEL_NAMES`) independent of any checkpoint's actual `channel_order` --
so if a non-default-order arm from this sweep is ever taken into stage-2
(DiT) latent preprocessing or inference, physical fields will land in the
wrong channel slots **silently**. The fingerprint check in
`windinet/vae_adapter.py` (`_compute_fingerprint`-style guard around lines
372-418) does not catch this either -- it only hashes `encoder.conv_in`
weights plus the raw `channel_mean`/`channel_std`/`normalization_clip`
numbers, never field identity/order, so a channel-order mismatch between
training and inference passes it undetected. **Do not reuse a
non-`mx_my_pr` checkpoint for preprocessing/inference until both scripts
are updated to read and thread through `channel_order`.** Training and
eval within this sweep are unaffected (verified: stacking and
normalization both re-paired correctly, see the code-review conversation
that added this feature).

Launch (once ready):
```
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_baseline_chorder_mx_pr_my.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_baseline_chorder_my_mx_pr.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_baseline_chorder_my_pr_mx.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_baseline_chorder_pr_mx_my.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_baseline_chorder_pr_my_mx.yaml
```

### Channel-order sweep v2 + copy-init extension (Open Questions 8/10 revisited, resolved 2026-08-12)

**ANSWERED: baseline reconfirmed; the pressure-last mechanism (Q8) persists
under whole-structure unfreeze; copy-init (Q10) stays null on 5 of 6 arms
but flips to a real win on the 6th; density's index-0 position still has no
special status.** Follow-up to promoting ep18 into
`finetune_vae_whole_structure_baseline.yaml` (Open Question 7, above):
the channel-order sweep (Q8) and copy-init question (Q10) were both
originally answered on older baselines (Q8 on the frozen-trunk baseline,
Q10 on `pr_my_mx` under that same frozen-trunk lineage) -- neither has
been checked against the current 18-epoch whole-structure baseline. This
batch does three things at once:

1. **Confirms the promoted baseline reproduces** ~0.0834 under its own
   name (the header changelog in `finetune_vae_whole_structure_baseline.
   yaml` flags this as still needed).
2. **Extends Q10's copy-init question from 1 arm to all 4** of Q8's
   original "momentum-split" arrangements (previously only `pr_my_mx` had
   a copy-init variant) -- each paired with a same-arrangement mean-init
   control run on the *new* baseline, so the copy-vs-mean diff is clean
   (not confounded by the frozen-trunk-vs-whole-structure difference the
   original Q10 read-out had against `pr_my_mx`'s old number).
3. **Tests a variable Q8 never varied: density's own position.** Every
   arm in the original 6-arm Q8 sweep fixed `density` at index 0 by
   convention, never as a swept variable. Two new arrangements here move
   `pressure` into index 0 and `density` into index 1 instead, on top of
   the existing momentum-split logic -- if these read differently from
   their density-at-0 counterparts, index-0 was doing more than just
   "holds whichever field is there," which Q8's original framing didn't
   distinguish.

**The 6 channel-order arms** (each run twice: mean-init control +
copy-init), all built on `finetune_vae_whole_structure_baseline.yaml`
(18 epochs, 0.3x encoder LR, whole-structure unfreeze, seed 42) with
**only** `data.channel_order` + `adapter.channels` (+ `adapter.
inflate_init` / `adapter.inflate_copy_channel` for the copy variant)
changed per protocol point 1:

| # | channel_order | fresh slot | copy source | prior result (old frozen-trunk baseline, NOT a clean diff) |
|---|---|---|---|---|
| 1 | density, pressure, momentum_y, momentum_x | momentum_x | momentum_y | `pr_my_mx`, job 522476, 0.09841 (mean); copy-init already tested once, job 523495, 0.09831 -- see "Copy-init" above |
| 2 | density, pressure, momentum_x, momentum_y | momentum_y | momentum_x | `pr_mx_my`, job 522475, 0.09922 (mean only) |
| 3 | density, momentum_x, pressure, momentum_y | momentum_y | momentum_x | `mx_pr_my`, job 522472, 0.10136 (mean only, worst of original 6) |
| 4 | density, momentum_y, pressure, momentum_x | momentum_x | momentum_y | `my_pr_mx`, job 522474, 0.10043 (mean only) |
| 5 | **pressure, density**, momentum_x, momentum_y | momentum_y | momentum_x | none -- density has never been moved off index 0 before |
| 6 | **pressure, density**, momentum_y, momentum_x | momentum_x | momentum_y | none -- density has never been moved off index 0 before |

Configs: `finetune_vae_whole_structure_baseline_chorder_<tag>.yaml`
(mean) / `finetune_vae_whole_structure_baseline_chorder_<tag>_copyinit.
yaml` (copy), `<tag>` in `{pr_my_mx, pr_mx_my, mx_pr_my, my_pr_mx,
pr_d_mx_my, pr_d_my_mx}`. All 12 verified loading through
`VaeTrainerConfig`; spot-checked `pr_d_mx_my`'s resolved `channel_mean`/
`channel_std` against `normalization_stats_file` -- correctly reorders to
`[pressure, density, momentum_x, momentum_y]` (pressure's mean 0.9032
first, density's 0.9051 second), confirming the density-moved case is
handled the same way arm-permutation already was for Q8/Q10.

**Read-out criteria:**
- Arms 1-4: compare each arm's copy-init vs its own mean-init sibling
  (same channel_order, same new baseline) against the ~1.1% seed-noise
  floor (Open Question 1) -- does copy-init help on any arm now that the
  comparison is clean, not just `pr_my_mx` (already null on the old
  comparison, job 523495 vs 522476).
- Arms 1-4 mean-init vs each other, and vs arms 5-6's mean-init: does the
  ~6% spread Q8 found on the old baseline persist under whole-structure
  unfreeze + 18 epochs, and do arms 5-6 (density moved) fall inside or
  outside that spread -- if inside, index-0 has no special status beyond
  "holds a pretrained slot like any other"; if arms 5-6 are outliers
  (better or worse), density's specific position matters beyond that.
- All arms vs the reconfirmed baseline (`finetune_vae_whole_structure_
  baseline.yaml` under its own name, expected ~0.0834): sanity bound, not
  expected to beat the default `mx_my_pr` order given Q8's original
  finding that pressure-last arrangements won.

**Kill criterion:** usual (epoch 2 `train_loss` above epoch 1's) for
every arm -- given this pull's own finding that isolated epoch-to-epoch
`train_loss` bumps are common and usually benign (declr2x, adapterlr0.3x,
the Q12 overfit diagnostic, all above), don't kill on a single bump alone;
check whether `val_vrmse` also regresses before treating one as real.

**RESULT, jobs 523577-523589 (sng_pvc), launched 2026-08-09 07:22 CEST, all
13 runs complete, 18 epochs, LR decayed to floor:**

**1. Baseline reconfirmed.** Job 523577 (`finetune_vae_whole_structure_
baseline.yaml` under its own name): val_vrmse **0.08360**, vs the original
ep18 promotion's 0.08344 (job 523486) -- +0.19%, inside the ~1.1% noise
floor. Reproduces as expected; this is the `0.08360` cited as "the
baseline" throughout every other result in this pull (256x256, h2,
GradNorm, h1x2/ssimx2/mlw, KL, above).

**2. All 12 new arms, mean-init vs copy-init:**

| tag | channel_order | fresh (copy src) | mean-init val_vrmse | copy-init val_vrmse | copy vs mean |
|---|---|---|---|---|---|
| `pr_my_mx` | d, pr, my, mx | mx (my) | 0.08505 (523578) | 0.08550 (523579) | +0.53% |
| `pr_mx_my` | d, pr, mx, my | my (mx) | 0.08507 (523580) | 0.08563 (523581) | +0.66% |
| `mx_pr_my` | d, mx, pr, my | my (mx) | 0.08678 (523582) | 0.08717 (523583) | +0.45% |
| `my_pr_mx` | d, my, pr, mx | mx (my) | 0.08699 (523584) | 0.08712 (523585) | +0.15% |
| `pr_d_mx_my` | pr, d, mx, my | my (mx) | 0.08542 (523586) | 0.08581 (523587) | +0.46% |
| `pr_d_my_mx` | pr, d, my, mx | mx (my) | 0.08759 (523588) | **0.08584 (523589)** | **-2.00%** |

**Read-out 1 (copy-init vs mean-init, per arm):** 5 of 6 arms land inside
the ~1.1% noise floor, copy-init if anything very slightly worse --
reconfirms Q10's original null result (`pr_my_mx`, job 523495 vs 522476)
now on the whole-structure lineage, clean single-variable comparison this
time. **`pr_d_my_mx` is the one exception**: copy-init beats mean-init by
2.00%, clearing the floor -- but this arm's mean-init sibling (0.08759) is
also the single worst of all 12 arms, so this reads more like "copy-init
avoids whatever made that particular mean-init arrangement land badly"
than a robust general copy-init win -- consistent with the two
density-moved copy-init arms converging to nearly the same value (0.08581
vs 0.08584) while their mean-init siblings differ by 2.5% (0.08542 vs
0.08759), i.e. copy-init looks *more* consistent across arrangements here,
not just occasionally better. Worth a closer look before generalizing from
n=1, not enough on its own to flip Q10's overall null verdict.

**Read-out 2 (does Q8's ~6% spread / pressure-last mechanism persist?):**
yes. All 6 mean-init arms (0.08505-0.08759, ~3% spread among themselves)
land **worse** than the default `mx_my_pr` order's 0.08360 -- every arm
here deliberately moves pressure off index 3 (since the original Q8 winner
keeps it there), so this is exactly the expected direction, now confirmed
under whole-structure unfreeze + 18 epochs rather than just the old
frozen-trunk baseline. No arm beats the default; `mx_my_pr` stays the
config's channel order.

**Read-out 3 (arms 5-6, density moved off index 0):** both fall inside
arms 1-4's mean-init spread (0.08542 and 0.08759 vs range 0.08505-0.08699)
rather than standing out as clear outliers in either direction -- supports
"index-0 has no special status beyond holding a pretrained slot like any
other," same conclusion the read-out criterion flagged as the "inside the
spread" outcome.

**Launch (13 jobs total -- run these from the cluster, this environment
has no `sbatch` access):**
```
# 1. Confirm the promoted baseline (18ep + 0.3x encoder LR) reproduces ~0.0834
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml

# 2. Six channel-order arms x {mean, copy}
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_my_mx.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_my_mx_copyinit.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_mx_my.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_mx_my_copyinit.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_mx_pr_my.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_mx_pr_my_copyinit.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_my_pr_mx.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_my_pr_mx_copyinit.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_d_mx_my.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_d_mx_my_copyinit.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_d_my_mx.yaml
sbatch jobs/sng_pvc/finetune_vae.sbatch configs/finetune_vae/finetune_vae_whole_structure_baseline_chorder_pr_d_my_mx_copyinit.yaml
```
Each is an independent 8-tile, 18-epoch run; based on this pull's timings
(~1200-1250s/epoch train + ~38s eval on the current eval-fix'd codebase),
expect each job to take roughly **6.5-7h wall-clock** (18 epochs at the
whole-structure baseline's own per-epoch rate). 13 jobs x ~7h is a
substantial cluster-time commitment -- consider staggering submission if
sng_pvc's queue/fair-share makes running all 13 concurrently impractical.

### Seed noise floor sweep (Open Question 1, configs written 2026-08-06)

**THE QUESTION:** every result in this ledger uses `seed: 42`, chosen once,
arbitrarily, never varied as an experimental knob. Every "~1-2%, treat as
noise" caveat scattered through this document (tail unfreeze +1.8%,
slowdecay -1.3%, the encoder-LR-sweep arms' epoch-5 gaps, etc.) is a guess
that such gaps are inside seed noise -- nobody has actually measured what
that noise band is. This sweep measures it directly: re-run
`finetune_vae_baseline.yaml` at `seed: 1` and `seed: 2`, holding every other
config field fixed, and compare both against the existing `seed: 42` result
(0.095396, jobs 21632/21666).

**Important: `cfg.seed` controls two things at once, not one.**
`vae_trainer.py:478` (`set_seed(cfg.seed)`, training-run randomness -- data-
loader shuffle order, etc.) and `vae_trainer.py:498`
(`torch.Generator().manual_seed(cfg.seed)`, the train/eval split itself) both
key off the same field. Changing `seed` therefore also changes *which 675
simulations are held out for eval* -- a seed-1 run is not evaluated on the
same validation set as the seed-42 baseline. This is deliberate, not a
confound to engineer away: since every other run in this ledger also only
ever had one arbitrary seed's worth of split, "how much does val_vrmse move
under a different arbitrary seed choice, split included" is the actual
question that needs answering to interpret this ledger's existing small
gaps -- a narrower measurement that held the eval split fixed and varied
only training-run randomness would answer a different, less relevant
question. (If that narrower question is ever wanted later, it would need a
new `data.split_seed` field decoupled from `cfg.seed` -- not implemented,
not needed for this sweep.)

**Grid:**

| seed | config | val_vrmse |
|---|---|---|
| 42 | `finetune_vae_baseline.yaml` (reuse) | 0.09513 |
| 1 | `finetune_vae_baseline_seed1.yaml` (job 522477) | 0.09519 |
| 2 | `finetune_vae_baseline_seed2.yaml` (job 522478) | 0.09411 |

Single variable vs `finetune_vae_baseline.yaml`: `seed` only (42 -> 1, 42 ->
2). `output_dir`/`wandb.tags` also change (required for every run to get
its own directory per this ledger's own protocol point 6) but are not
experimental variables.

**ANSWERED (2026-08-08): seed noise floor is ~1.1%.** Spread across the
three seeds is 0.09411-0.09519 (0.00108 absolute, ~1.1% relative to their
mean). This is a **small** spread -- nowhere near the ~9% fullenc-vs-
baseline gap from the head-vs-tail sweep, so that result (and the encoder-
LR sweep's 0.03x/0.3x-vs-baseline gaps, both >1.7%) stand as real effects,
not seed artifacts. Going forward, **use ~1.1% as the actual noise
threshold** in place of every earlier "~1-2%, treat as noise" placeholder
in this document -- concretely this affects:
- Open Question 5's slowdecay verdict (-1.3% vs baseline) -- now barely
  *outside* the measured floor rather than comfortably inside an assumed
  2% one; that "ANSWERED: yes, decay length was right" conclusion is
  weaker than it reads and would benefit from a repeat run if it becomes
  load-bearing for a future decision.
- Open Question 6's encoder-LR sweep -- 0.3x-vs-1.0x (~0.8%) is inside the
  floor (not distinguishable), but 0.03x/0.3x-vs-0.1x (2.1%/1.7%) both
  clear it (real effects).
- Open Question 8's channel-order sweep -- all pairwise gaps there (up to
  ~6.1%) clear the floor by a wide margin (real effect).

**Kill criterion:** none triggered (epoch 2 `train_loss` never exceeded
epoch 1's in either seed run).

## sng_pvc throughput diagnostic

**THE PUZZLE:** sng_pvc job 520211 (`finetune_vae_baseline.yaml`, 8 Intel XPU
tiles) averaged 38.7 min/epoch -- roughly the SAME as lundquist's 2-GPU
(A6000) estimate of 32-36 min/epoch, despite 4x the raw compute. Something
was eating the expected speedup.

**METHOD:** `windinet/training/vae_trainer.py` now logs a per-epoch phase
breakdown (`train=`/`eval=`/`viz=`/`ckpt=`/`total=`), so a 2-epoch throwaway
run (epoch 1 = warmup/compile, epoch 2 = clean reading) isolates where the
time goes without re-reading a full job. Three single-variable diagnostics,
`jobs/sng_pvc/finetune_vae_diag.sbatch` (+ `finetune_vae_diag_4rank.sbatch`
for the 4th):

| exp | job | change | epoch-2 train | epoch-2 eval | epoch-2 total | verdict |
|---|---|---|---|---|---|---|
| 0 (base) | 520300 | none (`num_dataloader_workers: 0`, full ckpt+viz) | 1426.8s | 1002.1s | 2445.4s (40.8 min) | reference |
| 1 (H1: I/O-bound step) | 520301 | `num_dataloader_workers: 0 -> 2` | 872.3s | 1045.5s | 1933.0s (32.2 min) | **train time -39%, total -21%** -- confirmed |
| 2 (H3: ckpt/viz I/O to DSS) | 520302 | `save_last_state: false`, `visualization.enabled: false` | 1441.9s | 1002.1s | 2445.4s (40.8 min, unchanged) | ckpt+viz was only ~16s/epoch of the base run (`viz=11.7s ckpt=4.3s`) -- **ruled out**, not the bottleneck |
| 3 (H2: comms topology) | 520303 | 4 ranks (COMPOSITE) x accum 8, vs 8 ranks (FLAT) x accum 4 | -- | -- | **crashed every rank** | `RuntimeError: Invalid mt19937 state` -- untested, see fix below |
| 4 (H1 cont.: workers=4) | 520456 | `num_dataloader_workers: 2 -> 4` | 869.3s | 921.1s | 1805.9s (30.1 min) | **-6.6% total vs workers=2, no crash** -- best confirmed so far |
| 5 (H1 cont.: workers=8) | 520457 | `num_dataloader_workers: 2 -> 8` | 866.8s | 923.3s | 1809.6s (30.2 min) | statistically same as workers=4 -- no further gain past 4 |
| 6 (H2 retry, post-fix) | 520459 | 4 ranks (COMPOSITE), `rng_types=[]` fix applied | -- | -- | **hit time limit mid-epoch-1** | no mt19937 crash this time, but no timing data either -- H2 still open, needs a longer time limit |

**Exp 3 crash and fix:** all 3 non-rank-0 processes died with `Invalid
mt19937 state` inside `accelerate/utils/random.py`'s `synchronize_rng_state`.
Root cause: Accelerate's default `rng_types=["generator"]` broadcasts the
train-loader sampler's RNG state from rank 0 to every other rank on every
`accelerator.prepare()` call; that broadcast corrupts in transit through
oneCCL specifically under `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` (4-rank) --
the identical 8-tile FLAT launches (exp 0-2) hit the same code path every
epoch without issue. Fixed in `windinet/training/vae_trainer.py`
(`_setup_accelerator`): `rng_types=[]`, safe because `train()` calls
Accelerate's `set_seed(cfg.seed)` identically on every rank *before* the
DataLoader/sampler is built, and the train/eval split uses its own
`cfg.seed`-derived `torch.Generator`, not the global RNG -- so every rank's
default sampler already gets an identical seed without any cross-rank
broadcast; the sync was redundant (and buggy) in this codepath. **Exp 3 has
not been re-run since the fix** -- H2 (communication topology) is still
open.

**ESTABLISHED:** `num_dataloader_workers: 0` on sng_pvc (`windinet/cluster_config.py`
`CLUSTER_DEFAULTS["sng_pvc"]`) was a crash-avoidance choice, not a
performance-tuned one -- the comment there notes 4 workers x 8 ranks crashed
opening `train.h5` concurrently on the DSS network filesystem. `workers: 2`
is confirmed safe at 8 ranks and cuts wall-clock per epoch by ~21%
(train-step time by ~39%) with no code change beyond the config's
`data.num_dataloader_workers` field. **Promoted to the cluster default**
(`CLUSTER_DEFAULTS["sng_pvc"]["num_dataloader_workers"] = 2`) -- every future
sng_pvc launch (including the head-vs-tail unfreeze sweep in Open Question 3)
gets this for free with no per-config change.

**Promoted further (2026-08-06):** exp 4-5 above show `workers: 4` shaves
another ~6.6% off `workers: 2` (30.1 vs 32.2 min/epoch, mostly from eval time
-- 921.1s vs 1045.5s -- not train, which was flat 869.3s vs 872.3s) with no
crash, and `workers: 8` gains nothing more past 4. Both diagnostics were only
2 epochs, so the transient-open-failure risk the old N=4 comment warned about
was never confirmed ruled out over a full 15-epoch run before this promotion
-- `CLUSTER_DEFAULTS["sng_pvc"]["num_dataloader_workers"]` is now **4**
(`windinet/cluster_config.py`), a deliberate decision to accept that
unconfirmed risk rather than spend a dedicated full-length run just to
de-risk it first. **Watch the first few post-change full runs' `.err` logs**
for the `train.h5`-concurrent-open failure mode the old N=4 comment
described; if it recurs, drop back to 2 and note it here.

**Correction (2026-08-06, later same day):** the "-6.6%, mostly from eval
time" reading above is likely **not actually caused by `num_dataloader_workers`**.
Before the "Eval parallelization fix" below, `eval_loader` was constructed
as `DataLoader(eval_set, batch_size=1, shuffle=False)` with no `num_workers`
argument at all -- it silently ignored `cfg.data.num_dataloader_workers` and
always ran with 0 workers, in both the `workers: 2` and `workers: 4`
diagnostics. The 921.1s vs 1045.5s eval-time gap between those two jobs was
therefore almost certainly ordinary run-to-run filesystem-contention noise
(consistent with the up-to-32% eval-time drift *within a single run*
documented in the `--time` paragraph below), not a real effect of the
`workers` config. `train` time being flat (869.3s vs 872.3s) across the same
two jobs is consistent with this: `train_loader` *does* read
`num_dataloader_workers`, and going 2->4 barely moved it. Net: the
`workers: 2 -> 4` promotion above still stands (it's free and not shown to
hurt), but don't expect the ~6.6% total-time win it was promoted on --
expect closer to 0, since eval (the larger of the two time sinks, ~41% of
total) was never actually reading this setting. The real eval fix is below.

Same date: `jobs/sng_pvc/finetune_vae.sbatch`'s `--time` was cut from
`24:00:00` to `12:00:00` (first set to `10:00:00`, then raised to `12:00:00`
before any job was submitted under the new default -- extra margin against
the eval-time-drift risk below). Every full-data 15-epoch run observed so
far finishes in 7h25m-8h54m at `workers: 2` (see the head-vs-tail sweep
table above); `workers: 4` should only shrink that further, so 12h leaves
comfortable margin even at the slowest observed runs (`fullenc`/`tail3`,
8h54m at `workers: 2`). Note `finetune_vae_baseline_tail3`'s own per-epoch
log showed eval time drifting up over the run (793s at epoch 6 -> 1048s at
epoch 15, train flat throughout) for reasons not yet understood (filesystem
contention from other jobs is the leading guess, not confirmed) -- worth
keeping an eye on if a run ever gets close to the limit despite this
margin. Shorter `--time` should also mean shorter queue wait, which is part
of the reason for the change; if a future config (e.g. a slower encoder-LR
sweep arm) still times out at 12h, raise it back up for that config
specifically rather than reverting the shared default.

**OPEN:** H2 (does 4-rank COMPOSITE vs 8-rank FLAT communication topology
matter?) -- job 520459 hit the diagnostic's time limit before finishing even
epoch 1 (the `rng_types=[]` fix itself worked, no crash), so re-run
`jobs/sng_pvc/finetune_vae_diag_4rank.sbatch` now that
`rng_types=[]` unblocks it. **Deprioritized** -- not being chased right now.

**H4 (node-count scaling): BLOCKED on rendezvous, diagnosed, fix applied but
UNTESTED (2026-08-06).** Does requesting 2 nodes (16 XPU tiles) train faster
than 1 node (8 tiles), or does crossing the inter-node fabric eat the extra
compute? Hypothesis: if communication was already close to the bottleneck
intra-node, going inter-node only makes it worse. Script:
`jobs/sng_pvc/finetune_vae_diag_2node.sbatch` (2-epoch diagnostic, same
`DIAG_TAG`/`DIAG_WORKERS`/`DIAG_EPOCHS` knobs as `finetune_vae_diag.sbatch`;
`patch_config_for_cluster` halves `gradient_accumulation_steps`
automatically to hold effective_batch=32 at 16 ranks). **Read-out:**
compare its epoch-2 `train=` time to job 520456's 869.3s (1 node, workers=4,
the fastest confirmed 1-node reference). **Kill criterion:** `train= >= 90%`
of 869.3s means 2-node scaling isn't worth chasing further (don't try 4
nodes next).

**First attempt, job 521396, failed before reaching epoch 1** -- this was
the first multi-node launch attempted on sng_pvc for this project, and the
rendezvous logic `finetune_vae.sbatch`/`finetune_vae_diag_2node.sbatch`
already carried for it (MASTER_ADDR = first node in `$SLURM_NODELIST`,
`--num_machines`/`--machine_rank` via srun's per-node `SLURM_PROCID`) had
never actually been exercised past `--nodes=1` before this:

```
torch.distributed.DistStoreError: wait timeout after 900000ms, keys:
/none/torchelastic/role_info/0, /none/torchelastic/role_info/1
```

**Diagnosis.** The `.err` log's c10d socket trace is the key evidence:

```
[W805 01:44:10...] waitForInput: poll for socket addr=[i20r02c04s06...]:39582, remote=[i20r02c04s04...]:45361 returned 0, likely a timeout
[W805 01:44:10...] ...timed out after 900000ms
[W805 01:44:11...] waitForInput: poll for socket addr=[i20r02c04s04...]:47330, remote=[i20r02c04s04...]:45361 returned 0, likely a timeout
[W805 01:44:11...] ...timed out after 900000ms
```

The first pair is the cross-node connection (node 1 -> node 0's rendezvous
store). The second pair is node 0 connecting to **its own** store --
loopback, never leaving the host. Both completed their TCP handshake
(there's a live `fd`/local+remote address on each) and then both hung
identically waiting for a response. Loopback hanging the same way as the
cross-node connection rules out an inter-node firewall/routing problem --
loopback traffic can't be blocked by a firewall between nodes -- and points
at the TCPStore **server** on the master never responding to any client,
local or remote. That matches a known failure signature of PyTorch's
libuv-backed TCPStore (the default since this torch version's `USE_LIBUV`
defaults to `"1"`; confirmed by reading `torch/distributed/rendezvous.py`
and `torch/distributed/elastic/utils/distributed.py` in the installed
package), which has reported hangs on some HPC/interconnect setups where
the older, pre-libuv TCPStore implementation works fine.

Also present in the same `.err`, but judged **not** the cause: `slurmstepd:
error: Unable to create TMPDIR [/hppfs/scratch/0D/go76fuz2/tmp]` (falls
back to per-node local `/tmp`). This is emitted by `slurmstepd` itself
before the sbatch script's own commands run, so it can't be fixed from
inside the script -- a one-time `mkdir -p /hppfs/scratch/0D/go76fuz2/tmp`
on the login node would clear it -- but it's a local-vs-shared-tmp issue,
which wouldn't explain a *loopback* connection also hanging, and every
earlier 1-node job hit the same TMPDIR fallback without failing.

**Fix applied to both `finetune_vae_diag_2node.sbatch` and
`finetune_vae.sbatch`:** `export USE_LIBUV=0`, forcing the older TCPStore
backend. Cheap/safe regardless of whether this is really the cause: it's a
no-op for every already-working 1-node job (`num_machines=1` never
exercises the real multi-node TCPStore path this flag affects). The diag
script additionally sets `TORCH_DISTRIBUTED_DEBUG=DETAIL` and
`PYTHONUNBUFFERED=1` so that if `USE_LIBUV=0` alone doesn't fix it, the
*next* failure comes with far more to go on than a bare timeout.

**NOT YET RE-RUN.** This fix has not been tested against real hardware.
Launch with:
```
sbatch jobs/sng_pvc/finetune_vae_diag_2node.sbatch
```
If it still hangs at the same `_assign_worker_ranks`/`role_info` point even
with `USE_LIBUV=0`, the `TORCH_DISTRIBUTED_DEBUG=DETAIL` log should show
what the store is actually doing (or not doing) during the hang -- read
that before trying anything else. If it fails differently or earlier,
that's also useful signal (rules libuv out, points elsewhere).

## Eval parallelization fix, CONFIRMED WORKING (2026-08-06)

**Motivating question:** why does eval (675 sims) take almost as long per
epoch as train (3825 sims, ~5.7x more data), e.g. `finetune_vae_baseline_tail3`
(job 521395): train averaged 1220s/epoch, eval 853s/epoch -- eval is 41% of
total wall-clock despite the much smaller dataset.

**Root cause, found by reading `windinet/training/vae_trainer.py`:** eval was
never actually parallelized like train is.

1. The whole eval+log+viz+checkpoint block ran inside `if IS_MAIN_PROCESS:`
   -- only rank 0 ever called `_evaluate()`. On an 8-tile sng_pvc run, the
   other 7 ranks sat idle in `wait_for_everyone()` for the entire eval
   phase while rank 0 processed all 675 samples alone. Train, by contrast,
   is sharded 8-way (`train_loader = self._accelerator.prepare(train_loader)`,
   `train.py` around the optimizer setup) -- ~478 samples/rank, in parallel.
   Comparing the *per-active-rank* sample count is the right comparison:
   675 (eval, 1 rank) vs. 478 (train, per rank, x8 in parallel) -- eval was
   never the "smaller" job from a single rank's point of view.
2. `eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)` had no
   `num_workers` argument -- always 0 (synchronous, no prefetch), silently
   ignoring `cfg.data.num_dataloader_workers` regardless of its value. (This
   is also why the `workers: 2 -> 4` cluster-default promotion above almost
   certainly didn't do what its own read-out claimed -- see the correction
   in "sng_pvc throughput diagnostic".)

(A third suspected cause, `_evaluate` missing `@torch.no_grad()`, was
**checked and is wrong** -- the decorator was already there; a first read
that started exactly on the `def` line missed it on the line above. Not a
real factor, noted here only so it isn't re-suspected later.)

**Fix**, `windinet/training/vae_trainer.py`:

- `eval_loader` is now built from a hand-strided per-rank shard
  (`Subset(eval_set, list(range(rank, len(eval_set), world_size)))`)
  instead of `accelerator.prepare()` -- deliberately not using `prepare()`,
  because its default `even_batches` padding repeats a few samples across
  ranks to equalize batch counts, which would double-count them in the
  metric average. The strided split has zero padding: every sample lands on
  exactly one rank, shard sizes differ by at most 1 (checked for 675/8:
  sizes 85,85,85,84,84,84,84,84, sums to 675 exactly). It also now gets
  `num_workers=cfg.data.num_dataloader_workers`, same as `train_loader`.
- The `_evaluate()` call moved outside the `IS_MAIN_PROCESS` gate -- every
  rank now calls it (over its own shard); the surrounding log/metrics/viz/
  checkpoint code stays `IS_MAIN_PROCESS`-only, using `val_metrics` computed
  above.
- `_evaluate()` all-reduces its per-rank sums (`total_loss`, `vrmse`,
  `rmse`, `h1`, `ssim`, `mlw`, `count`) with `dist.all_reduce(..., op=SUM)`
  before averaging, in float32 -- same manual-all-reduce-instead-of-DDP
  pattern `_sync_grads` already uses for gradients, and for the same reason
  (the VAE isn't DDP-wrapped, so nothing does this automatically). Guarded
  by `self._accelerator.num_processes > 1`, matching `_sync_grads`'s guard,
  so single-process runs (the overfit diagnostics, single-tile debugging)
  are unaffected.
- `_save_visualization` still reads from `eval_loader`, called on rank 0
  only -- it now sees rank 0's *shard* (every `world_size`-th sample
  starting at index 0) instead of the full eval set's first
  `vis_cfg.num_samples` samples. Index 0 is still included (rank 0's stride
  starts at 0), so the reconstruction panels won't be empty or wildly
  different, but the exact sample IDs visualized each epoch will shift
  slightly from pre-fix runs -- not a correctness issue, just don't expect
  panel sample IDs to match older runs' when comparing side by side.

**Expected effect:** eval should drop from ~one-rank-serial-675-samples to
~one-rank-parallel-84/85-samples-with-prefetch -- directionally a large cut
to the eval share of wall-clock (currently ~41% of total), though the exact
number needs a real run to confirm (rough pre-fix numbers, not a promise:
if eval scaled purely with shard size it'd be ~8x fewer samples per rank
plus whatever the worker prefetch buys, but network-filesystem contention
under concurrent multi-rank reads is untested and could eat into that).

**CONFIRMED, job 522099 (`DIAG_TAG=evalfix_confirm`, 2 epochs,
`finetune_vae_baseline.yaml`):**

| epoch | train= | eval= (pre-fix was ~800-1050s) |
|---|---|---|
| 1 | 895.9s | **38.1s** |
| 2 | 868.1s | **38.0s** |

**~22-27x faster** than the pre-fix per-epoch eval times seen throughout
this ledger (e.g. `finetune_vae_baseline_tail3`'s 793-1048s/epoch) -- far
beyond the "directionally large, exact number TBD" estimate above. Eval
went from ~41% of total wall-clock to a rounding error. Correctness check:
val_vrmse landed at 0.297 (epoch 1, lr still mid-warmup) -> 0.136 (epoch 2,
schedule already compressed to the floor for a 2-epoch diagnostic run) --
sane, in-range numbers, no NaN/blowup, consistent with the all-reduce
producing a real average rather than a corrupted one. No crash, no
DSS-locking symptom despite 8 ranks now reading `train.h5` concurrently
during eval too.

**File-locking question (Exp 4) ALSO ANSWERED, job 522100
(`DIAG_TAG=filelock_true DIAG_FILE_LOCKING=TRUE`):** finished clean, no
"Unable to synchronously open object" crash. Timing statistically
identical to 522099 (train 897.1s/869.8s, eval 37.9s/37.9s -- within noise
of the `FALSE` run). **Conclusion: `HDF5_USE_FILE_LOCKING=FALSE` is not
(or no longer) load-bearing for eval's newly-concurrent reads** -- but
since `FALSE` costs nothing and already works, there's no reason to flip
the default; this just confirms the eval fix didn't quietly reintroduce
the crash risk `FALSE` was set to avoid on the train side.

Both jobs used `finetune_vae_baseline.yaml` (`jobs/sng_pvc/finetune_vae_diag.sbatch`'s
default `BASE_CONFIG`), kept separate from the model-architecture sweeps
(head-vs-tail, encoder-LR) so infra effects aren't confounded with which
encoder modules are unfrozen -- launched together:

```
DIAG_TAG=evalfix_confirm sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
DIAG_TAG=filelock_true DIAG_FILE_LOCKING=TRUE sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
```

## New loss components (H2/PCC/VRMS/KL), opt-in, NOT ENABLED anywhere yet (2026-08-06)

Four new loss components added to `windinet/losses/` alongside the existing
RMSE/H1/SSIM/MLW, for future experiments to opt into -- no config has been
changed to actually use any of them yet, this is capability only.

| name | file | needs | what it penalizes |
|---|---|---|---|
| `h2` | `h2_semi_norm.py` | pred, target | second-order spatial derivatives (curvature) -- one order past H1's first derivatives, e.g. over-smoothed shock peaks that H1 alone can miss |
| `pcc` | `pcc.py` | pred, target | 1 - Pearson correlation, per sample -- structural/pattern match, magnitude-invariant (complements RMSE, which is magnitude-sensitive) |
| `vrms` | `vrms.py` | pred, target | variance-normalized RMSE -- the same metric already tracked as `val_vrmse`, now also usable as a training objective, not just monitored. `VaeTrainer.vrmse()` (the eval metric) now wraps this instead of duplicating the formula |
| `kl` | `kl_divergence.py` | encoder posterior mean+logvar (NOT pred/target) | standard VAE ELBO regularizer, KL(q(z\|x) \|\| N(0,I)). Only computed when the caller supplies `latent_mean`/`latent_logvar`; the other three are always computed regardless of weight, same as MLW already was |

**How to enable one:** add it to a config's `loss_weighting.weights` with a
nonzero weight, e.g. `vrms: 1.0`, alongside the existing four -- no other
change needed. `windinet/config.py`'s `LossWeightingConfig` validator now
requires the original four (`rmse`/`h1`/`ssim`/`mlw`) but only *permits*
these as extras, so every existing checked-in config is valid unchanged
(they don't list the new names, which is fine -- an unlisted name just
contributes 0, same mechanism MLW's `weights: {mlw: 0.0}` already relied on
before this).

**KL needed extra plumbing the other three didn't.** RMSE/H1/H2/SSIM/MLW/
PCC/VRMS only need `(pred, target)`, so they slot straight into
`reconstruction_losses()`. KL needs the encoder's raw posterior
distribution instead, which `VaeTrainer._encode` used to discard (it only
returned `out.latent_dist.mean`, rescaled, for the reconstruction path).
`_encode`/`_forward_pass` now also return `(posterior_mean, posterior_logvar)`
-- the RAW (pre-rescale) distribution -- threaded through to
`reconstruction_losses(..., latent_mean=..., latent_logvar=...)` at both
call sites that matter (`train`'s loop, `_evaluate`). Deliberately does
**not** use diffusers' `DiagonalGaussianDistribution.kl()`: that method
hardcodes `dim=[1, 2, 3]` (assumes 4D image latents), but this project's
video-VAE latents are 5D `[B, C, T, H, W]` -- the diffusers method would
silently leave one spatial axis unreduced. `kl_divergence_loss` reduces
over every non-batch dim instead, robust to the actual shape.

Verified so far (pre-2026-08-08): each loss function alone
against hand-built tensors (sane values, correct edge cases -- e.g. `pcc_loss`
is ~0 for a signal against itself, `kl_divergence_loss` is exactly 0 for
mean=0/logvar=0); `reconstruction_losses()` returns the right key set with
and without `latent_mean`/`latent_logvar`; every one of the 28 real
`configs/finetune_vae/*.yaml` files still loads through `VaeTrainerConfig`
unchanged.

**Update (2026-08-09): this section was stale -- it has since been run,
repeatedly.** `h2`/`pcc`/`vrms`/`kl` are logged unconditionally (weight or
not, per the design above), so every one of the 9 whole-structure runs in
this pull (`ep18`, the two LR grids, `logdensity`, `rmseonly`, plus
`copyinit` on the frozen-trunk lineage) produced real `train_h2`/`val_h2`,
`train_pcc`/`val_pcc`, `train_vrms`/`val_vrms`, `train_kl`/`val_kl`
columns -- none of them have nonzero weight in any of these configs (only
`rmse`/`h1`/`ssim`/`mlw` appear under `loss_weighting.weights`), so this
is passive monitoring only, not yet a claim about their effect on
training. Values are sane and stable across all runs (`train_h2` tracks
`train_h1` at roughly 2x its magnitude throughout, as expected for a
higher-order derivative penalty; `pcc`/`vrms` track `ssim`/`rmse` shape
closely) -- **with one exception, `val_kl`:** it spikes 3-4 orders of
magnitude (up to ~644M vs. a steady-state ~195,000-215,000) in epoch 1 of
`copyinit` and `declr2x`, and epochs 1-2 of `adapterlr0p3x`; `ep18`'s
epoch 2 also spikes once. Four of the eight whole-structure-lineage runs
show no blip at all (`declr0p5x`, `adapterlr3x`, `logdensity`,
`rmseonly`). Since `kl` carries weight 0 everywhere, this has **no effect
on `train_total_loss`, gradients, or any reported `val_vrmse` result** --
it reads as the freshly-inflated encoder's posterior starting far from
`N(0,I)` before the first step or two of training pulls it back, and it
being inconsistent across otherwise-similar runs (different only in LR/
loss-weight config, not init) suggests it's sensitive to which specific
random draw the inflated channel's fresh weights got, not to any of the
swept variables. Not a bug to fix, just worth knowing before it's
mistaken for a real problem the first time someone actually assigns `kl`
a nonzero weight.

Still true: watch the first job's epoch 1 for
any *newly-weighted* component (does `train_loss` look sane, does the new
`train_<name>`/`val_<name>` column look sane, does it crash) --
this pull only exercised these four as passive monitors, none of them
has been given nonzero weight in an actual training objective yet.

**Everything else needed to change so an unlisted loss name doesn't
crash a run:** `windinet/training/vae_trainer.py`'s `loss_sum`/
`grad_norm_sum` (`train`) and `sums` (`_evaluate`) switched from
hardcoded 4-key dicts to `defaultdict(float)`, and every
`weights[name]` lookup that assembles the backward-pass total loss
switched to `weights.get(name, 0.0)` -- both were previously exact-key
assumptions that would `KeyError` the moment `reconstruction_losses()`
started returning more than four entries.

## Per-channel VRMSE, active by default (2026-08-08)

`val_vrmse` (the eval metric everything in this ledger is ranked on) was
always the 4-channels-mixed-together aggregate -- no way to tell whether a
given result's gain or loss was concentrated in one physical field (e.g.
"did pressure specifically get worse") versus spread evenly. `_evaluate`
now also computes VRMSE **per physical channel** and writes it to
metrics.csv as `val_vrmse_<channel_name>` (one column per entry of that
run's own `data.channel_order`, e.g. `val_vrmse_density`,
`val_vrmse_momentum_x`, `val_vrmse_momentum_y`, `val_vrmse_pressure`) --
named by physical field, not raw index, so the columns stay directly
comparable across channel-order-sweep configs that permute which field
sits in which slot.

**Mechanism:** `windinet.losses.vrms_per_channel` (`windinet/losses/vrms.py`)
is `vrms_loss`'s per-channel sibling -- same `sqrt(mse/var)` formula, but
reduces over every dim except batch *and* channel instead of collapsing
channels into the scalar too, returning a `[C]` tensor. `VaeTrainer.
vrmse_per_channel()` wraps it the same way the existing `vrmse()` wraps
`vrms_loss` (`@torch.no_grad()`, always a detached tensor). `_evaluate`
calls it once per batch alongside the existing aggregate `vrmse()` call,
accumulates into the same `defaultdict(float)` `sums` used for h2/pcc/vrms/
kl, keyed `f"vrmse_{name}"` for `name` in `cfg.data.channel_order` -- no
changes needed to the all-reduce, averaging, or `metrics_row`/CSV-writing
code, all of which already iterate `sums`/`val_metrics` dynamically (same
extensibility the H2/PCC/VRMS/KL addition above relies on).

**No config changes anywhere** -- this is always-on, not opt-in (unlike
H2/PCC/VRMS/KL as training losses, which stay opt-in via `loss_weighting.
weights`). Every run launched from this commit onward gets the 4 extra
columns automatically. **Does not backfill already-recorded runs** --
every metrics.csv committed before this change (baseline, encoder-LR
sweep, channel-order sweep, seed sweep, etc.) only has the aggregate
`val_vrmse` column; per-channel breakdowns are only available for runs
launched after `windinet/training/vae_trainer.py` picked this up. Directly
useful starting with the copy-init experiment (Open Question 10, above):
its hypothesis is specifically about momentum_x's own reconstruction
quality, which the aggregate `val_vrmse` could not previously isolate.

Verified with hand-built tensors (`vrms_per_channel` matches a manual
per-channel `sqrt(mse/var)` computation exactly; returns all-zero for
`pred == target`) and a dry-run of the `_evaluate` accumulation logic with
a fake `channel_order` -- produces the expected `val_vrmse_<name>` keys.

**Update (2026-08-09): also stale -- has run.** Four runs in this pull
have the `val_vrmse_<channel>` columns: `copyinit`, `logdensity`,
`rmseonly`, and the Q12 overfit diagnostic (all launched 2026-08-08
12:54 CEST or later). The five earlier jobs from the same day (`ep18`,
both LR-sweep grids, all launched 10:43 CEST) do **not** have these
columns -- they were submitted before the commit that added this feature
landed on the cluster, same kind of mid-pull cutover already documented
for the eval-parallelization fix below. Concretely useful in this pull:
the copy-init result above (Open Question 10) can report
`val_vrmse_momentum_x` directly, though without a per-channel baseline to
diff against (see that section's caveat); the log-density and RMSE-only
results likewise get per-channel numbers with the same baseline-gap
caveat, since job 521887 (the aggregate baseline) predates this feature
too.

## Canonical normalization-stats file, `normalization_stats_file` (2026-08-08)

**THE PROBLEM:** every VAE training config carried `data.channel_mean`/
`data.channel_std` as inline literals, copy-pasted into every single
config file. Channel-order-sweep configs additionally had to **hand-
reindex** the same four numbers per arm (e.g. `pr_my_mx`'s config manually
reordered them to match its `channel_order`) -- easy to get wrong silently
(a transposition would validate fine, just train on subtly wrong stats),
and there was no single place to update if the numbers themselves were
ever revised (e.g. after `scripts/compute_channel_stats.py`, added
earlier this pull, recomputes them from the real dataset instead of
`configs/finetune_vae/euler_mq_128_only_train.yaml`'s original hand
computation).

**FIX:** `VaeDataConfig` gained `normalization_stats_file: str | Path |
None`. When set, `channel_mean`/`channel_std` are loaded from that file's
`data_normalization_stats` block (same `{channel_name: {mean, std, ...}}`
shape `euler_mq_128_only_train.yaml` already used, and what
`compute_channel_stats.py --output` writes) and **reordered to match
`channel_order` automatically** -- the hand-reindexing step is gone.
`channel_mean`/`channel_std` became optional (`None` default); a
`model_validator` requires exactly one of "both set inline" or
"`normalization_stats_file` set," so there's no silent-precedence
ambiguity if a config somehow specified both.

**Migrated the 8 active (non-archived) VAE configs** to
`normalization_stats_file: "configs/finetune_vae/euler_mq_128_only_train.yaml"`
in place of their inline numbers (`finetune_vae_baseline.yaml`, the
whole-structure baseline + its `ep18`/decoder-LR-sweep/adapter-LR-sweep
variants, and the copy-init config) -- verified byte-identical
`channel_mean`/`channel_std` before/after the migration for all 8, so this
is a pure refactor, not a behavior change. Archived (`archive/done/`)
configs were left untouched on purpose -- they're closed, citable
snapshots, not live code to keep in sync.

**Sets up 256x256_ds support (Open Question 4) for free:** once
`scripts/compute_channel_stats.py` runs against `256x256_ds/train.h5` and
its `--output` JSON exists, any 256-resolution config just points
`normalization_stats_file` at that JSON instead of
`euler_mq_128_only_train.yaml` -- no code change, and no risk of
accidentally training a 256-resolution run on the 128-resolution dataset's
stats (previously plausible, since copy-pasting the wrong file's numbers
would validate fine, same 4-value list shape either way).

**`yaml.safe_load` reads both `.yaml` and `.json`** (JSON is a YAML
subset) -- the hand-written `euler_mq_128_only_train.yaml` and
`compute_channel_stats.py`'s JSON output work through the identical code
path with no format-specific branching.

Verified: `VaeDataConfig` unit-level (default channel_order, permuted
channel_order reproduces the previously-hand-transcribed `pr_my_mx`
numbers exactly, both-set/neither-set both raise, loading a
`compute_channel_stats.py`-shaped JSON file works), plus every checked-in
config across the whole repo still loads through `VaeTrainerConfig`
(same one pre-existing, unrelated failure as before -- a template
fragment that was never a valid trainer config to begin with).

**Full-dataset confirmation run, job 523501 (`wh_channel_stats`), launched
2026-08-08 13:44 CEST, wall-clock 1h08m** (this was missed in the first
write-up of this pull -- added 2026-08-09): `compute_channel_stats.py`
run against the **entire** `128x128_ds/train.h5` (4500 simulations,
7,446,528,000 values/channel, not a sample) with `--compare-to
configs/finetune_vae/euler_mq_128_only_train.yaml`.

**Result: every stat (mean/std/min/max) for all 4 physical channels plus
`log_density` matches the hand-computed reference to floating-point noise
(abs diff 1e-16 to 1e-18, 0.00% relative)** -- `euler_mq_128_only_train.
yaml`'s original hand computation was already exactly right, not an
approximation. This is a real confirmation, not a null result by default:
it retires the implicit risk flagged in "THE PROBLEM" above (that the
hand-computed numbers might silently differ from the true full-dataset
values) for the 128x128 dataset specifically.

**One gap surfaced by the diff table: `log_pressure` is computed by the
scan (mean -0.347238, std 0.692862, min -6.883690, max 3.208804) but is
`MISSING` from `euler_mq_128_only_train.yaml`.** `VaeDataConfig.
log_transform_channels` already validates `density`/`pressure` as the two
allowed entries (see "Log-density experiment" above), but only
`log_density`'s stats actually exist in the reference file today -- a
`log_transform_channels: ["pressure"]` or `["density", "pressure"]` run
would currently fail at config-load time (missing `log_pressure` key in
`normalization_stats_file`) rather than at training time. Not blocking
anything run so far (only `density` has been log-transformed, Open
Question 11), but worth noting for whoever extends that experiment to
pressure -- `euler_mq_128_only_train.yaml` needs the `log_pressure` block
added first (the scan's own output above has the numbers ready to copy
in).

**Not yet run against `256x256_ds/train.h5`** -- the "sets up 256x256_ds
support for free" note above is still aspirational; this run only
confirms the existing 128x128 stats, it doesn't produce the 256-resolution
file Open Question 4 would need.

## Where things live

| | |
|---|---|
| Rationale, hypotheses, verdicts | **this file** |
| Cluster job launchers | `jobs/lundquist/*.sbatch`, `jobs/sng_pvc/*.sbatch` (submit from repo root) |
| Portable entry points | `scripts/*.py` |
| Slurm logs | `log_finetuning_vae/lundquist/<jobname>_<jobid>.log`, `log_finetuning_vae/sng_pvc/<jobid>-<jobname>.{out,err}` |
| job -> config -> output_dir map | `log_finetuning_vae/{lundquist,sng_pvc}/INDEX.tsv` (appended by the sbatch scripts) |
| Per-run metrics, panels, resolved config | `finetune_vae_outputs/lundquist/<run>/{metrics,visualizations,training_config.yaml}` -- **tracked in git** since `f88eb09` |
| Checkpoints | `finetune_vae_outputs/lundquist/<run>/checkpoints/vae_shockwave_best.safetensors` -- **gitignored**, this machine is the only copy |
| Retired runs | git history, see Artifact retention under the Ledger |
| Closed configs/outputs (2026-08-06 cleanup, extended 2026-08-08, 2026-08-12) | `configs/finetune_vae/archive/done/` + `finetune_vae_outputs/{lundquist,sng_pvc}/archive/done/<run>/` for finished, citable results (capacity diagnostics, full head-vs-tail sweep, encoder-LR sweep, channel-order sweep v1+v2, copy-init, seed noise floor sweep, decoder/adapter LR sweep, log-density, RMSE-only, 18-epoch confirmation, loss-function retest batch); `.../archive/known-bad/` for runs whose numbers are flagged uninterpretable (the 8-sim diagnostic tier of the head-vs-tail sweep, capacity attempt 2, GradNorm) -- see each folder's `README.md` before reusing anything inside |

**Directory rename (2026-08-12):** the two per-cluster output trees moved
from top-level `finetune_vae_outputs_lundquist/` / `finetune_vae_outputs_
sng_pvc/` to `finetune_vae_outputs/lundquist/` / `finetune_vae_outputs/
sng_pvc/` (one shared parent, cluster as a subdirectory) -- pure
`git mv`, no content changes. `windinet/cluster_config.py`'s
`output_root` entries and the two hardcoded `LOCAL_DIR` mirror targets in
`jobs/sng_pvc/finetune_vae.sbatch`/`finetune_vae_debug.sbatch` were
updated to match. Historical citations elsewhere in this file that point
into a specific git commit (e.g. `git show f88eb09:finetune_vae_outputs_
lundquist/...`) were deliberately left unchanged -- that commit's tree
still has the pre-rename layout, so rewriting the citation would make it
wrong, not right.

## Protocol going forward

1. **One variable per run**, and state it explicitly against a *named* baseline
   row in this ledger.
2. **Write the row before launching**: hypothesis, the one changed variable, the
   read-out criterion, and the kill criterion.
3. **No result under 2x the seed noise floor counts as a result** until (1) in
   Open Questions is answered.
4. **Config naming**: `vae_<baseline-tag>_<change>.yaml`. Do not chain more than
   two change tags -- if the name needs a third, the run is no longer a
   single-variable experiment.
5. Rationale lives here, not in config headers. Config files carry the diff and
   a one-line pointer to the ledger row.
6. **Never re-run into an existing `output_dir`.** `clean_output_dir: true`
   wipes it at startup, before a single step -- that is how #3 was lost. Give
   every launch its own directory, and commit metrics + panels once the run
   finishes; the weights stay gitignored, so they are only ever on this disk.
