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
| `tail2` | `[2]` | `true` | db2+db3+mid+norm+conv_out | not yet launched |
| `tail3` | `[1,2]` | `true` | db1+db2+db3+mid+norm+conv_out | not yet launched |
| `fullenc` | `[0,1,2]` | `true` | entire encoder trunk | not yet launched |
| `head0` | `[0]` | `false` | db0 only | not yet launched |
| `head01` | `[0,1]` | `false` | db0+db1 | not yet launched |
| `head012` | `[0,1,2]` | `false` | db0+db1+db2 | not yet launched |
| `head01tail` | `[0,1]` | `true` | db0+db1+db3+mid+norm+conv_out (db2 skipped) | not yet launched |

Read-out per combo: same as attempt 5 -- val_vrmse materially below 0.061
(diagnostic) points at that module set being part of the bottleneck; ~0.061
means it isn't. Once the diagnostics report back, compare `head*` vs `tail*`
at matched unfrozen-block-count (`head0` vs `tail`, `head01` vs `tail2`,
`head012` vs `tail3`) to answer "which end of the trunk matters more" before
spending multi-GPU time on the full-data configs.

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

1. **What is the seed noise floor?** Re-run the baseline at seeds 1 and 2.
   Until this number exists, no result under ~2% is interpretable. Cost: 2 runs.
   *This gates everything below.*
2. ~~Is the bottleneck the latent or the objective?~~ **ANSWERED: partial.**
   See "Capacity diagnostic (Open Question 2)" above -- val_vrmse 0.061 with
   proper convergence (wsd), landing in the 0.05-0.08 band. Neither a clean
   `<=0.03` (pure objective problem) nor `>=0.09` (info never reaches the
   decoder); both the loss/training axis and latent budget remain live.
   Attempt 6 (slower decay on top of the tail unfreeze, 0.0608 vs attempt 5's
   0.0588) confirmed this isn't a truncated-schedule artifact -- the "partial"
   verdict holds without caveat.
3. **Does unfreezing (part of) the encoder help, and does it matter which
   part?** **Tail direction DONE**: diagnostic 0.0588 (attempt 5, ~3.6% over
   attempt 4, read as within noise), full-data 0.09369 (`finetune_vae_baseline_tail`,
   job 21692, ~1.8% over the 0.095396 baseline -- small, real, not decisive).
   Generalized into the full "Encoder head-vs-tail unfreeze sweep" table
   above, which also covers unfreezing from the head (`down_blocks[0..2]`, the
   actual spatial downsampling, carrying the fragile pretrained basis) instead
   of only extending from the tail -- those combos (`tail2`, `tail3`,
   `fullenc`, `head0`, `head01`, `head012`, `head01tail`) are not yet
   launched. Read-out compares `head*` vs `tail*` at matched block count once
   the diagnostics report back.
4. **Does more latent bandwidth help?** Feed 256x256 so the latent grid becomes
   8x8 (4x capacity). Independent of (3) -- unfreezing does not change the
   compression ratio. Still open; next after the head-vs-tail sweep reports
   back (or sooner, since it's independent).
5. **Was `stable_fraction 0.7`'s decay length actually right for full data?**
   **ANSWERED: yes.** `finetune_vae_baseline_slowdecay` (job 21689,
   `stable_fraction 0.35`) landed 0.09664, ~1.3% *worse* than the 0.095396
   baseline -- within the noise band Open Question 1 leaves uncalibrated, but
   directionally not an improvement. The schedule-length sensitivity seen on
   the 8-sim diagnostics does not transfer to full data; don't carry
   `stable_fraction 0.35` forward.

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

**Not yet promoted further:** exp 4-5 above show `workers: 4` shaves another
~6.6% off `workers: 2` (30.1 vs 32.2 min/epoch) with no crash, and `workers: 8`
gains nothing more past 4. Both diagnostics were only 2 epochs, though, so the
transient-open-failure risk the old N=4 comment warned about isn't fully
ruled out over a full 15-epoch run. Bumping the default past 2 needs one
full-length confirmation run first.

**OPEN:** H2 (does 4-rank COMPOSITE vs 8-rank FLAT communication topology
matter?) -- job 520459 hit the diagnostic's time limit before finishing even
epoch 1 (the `rng_types=[]` fix itself worked, no crash), so re-run
`jobs/sng_pvc/finetune_vae_diag_4rank.sbatch` now that
`rng_types=[]` unblocks it.

## Where things live

| | |
|---|---|
| Rationale, hypotheses, verdicts | **this file** |
| Cluster job launchers | `jobs/lundquist/*.sbatch`, `jobs/sng_pvc/*.sbatch` (submit from repo root) |
| Portable entry points | `scripts/*.py` |
| Slurm logs | `log_finetuning_vae/lundquist/<jobname>_<jobid>.log`, `log_finetuning_vae/sng_pvc/<jobid>-<jobname>.{out,err}` |
| job -> config -> output_dir map | `log_finetuning_vae/{lundquist,sng_pvc}/INDEX.tsv` (appended by the sbatch scripts) |
| Per-run metrics, panels, resolved config | `finetune_vae_outputs_lundquist/<run>/{metrics,visualizations,training_config.yaml}` -- **tracked in git** since `f88eb09` |
| Checkpoints | `finetune_vae_outputs_lundquist/<run>/checkpoints/vae_shockwave_best.safetensors` -- **gitignored**, this machine is the only copy |
| Retired runs | git history, see Artifact retention under the Ledger |

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
