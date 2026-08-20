# VAE Finetuning Experiment Ledger

Single source of truth for what has been run, what it showed, and what is
still open. Every new run gets a row here **before** it is launched
(hypothesis) and a verdict **after** it finishes.

**Reorganized 2026-08-17** to stay lean: this file now holds the *current*
state (baseline, established facts, open questions) plus a one-line verdict
per question. Two companion files carry the detail:

- **[infra.md](infra.md)** -- cluster/hardware/throughput knowledge:
  per-cluster production `batch_size`, the sng_pvc dataloader/eval/rendezvous
  investigation, where job logs/configs/outputs live on disk.
- **[EXPERIMENTS_archive.md](EXPERIMENTS_archive.md)** -- full historical
  write-ups: superseded baselines, the old pre-whole-structure-unfreeze
  ledger, and every open question's complete methodology/results/kill-
  criterion detail (including rejected approaches -- GradNorm, RMSE-only
  loss, `h2`, the `stable_end_epoch` schedule variant, cosine schedule,
  copy-init, log-density).

Nothing was deleted in the split -- everything below either stayed here
because it's current, or moved verbatim into one of the two files above.

## Table of contents

- [Current baseline](#current-baseline)
- [Fixed setup](#fixed-setup)
- [Weight provenance](#weight-provenance)
  - [DiT (stage 2, not yet run)](#dit-stage-2-not-yet-run)
- [Established](#established)
- [NOT established (previously treated as if it were)](#not-established-previously-treated-as-if-it-were)
- [Open questions, in priority order](#open-questions-in-priority-order)
- [Loss components, per-channel metrics, normalization stats -- quick reference](#loss-components-per-channel-metrics-normalization-stats----quick-reference)
- [Protocol going forward](#protocol-going-forward)

- **Task**: LTX-Video 0.9.x VAE adapted to 4-channel Euler CFD fields
  (density, momentum_x, momentum_y, pressure), inflate mode.
- **Metric**: `val_vrmse` on 675 held-out sims (fixed seed + randperm,
  identical across all runs).

## Current baseline

`configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml`
(128x128 resolution). Verified against the actual checked-in config, not
just the ledger prose below (the two had drifted -- see the zeros-init note):

| setting | value |
|---|---|
| Encoder unfreeze | whole trunk (decoder + `encoder.conv_in` + `down_blocks[0:3]` + tail bundle) -- hardcoded in `VaeTrainer._UNFROZEN_DOWN_BLOCKS`, not config-selectable (Open Question 3) |
| Encoder LR | same as decoder, 1.0x, no reduced-caution multiplier (Open Question 6, re-verified at 30 epochs 2026-08-15) |
| Epochs | 30, `wsd` schedule, `stable_fraction 0.7`, LR 5e-5 -> floor 1e-6, warmup 200 steps (Open Question 20) |
| `inflate_init` | `"zeros"` (switched from `"mean"` 2026-08-16, PhD-advisor parsimony rationale -- see note below) |
| Loss weights | `rmse 1.0, h1 50.0, ssim 0.15, mlw 0.0` -- **`kl` is NOT in this config's weights**, despite Open Questions 19/22 finding it "free" up to ~1e-6; it stays a sibling experiment (`finetune_vae_whole_structure_baseline_kl.yaml` and its sweep siblings), not yet merged into the default baseline |
| Seed | 42 |
| Resolution | 128x128 -- see the 256x256 note below for where that stands |

**Confirmed val_vrmse: 0.078662** (job 524322) -- but that confirmation ran
under `inflate_init: "mean"`, *before* the 2026-08-16 switch to `"zeros"`.
The zeros switch was applied directly to this file on parsimony grounds
(mean has no principled justification for the fresh channel; zeros is the
more defensible default absent a specific reason otherwise), citing Open
Question 21's separately-measured zeros-vs-mean deltas (roughly a wash on
aggregate val_vrmse, but a real ~4.3-4.7% *per-channel* cost to the fresh
pressure channel specifically) rather than a fresh confirmation run of this
exact combination (zeros + 1.0x encoder LR + 30 epochs). **No run has
confirmed a val_vrmse number for the file as it stands today** -- expect
something close to 0.0786-0.079 by extrapolation, not a hard result.

**256x256 resolution:** decided 2026-08-16 as the new default going forward,
superseding 128x128 (Open Question 4's -30.9% win was real but is pinned to
now-superseded 18-epoch/0.3x-encoder-LR/mean-init settings, not this
baseline). `finetune_vae_whole_structure_baseline_ep30_256res.yaml`
re-derives the comparison as a single-variable step off *this* baseline, but
**still has not completed anywhere as a standalone (no-anchor, no-KL) run**
-- its only lrz_ai launch (job 5750941, `BATCH_SIZE=16`) crashed on a CUDA
OOM in epoch 2, and no one has resubmitted it since the `BATCH_SIZE`
env-var fix (5ae27cf, 2026-08-17).

Status of the three sibling arms as of the 2026-08-20 pull (correcting the
previous update, which was written from a stale local view):
- **`_anchor`** (latent-distribution anchor-loss experiment, unrelated to
  the KL sweep) -- **completed all 30 epochs** via resume job 5752185
  (2026-08-18): **val_vrmse 0.054342** (`finetune_vae_outputs/lrz_ai/..._anchor/metrics/metrics.csv`),
  a real ~31% drop from the 128res baseline's confirmed 0.078662. This is
  the only 256res arm with a full result, but it conflates two changes
  (resolution *and* the anchor loss) against the 128res baseline, so it
  cannot yet be attributed to resolution alone.
- **`_kl1e6` / `_kl1e5`** -- **still incomplete, blocked on a resume-time
  OOM** that recurred even after the `BATCH_SIZE` fix: both have now had
  two resume attempts each (5752177->5752187 for kl1e6, 5752178->5752188
  for kl1e5), and every one died on a CUDA OOM before epoch 30. The
  committed `metrics.csv` for both only has an epoch-1 row, but the
  `.out` logs of the latest attempts show real progress not yet
  committed: kl1e5 reached epoch 3 (val_vrmse 0.0791) before dying mid
  epoch 4; kl1e6 reached epoch 3 (val_vrmse 0.1767) before the same. Root
  cause not yet diagnosed -- possibly a resume-specific memory spike
  (optimizer-state reload), since the same `BATCH_SIZE=16` ran anchor's
  resume to completion without incident.

**Planned runs, 2026-08-20 (written before launch, per protocol #2 below):**
three coordinated 256res experiments, decided with a **15h `--time`
request per arm** (raised from an initial 10h plan -- see the time-budget
note below for why). All fresh launches, `clean_output_dir: true`
(protocol #6 -- none resume the stuck kl1e6/kl1e5 attempts above; those
attempts' partial progress is abandoned in favor of a clean single-variable
run under current settings).

- **Experiment 1** -- ep30 256res KL sweep, all 4 arms already have configs
  (this is a clean relaunch of the existing sweep, not new files):
  `..._ep30_256res.yaml` (kl=0), `..._kl1e7.yaml`, `..._kl1e6.yaml`,
  `..._kl1e5.yaml`.
- **Experiment 2** -- new 20-epoch sibling sweep, 4 new configs
  (`..._ep20_256res.yaml` + `_kl1e7`/`_kl1e6`/`_kl1e5` siblings). Originally
  sized to fit inside 10h without a resume (~8-8.7h estimated at
  BATCH_SIZE=16's rate, per-epoch cost doesn't depend on epoch count --
  BATCH_SIZE=4 below is somewhat slower, no confirmed 256res number for it
  yet); now also requesting 15h along with the other two experiments for a
  uniform `--time`, well clear of the estimate either way.
- **Experiment 3** -- anchor-loss KL arm 2/2: new
  `..._ep30_256res_anchor_kl1e7.yaml` (kl=1e-7 on top of the anchor loss).
  Arm 1/2 (anchor, kl=0) is already complete -- see above.

**`BATCH_SIZE=4` uniformly across all 9 arms above** (2026-08-20 user
preference: memory headroom over speed, rather than mixing batch sizes
across siblings). Two of Experiment 1's arms (kl=0, kl1e7) confirmed OOM'd
at `BATCH_SIZE=16` in epoch 2 (jobs 5750941/5750942); the other two
(kl1e6, kl1e5) and the anchor kl=0 arm all ran clean at 16 through 29-30
epochs, so 16 was likely fine for them too -- but 4 is used everywhere
here for consistency and to avoid re-hitting the confirmed OOM on the two
arms that need it.

**Time-budget note:** anchor's real ~24-26 min/epoch x 30 epochs ~=
12-13h at BATCH_SIZE=16 -- and every arm in this plan runs at BATCH_SIZE=4
instead (see below), which is somewhat slower still (no confirmed 256res
number for 4 specifically). An initial 10h/arm plan was raised to **15h**
2026-08-20 specifically to give the five 30-epoch arms (Experiments 1 and
3) real margin over that ~12-13h-at-best estimate instead of banking on a
resume; Experiment 2's 20-epoch arms (~8-8.7h at BATCH_SIZE=16's pace) get
the same 15h for a uniform `--time` across all 9 arms, not because they
need it. **This is a resolution effect, not a KL effect** -- Open Question
4's clean resolution-only comparison (sng_pvc, no KL/anchor) already
measured 256res at ~2.2x 128res's per-epoch cost, and the anchor run's own
~24-26 min/epoch carries no KL term either, so the per-epoch cost driving
this budget is inherent to 256res itself, not something a different KL
weight would change.

Read-out for Experiment 1: val_vrmse vs. the 128res baseline (0.078662)
and the anchor+256res combined result (0.054342) -- isolates the
resolution-only effect. Kill criterion: same OOM-in-epoch-2 pattern as job
5750941 on the kl=0/kl1e7 arms (already mitigated by `BATCH_SIZE=4` above,
so a recurrence would mean 4 isn't safe either and needs lowering to 2).

## Fixed setup

Architecture/data facts that don't depend on which sweep is running. Trainable
split reflects the *current* whole-structure-unfreeze baseline (Open Question
3) -- the frozen-trunk numbers from the original 15-epoch baseline are in
`EXPERIMENTS_archive.md`'s old Ledger section, not repeated here since they
describe a superseded config.

| | |
|---|---|
| Base model | LTXV 2B, `spatial_compression_ratio=32`, `temporal=8`, `latent_channels=128` |
| Latent grid (128x128 input) | 128x128 field -> **4x4** latents (each cell covers 32x32 px), ~250:1 |
| Latent grid (256x256 input) | 256x256 field -> **8x8** latents, same ~250:1 ratio, 4x more latent cells |
| Trainable (whole-structure baseline) | decoder + `encoder.conv_in` + `encoder.down_blocks[0:3]` + tail (`down_blocks[-1]`+`mid_block`+`norm_out`+`conv_out`) -- **~1.25B params** (approximate, see `EXPERIMENTS_archive.md`'s GradNorm section) |
| Frozen | nothing in the encoder under the current baseline -- the whole trunk unfreezes (this is *not* true for `adapter.mode: "adapter"` configs, a different/retired architecture) |
| Effective batch | 32 (per-cluster `batch_size`/accum split normalized by the sbatch/job launchers -- see `infra.md`) |
| Total opt steps (current 30-epoch baseline) | 3600 (120/epoch x 30), warmup 200 |
| Data | 3825 train / 675 eval |
| Seed | 42 (**every run** -- no repeats outside the dedicated seed-noise sweep, Open Question 1) |

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

## Established

(`#N` references below are to the old 10-run ledger, now in
`EXPERIMENTS_archive.md`'s "Ledger" section.)

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
- **Seed noise floor is ~1.1%.** Spread across seeds 42/1/2 on the frozen-trunk
  baseline was 0.09411-0.09519. Use ~1.1% (protocol point 3's "2x" bar is
  ~2.2%) wherever this document says a gap needs to clear noise. Full sweep:
  `EXPERIMENTS_archive.md`'s "Seed noise floor sweep" section.

## NOT established (previously treated as if it were)

(`#N` references below are to the old 10-run ledger, now in
`EXPERIMENTS_archive.md`'s "Ledger" section.)

- **`mean` vs `zeros` init has never been cleanly isolated** on this old
  ledger. The only comparable pair is #5 (zeros, 0.101276) vs #6 (mean,
  0.101768) -- zeros is *slightly better*, and #5 additionally differs by
  the tail3x schedule. The claim "mean beat zeros/random on every prior
  ablation", repeated in five config headers, is supported only for
  `random`. (Superseded by a real, later, clean comparison -- Open
  Question 21 -- which the current baseline's `inflate_init: "zeros"`
  choice is based on; see [Current baseline](#current-baseline).)
- **`wsd` vs `cosine` is confounded with epoch count** on this old ledger.
  The 6% gain from the 10-epoch group to the 15-epoch wsd group changed
  schedule *and* 10->15 epochs simultaneously. Neither is attributable.
  (Also superseded: `cosine` was later tested cleanly at matched epoch
  count and rejected, see Open Question 20's write-up.)
- **Run-to-run variance was unmeasured** at the time this was written. All
  ten old-ledger runs use seed 42, n=1. (Now measured -- ~1.1%, see
  Established above.)

## Open questions, in priority order

One line per question: the verdict, then a link into `EXPERIMENTS_archive.md`
(or `infra.md` for the one hardware question) for the full methodology,
results table, and kill-criterion detail. Items 5/6/7/9 were answered inline
in the archive's "Open questions 1-22" section rather than getting their own
subsection -- linked to that section generically, not a specific anchor.

1. Seed noise floor -- **ANSWERED ~1.1%.** [Seed noise floor sweep](EXPERIMENTS_archive.md#seed-noise-floor-sweep-open-question-1-configs-written-2026-08-06).
2. Latent vs. objective bottleneck -- **ANSWERED: partial** (0.05-0.08 band, neither a clean objective-only nor latent-only story). [Capacity diagnostic](EXPERIMENTS_archive.md#capacity-diagnostic-open-question-2----resolved-partial).
3. Does unfreezing (part of) the encoder help -- **ANSWERED: whole trunk wins**, promoted to the baseline. [Head-vs-tail unfreeze sweep](EXPERIMENTS_archive.md#encoder-head-vs-tail-unfreeze-sweep-open-question-3-generalized).
4. Does more latent bandwidth help -- **ANSWERED: yes, largest single-variable gain in the whole ledger** (-30.9% at the time, since superseded by baseline moves -- see [Current baseline](#current-baseline)). [256x256 resolution](EXPERIMENTS_archive.md#256x256-resolution-open-question-4-resolved-2026-08-11).
5. Was `stable_fraction 0.7`'s decay length right for full data -- **ANSWERED: yes.** Full detail: [Open questions 1-22](EXPERIMENTS_archive.md#open-questions-1-22-full-write-ups), item 5.
6. Encoder-LR sweep on the whole-structure baseline -- **ANSWERED: 0.3x won the original sweep**, later retired to 1.0x (see [Current baseline](#current-baseline)). Full detail: [Open questions 1-22](EXPERIMENTS_archive.md#open-questions-1-22-full-write-ups), item 6.
7. Was 15 epochs enough for the whole-structure baseline -- **ANSWERED: no, real gain from more epochs** (this is the thread that eventually pushed epochs to 30, see Q20). Full detail: [Open questions 1-22](EXPERIMENTS_archive.md#open-questions-1-22-full-write-ups), item 7.
8. Does channel order matter -- **ANSWERED: yes**, tracks which field lands in the fresh (index-3) slot; default order stays adopted. [Channel-order sweep](EXPERIMENTS_archive.md#channel-order-sweep-open-question-8-new-2026-08-06), [v2 + copy-init](EXPERIMENTS_archive.md#channel-order-sweep-v2-copy-init-extension-open-questions-810-revisited-resolved-2026-08-12).
9. Decoder LR / conv_in adapter LR multiplier -- **ANSWERED: baseline values already near-optimal**, no change adopted. Full detail: [Open questions 1-22](EXPERIMENTS_archive.md#open-questions-1-22-full-write-ups), item 9.
10. Does copy-init from a paired sibling channel beat mean-init -- **ANSWERED: no, null result** (5 of 6 arms; the 6th's win looks like avoiding a bad mean-init arrangement, not a general copy-init effect). [Copy-init](EXPERIMENTS_archive.md#copy-init-for-a-physically-paired-momentum-channel-open-question-10-new-2026-08-08), [v2 revisit](EXPERIMENTS_archive.md#channel-order-sweep-v2-copy-init-extension-open-questions-810-revisited-resolved-2026-08-12).
11. Does log-compressing density help -- **ANSWERED: borderline, inside the noise floor**, not adopted. [Log-density experiment](EXPERIMENTS_archive.md#log-density-experiment-open-question-11-new-2026-08-08).
12. 8-sim memorization ceiling under whole-structure unfreeze -- **ANSWERED: ceiling doesn't move**, whole-trunk unfreeze's gain isn't from raw memorization capacity. [Capacity diagnostic re-run](EXPERIMENTS_archive.md#capacity-diagnostic-re-run-under-whole-structure-unfreeze-open-question-12-new-2026-08-08).
13. RMSE-only loss ablation -- **ANSWERED: worse, h1/ssim are net-positive regularizers**, not just gradient dilution. [RMSE-only loss ablation](EXPERIMENTS_archive.md#rmse-only-loss-ablation-open-question-13-new-2026-08-08).
14. Does `h2` (curvature loss) help on top of `h1` -- **ANSWERED: no, slightly worse**, regression concentrated in pressure. Not adopted. [H2 loss term](EXPERIMENTS_archive.md#h2-loss-term-open-question-14-resolved-2026-08-12).
15. GradNorm vs. hand-tuned fixed weights -- **ANSWERED: no, never converges (2-epoch oscillation) + ~5x cost.** [GradNorm loss weighting](EXPERIMENTS_archive.md#gradnorm-loss-weighting-open-question-15-resolved-2026-08-12).
16. h1 weight 50->100 retest -- **ANSWERED: no effect**, stays at 50. [Existing-loss weight retests](EXPERIMENTS_archive.md#existing-loss-weight-retests-h1x2-ssimx2-mlw-open-questions-161718-resolved-2026-08-12).
17. ssim weight 0.15->0.3 -- **ANSWERED: no effect**, stays at 0.15. Same section as Q16.
18. mlw weight 0.0->1e-4 retest -- **ANSWERED: still net-negative**, stays at 0.0. Same section as Q16.
19. KL divergence weight on vs. off -- **ANSWERED: regularizes (~359x lower val_kl) at no reconstruction cost.** Promoted candidate for the DiT stage; see [Current baseline](#current-baseline) for whether/where it's actually active. [KL divergence weight](EXPERIMENTS_archive.md#kl-divergence-weight-open-question-19-resolved-2026-08-12).
20. Epoch budget 18->30 + schedule shape -- **ANSWERED: epoch budget alone drove the gain** (-6.09%); neither slower decay nor a lower LR floor added anything on top. Adopted (epochs=30 in the current baseline). [Epoch budget / schedule-shape sweep](EXPERIMENTS_archive.md#epoch-budget-schedule-shape-sweep-open-question-20-resolved-2026-08-14) -- also covers the rejected cosine-schedule and incomplete-40-epoch extensions.
21. Fresh-channel init, zeros vs. mean -- **ANSWERED (quantitative half): zeros costs the fresh (pressure) channel ~4.3-4.7%, aggregate val_vrmse is a wash.** Adopted anyway on parsimony grounds (mean has no principled justification) -- see [Current baseline](#current-baseline). Qualitative half (does the cross-channel blending artifact go away) never checked. [Fresh-channel init](EXPERIMENTS_archive.md#fresh-channel-init-zeros-vs-mean-open-question-21-resolved-2026-08-14).
22. KL divergence weight sweep, where does it saturate -- **ANSWERED: nowhere in the range tested** (1e-8 through 1e-5, val_vrmse stayed flat). [KL divergence weight sweep](EXPERIMENTS_archive.md#kl-divergence-weight-sweep-open-question-22-resolved-2026-08-14).
23. Does per-GPU micro-batch size change results or just wall-clock/feasibility -- **CLOSED: larger batch_size is faster, no real val_vrmse cost; production batch_size set per cluster.** This is a hardware question, moved to [infra.md](infra.md#per-gpu-batch_size-vs-wall-clockoom-open-question-23-closed-2026-08-16).

## Loss components, per-channel metrics, normalization stats -- quick reference

Full development/verification history for all three is in
`EXPERIMENTS_archive.md`'s "Loss-component / per-channel-VRMSE /
normalization-stats-file feature history" section.

- **Loss components** (`windinet/losses/`): `rmse`/`h1`/`ssim`/`mlw` are the
  always-weighted defaults. `h2` (curvature), `pcc` (correlation), `vrms`
  (variance-normalized RMSE, same formula as the `val_vrmse` eval metric),
  and `kl` (VAE ELBO regularizer) are computed and logged every run
  regardless of weight (`train_<name>`/`val_<name>` columns in
  `metrics.csv`) -- opt into any of them by adding a nonzero weight under
  `loss_weighting.weights`. Only `kl` needs anything beyond `(pred,
  target)` (the encoder's raw posterior mean/logvar); the others slot
  straight in. Enable one at a time per protocol point 1.
- **Per-channel VRMSE**: `val_vrmse_<channel_name>` columns (one per entry
  of that run's `data.channel_order`) are written automatically, no config
  needed -- active on every run since 2026-08-08. Runs launched before that
  date only have the aggregate `val_vrmse` column, no per-channel breakdown
  to diff against.
- **`normalization_stats_file`**: point `data.normalization_stats_file` at
  a YAML/JSON with a `data_normalization_stats` block (same shape
  `scripts/compute_channel_stats.py --output` writes) instead of inlining
  `channel_mean`/`channel_std` -- values get reordered to match
  `channel_order` automatically. 128x128's stats
  (`configs/finetune_vae/euler_mq_128_only_train.yaml`) are confirmed
  against a full-dataset scan (not just the original hand computation) to
  1e-16 relative error; 256x256's (`euler_mq_256_only_train.yaml`) have
  not had the same independent confirmation.

## Protocol going forward

1. **One variable per run**, and state it explicitly against a *named* baseline
   row in this ledger.
2. **Write the row before launching**: hypothesis, the one changed variable, the
   read-out criterion, and the kill criterion.
3. **No result under 2x the seed noise floor counts as a result** (~2.2%,
   Established above).
4. **Config naming**: `vae_<baseline-tag>_<change>.yaml`. Do not chain more than
   two change tags -- if the name needs a third, the run is no longer a
   single-variable experiment.
5. **Rationale lives in `EXPERIMENTS_archive.md` (full write-up) with a
   verdict line here, not in config headers.** Config files carry the diff
   and a one-line pointer to the ledger row. Hardware/cluster-throughput
   findings go in `infra.md` instead -- see the split explained at the top
   of this file.
6. **Never re-run into an existing `output_dir`.** `clean_output_dir: true`
   wipes it at startup, before a single step -- that is how #3 was lost. Give
   every launch its own directory, and commit metrics + panels once the run
   finishes; the weights stay gitignored, so they are only ever on this disk.
