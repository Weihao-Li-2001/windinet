# WinDiNet Master's Thesis — Prep Material

Compiled from `EXPERIMENTS.md`, `EXPERIMENTS_archive.md`, `infra.md`,
`latent_space_shift_measure.md`, `README.md`, and configs in this repo, as of
2026-09-12. This is raw material/outline for drafting, not finished prose —
hand this to ChatGPT with the instruction to expand each section into full
academic writing, and flag anywhere a number needs re-confirming against the
live `EXPERIMENTS.md` before the thesis is finalized (DiT stage is still
actively changing).

**Status flag for the writer:** VAE stage (Stage 1) is mature/largely closed
out — dozens of controlled ablations, an established noise floor, a settled
baseline. DiT stage (Stage 2) is *actively in progress* — real eval numbers
exist (best: SDS arm, 5.9x the VAE-only reconstruction floor) but no
"final" DiT baseline has been chosen yet, and the leading open problem
(latent-space shift under KL regularization) is diagnosed but not fully
solved. Frame the thesis's Results/Conclusion honestly around this: VAE-stage
findings are presented as settled results, DiT-stage findings are presented
as "promising preliminary results establishing a clear best-known recipe and
an open research question," not as a finished pipeline.

---

## 0. Working title candidates

- "WinDiNet: Repurposing a Pretrained Video Diffusion Transformer as a
  Differentiable Surrogate for Compressible Euler-Flow Simulation"
- "From Natural Video to Shock Waves: Adapting a Latent Video Diffusion
  Model (LTX-Video) for Compressible CFD Surrogate Modeling"
- "Transfer Learning from Video Diffusion Transformers to Physical
  Simulation: A Case Study on Compressible Euler Flow"

## 1. Abstract — key points to include

- Problem: CFD simulation of compressible flow (Euler equations, with
  shocks) is expensive; learned surrogates that are fast and differentiable
  are valuable for design/optimization loops.
- Idea: repurpose a *pretrained natural-video* diffusion transformer
  (LTX-Video, a latent video diffusion model originally for text/image-to-
  video generation) as a CFD surrogate, exploiting the fact that video
  diffusion models already learn strong spatiotemporal priors. This project
  ("WinDiNet") is itself a fork/adaptation of an earlier urban-wind-flow
  model (2-channel u/v velocity) — this thesis adapts it further to
  4-channel compressible Euler fields with shocks.
- Method: two-stage pipeline. (1) Fine-tune the pretrained VAE's
  encoder/decoder to reconstruct 4-channel CFD fields (density,
  momentum_x, momentum_y, pressure) instead of RGB video, via a channel-
  inflation adapter. (2) Fine-tune the pretrained diffusion transformer
  (DiT) to roll out simulation trajectories in the resulting latent space,
  conditioned on the initial condition and a scalar physical parameter
  (gamma, the heat-capacity ratio) via Fourier-feature embeddings.
- Key results: VAE reconstruction improved substantially (val_vrmse
  0.0947 → ~0.054, roughly a 43% reduction) through architecture (full
  encoder unfreeze vs. adapter-only), resolution (128→256px), schedule and
  init choices, established over a large, systematically logged ablation
  study with a measured ~1.1% seed-noise floor. For the DiT stage, a novel
  self-distillation regularizer (SDS — distilling the frozen, pretrained
  DiT's own flow-matching velocity into the VAE encoder during VAE
  fine-tuning) produced the best rollout accuracy of any arm tested (5.9x
  the pure-reconstruction floor), outperforming both an unregularized
  baseline and a family of KL-divergence-regularized variants, which
  instead made the DiT's job monotonically *harder* with increasing KL
  strength — evidence for a "latent shift" phenomenon where the VAE's
  fine-tuned latent space drifts away from the distribution the pretrained
  DiT was trained to denoise.
- State the thesis's scope honestly: VAE stage is a complete, closed-out
  ablation study; DiT stage is presented as ongoing work with a clear
  current best recipe and a diagnosed, partially-open research question.

## 2. Introduction (1–2 pages)

Points to hit:
1. Motivation: CFD is a core tool in engineering (aerospace, automotive,
   energy) but simulating compressible flow with shocks is computationally
   expensive; ML surrogates promise orders-of-magnitude speedups for
   design-space exploration, real-time control, or embedding inside
   optimization loops (differentiability).
2. Why *video* diffusion models specifically: a CFD trajectory is
   structurally a video (a sequence of 2D fields evolving over time) —
   this is exactly the object video generation models are trained to
   model. Pretrained video diffusion transformers already encode strong
   priors over spatially coherent, temporally consistent dynamics; the
   thesis's central hypothesis is that this prior transfers to physical
   simulation with much less training data/compute than training a
   simulator-specific model from scratch.
3. Specific approach: adapt LTX-Video (a fast, latent, DiT-based video
   generation model built by Lightricks) — replace its natural-image VAE's
   3-channel (RGB) I/O with 4 physical channels, replace text conditioning
   with physical scalar conditioning, and fine-tune on a compressible Euler
   CFD dataset (`euler_mq`, quadrilateral-mesh Euler solves with shocks,
   parameterized by gamma) at multiple resolutions (128/256/512 px, native
   sim resolution 512x512).
4. Contributions of this thesis (adjust once DiT results are finalized):
   - A systematic, controlled ablation study of VAE fine-tuning choices
     for out-of-domain (scientific, non-RGB) channel adaptation of a
     pretrained video VAE — encoder unfreezing strategy, channel-init
     scheme, loss-term composition, KL regularization, resolution scaling,
     LR schedule — with an explicit measured noise floor so results are
     statistically interpretable, not anecdotal.
   - Diagnosis and partial solution of a **latent-space-shift** failure
     mode: fine-tuning the VAE encoder for reconstruction quality alone
     can move the latent distribution away from what the pretrained DiT
     expects, quantifiably increasing the downstream DiT's rollout error;
     this thesis proposes and validates a self-distillation regularizer
     (SDS) that fixes this without sacrificing reconstruction quality.
   - An engineering/infrastructure contribution: a reproducible, documented
     multi-cluster (3 HPC systems, heterogeneous GPU/accelerator hardware)
     training pipeline for this class of model, with lessons for anyone
     adapting large pretrained video models to a new scientific domain
     (mixed-precision checkpoint-save memory spikes, DDP + gradient-
     checkpointing ordering bugs, offline-cluster HF cache priming, etc. —
     use selectively, only if the thesis wants an infra/engineering
     subsection).
5. Thesis structure roadmap (one paragraph).

## 3. Literature review (5–6 pages) — topics and what to say about each

Organize into ~4 thematic clusters. (Concrete citations should be added by
the user/ChatGPT from a proper literature search — flagged with `[CITE]`
below for facts that need a specific paper.)

### 3.1 Learned surrogates for PDE / CFD simulation
- Classical neural PDE solvers: Fourier Neural Operator (FNO) `[CITE]`,
  DeepONet `[CITE]` — operator-learning approaches that map initial/
  boundary conditions to solution fields directly, resolution-independent.
- Graph-based simulators: MeshGraphNets `[CITE]`, GNS `[CITE]` — mesh-
  native, good for irregular geometry, autoregressive rollout.
- Physics-informed neural networks (PINNs) `[CITE]` — different paradigm
  (loss-constrained by PDE residuals rather than data-driven), worth a
  short contrast paragraph on why this thesis takes the data-driven,
  pretrained-prior route instead.
- Position this thesis: rather than a bespoke architecture trained from
  scratch on simulation data, it repurposes a *generic, pretrained*
  generative video model — closer in spirit to "foundation model for
  physics" work than to purpose-built operator learners. Autoregressive
  rollout error accumulation (a known issue for both graph simulators and
  this thesis's own DiT rollout) is a natural point of comparison.

### 3.2 Diffusion models, latent diffusion, and video diffusion
- Denoising diffusion probabilistic models (DDPM) `[CITE Ho et al. 2020]`,
  score-based generative models `[CITE Song et al.]`.
- Latent diffusion (Stable Diffusion, Rombach et al.) `[CITE]` — the core
  trick this project inherits: diffuse in a compressed VAE latent space
  rather than pixel space, for tractability. Explain the VAE + diffusion
  transformer (DiT) split explicitly, since it's the exact architecture
  this thesis's method reuses.
- Diffusion Transformers (DiT, Peebles & Xie) `[CITE]` — the transformer-
  based denoiser architecture (replacing U-Net), relevant since LTX-Video's
  denoiser is a DiT variant.
- Video diffusion models specifically: Video Diffusion Models `[CITE Ho et
  al.]`, and more directly, **LTX-Video** (Lightricks) `[CITE HuggingFace/
  arXiv — verify exact citation, the arXiv ID in this repo's README,
  2603.21210, looks anomalous and should be independently verified before
  citing]` — a fast, real-time-oriented latent video DiT; describe its
  spatial+temporal compression (this project uses its
  `spatial_compression_ratio=32`, `temporal=8`, 128 latent channels VAE)
  and flow-matching training objective (`v = eps - z0`, used verbatim in
  this project's latent-shift diagnostics).
- Rectified flow / flow matching as the diffusion training objective
  `[CITE Liu et al. / Lipman et al.]` — relevant since LTX-Video (and thus
  this project's DiT) is trained with a flow-matching velocity objective,
  not classical epsilon-prediction DDPM.

### 3.3 Transfer learning from pretrained generative vision/video models to non-natural-image domains
- General transfer-learning-for-science literature: reusing ImageNet/
  vision-pretrained backbones for scientific imaging (e.g. medical
  imaging, remote sensing) `[CITE]` — establishes the general pattern this
  thesis instance of.
- More specifically: prior work repurposing pretrained *diffusion* models
  for scientific data generation/surrogate tasks `[CITE — search for
  "diffusion model scientific surrogate", "diffusion PDE surrogate",
  "pretrained video diffusion physics"]`.
- **Direct predecessor**: the original WinDiNet urban-wind-flow model
  (2-channel u/v velocity, 256x256, this project's own upstream base) —
  cite as the direct prior work this thesis builds on and generalizes
  (from incompressible 2-channel wind flow to compressible 4-channel Euler
  flow with shocks). `[CITE the arXiv entry in this repo's README —
  2603.21210 — after independently verifying it resolves to a real paper;
  do not cite an unverified ID]`.
- Channel-inflation / architecture-surgery technique: growing a pretrained
  conv stem from 3→N channels by replicating/averaging/zero-padding
  pretrained weights is a known pattern (e.g. inflating 2D pretrained
  weights to 3D video models, "Inflated 3D ConvNet"/I3D-style channel
  inflation `[CITE Carreira & Zisserman, I3D]` as the conceptual ancestor,
  even though this project's channel-inflation is channel-count not
  temporal-dimension). Useful to frame the "mean-init vs. zero-init vs.
  random-init" ablation (this thesis: random destroys the run, +72% worse;
  zero and mean are close, with a real principled-parsimony argument for
  zero) against this literature.

### 3.4 Representation/latent-space stability under fine-tuning
- Catastrophic forgetting / representation drift under fine-tuning
  `[CITE — general continual-learning literature]` — frames the
  "latent-shift" problem this thesis diagnoses (Section on latent-space
  shift below) as an instance of a known general phenomenon: fine-tuning a
  pretrained representation for a new objective (reconstruction fidelity)
  can silently degrade its usefulness for a different downstream consumer
  (the pretrained DiT) that was never in the fine-tuning loop.
- Knowledge distillation `[CITE Hinton et al.]` — directly relevant, since
  this thesis's own fix (the SDS regularizer) is a distillation loss: it
  distills the *frozen, pretrained* DiT's flow-matching velocity
  prediction back into the VAE encoder during VAE fine-tuning, so the
  encoder is directly optimized for "the thing the downstream model needs"
  rather than only for pixel-level reconstruction.
- Representation similarity metrics: Centered Kernel Alignment (CKA)
  `[CITE Kornblith et al. 2019]` — used directly in this thesis's
  diagnostic methodology (`latent_space_shift_measure.md`) to measure how
  much the fine-tuned latent space has moved, in a way that's invariant to
  benign rotation/scaling the DiT doesn't care about.

## 4. Methodology

### 4.1 Task and dataset
- Task: given an initial CFD field and a scalar physical parameter
  (gamma, the ratio of specific heats), predict the full spatiotemporal
  rollout of a 4-channel compressible Euler flow field (density,
  momentum_x, momentum_y, pressure), including shock formation/
  propagation.
- Dataset: `euler_mq` (HuggingFace `rha6696/euler_mq`) — quadrilateral-
  mesh compressible Euler CFD simulations. Native simulation resolution is
  512x512; the dataset card also provides pre-downsampled 256x256 and
  128x128 versions. This project has trained VAE baselines at all three
  resolutions (128/256 mature, 512 attempted, not yet confirmed
  complete). 3825 training simulations / 675 held-out evaluation
  simulations, fixed split (seed-based permutation) across every run for
  comparability.
- Report the exact channel set and channel order (density, momentum_x,
  momentum_y, pressure) and note that channel order was itself found to
  matter (Open Question 8 — see Results), since whichever field lands in
  the newly-added (non-pretrained) channel slot is treated differently by
  the pretrained/fine-tuned weights.

### 4.2 Base model and architecture adaptation
- Base model: LTX-Video 2B (`Lightricks/LTX-Video`), a latent video
  diffusion transformer originally trained on natural RGB video, with a
  spatial compression ratio of 32x, temporal compression 8x, and a 128-
  channel VAE latent space. A 128x128-pixel input field compresses to a
  4x4 latent grid; a 256x256 field to 8x8 — both at the same ~250:1
  compression ratio the pretrained model was designed around.
- Two components, two separate pretrained weight sources (worth a
  diagram): the VAE comes from the `LTX-Video-0.9.5` diffusers checkpoint;
  the DiT/transformer comes from the different single-file 0.9.6-dev
  checkpoint. Verified by tensor-level hash comparison that the two
  checkpoints' *encoders* are byte-identical (only the decoder's final
  up-block/`conv_out` differ) — so this mixed-provenance setup introduces
  no latent-space mismatch, since the encoder (what actually produces the
  DiT's training target) is unaffected.
- **VAE channel adapter**: the pretrained encoder's first conv layer and
  decoder's last conv layer are natively 3-channel (RGB). "Inflation"
  grows these to 4 input/output channels by adding a new weight slice,
  initialized either as the mean of the existing 3 pretrained slices
  (`mean` init) or as zero (`zero` init) — ablated in Results. Random
  init was tested as a negative control and found catastrophic (+72%
  worse val_vrmse), confirming the pretrained patchify basis must be
  preserved rather than relearned.
- **Scalar conditioning**: LTX-Video's native text-conditioning pathway is
  replaced with Fourier-feature encoding of the scalar physical parameter
  (gamma) — enables physically parameterized generation instead of
  prompt-based generation. `windinet/scalar_embeddings.py`.
- **Encoder unfreeze strategy**: an ablated design choice, not fixed a
  priori — tested freezing only a lightweight "adapter" head vs.
  unfreezing progressively more of the encoder trunk vs. unfreezing the
  entire encoder (decoder + `encoder.conv_in` + all `down_blocks` + the
  tail bundle, ~1.25B trainable params). Whole-trunk unfreeze won clearly
  and became the baseline (Results, Open Question 3).

### 4.3 Two-stage training pipeline
1. **Stage 1 — VAE fine-tuning**: fine-tune the (channel-adapted) VAE
   encoder+decoder to reconstruct the 4-channel CFD fields, using a
   weighted combination of reconstruction and physics-motivated losses
   (below). Produces a VAE checkpoint whose encoder defines the latent
   space the DiT will operate in.
2. **Stage 2 — Dataset preprocessing**: encode the full training set
   through the fine-tuned VAE's encoder to precompute latents (never
   re-run online during DiT training) — `scripts/preprocess_dataset.py`.
   Records exact VAE-checkpoint provenance alongside the latents so a
   later DiT/inference run can verify it's decoding with the matching
   decoder (`latent_provenance.json`, `verify_latent_space`).
3. **Stage 3 — DiT fine-tuning**: fine-tune the pretrained diffusion
   transformer, via a flow-matching objective (`v = eps - z0`), to
   generate/roll out plausible latent trajectories conditioned on the
   initial condition and the scalar gamma embedding, using only the
   precomputed Stage-1 latents (the DiT trainer does construct an
   internal VAE object per the base LTX-Video code path, but it is
   confirmed frozen/unused dead weight in this setup, not a live
   component — worth a footnote as a "was this a bug" investigation the
   thesis resolved).
- Note explicitly: this decoupling (frozen precomputed latents for Stage
  2/3) means Stage 1's quality is a hard ceiling on Stage 2 — if the VAE's
  latent space is not something the (frozen, pretrained-then-fine-tuned)
  DiT can easily denoise, no amount of DiT training compute fixes that.
  This is exactly the mechanism behind the thesis's central "latent
  shift" finding (Results/Discussion).

### 4.4 Loss functions
List each with the one-line role, drawn from `windinet/losses/`:
- **RMSE** (`rmse.py`) — base pixel-level reconstruction loss, weight 1.0
  in every config.
- **H1 semi-norm** (`h1_semi_norm.py`) — gradient-matching loss
  (penalizes spatial-derivative mismatch, not just raw value mismatch);
  weight 50.0 in the baseline — a substantial, physically-motivated
  regularizer (shock/gradient sharpness matters in CFD). Ablated (weight
  25→50, 50→100) — 50 is where it lands, no further gain from doubling.
- **H2 semi-norm** (`h2_semi_norm.py`, curvature loss) — tested as an
  addition on top of H1; rejected (slightly worse, regression concentrated
  in the pressure channel).
- **SSIM** (`ssim.py`) — structural-similarity term, weight 0.15;
  doubling to 0.3 showed no effect.
- **MLW** (`mlw.py`, multi-level wavelet(?) — confirm exact meaning from
  code before writing) — tested at nonzero weight (1e-4), found net-
  negative both as a fixed weight and under earlier adaptive weighting;
  stays at 0.0 in the baseline.
- **PCC** (`pcc.py`, Pearson correlation) and **VRMS** (`vrms.py`,
  variance-normalized RMSE — same formula as the eval metric `val_vrmse`)
  — computed/logged every run for diagnostics regardless of weight, not
  optimized against directly in the baseline.
- **KL divergence** (`kl_divergence.py`) — the VAE's standard ELBO
  regularizer; ablated as its own axis (Open Questions 19/22): a wide
  sweep (1e-8 to 1e-5) found reconstruction quality essentially flat
  ("free" regularization at the VAE-reconstruction level) — but Stage-2
  DiT evaluation *later revealed* this same KL term makes the downstream
  DiT's job monotonically harder with increasing weight (the central
  "latent shift" finding — flag this contrast explicitly in Discussion:
  a regularizer that looks free/harmless by the Stage-1 metric alone can
  be actively harmful once you look at the Stage-2 consumer).
- **Latent anchor loss** (`latent_anchor.py`) — this thesis's first
  proposed mitigation for latent shift: a two-term loss (per-channel
  moment-matching toward the pretrained latent's N(0,1) statistics, plus
  a decorrelation term penalizing off-diagonal channel correlation) that
  keeps the fine-tuned latent distribution close to the *distribution*
  the pretrained DiT was trained on, without pinning to a literal
  reference point. Full derivation and rationale in
  `latent_space_shift_measure.md` (worth summarizing directly in the
  thesis methodology, it's a genuinely original piece of this project's
  method).
- **SDS distillation loss** (`sds_loss.py` /
  `windinet.training.sds_loss.SdsDistillationLoss`) — the second, and
  empirically stronger, mitigation: during VAE fine-tuning, distill the
  *frozen, already-trained* DiT's own flow-matching velocity prediction
  on the current encoder's latents into an auxiliary loss term, so the
  encoder is directly optimized for "what makes the DiT's job easy," not
  just reconstruction. Explicitly credit this as an idea proposed by the
  thesis advisor (2026-09-08) and empirically validated within this
  project as the best-performing Stage-2 result so far — good material
  for the "advisor guidance → hypothesis → experiment → result" narrative
  arc a thesis methodology section wants.
- **Loss weighting strategies**: fixed hand-tuned weights (the adopted
  approach) vs. two adaptive alternatives that were tried and rejected —
  GradNorm (`gradnorm.py`, never converged, 2-epoch oscillation, ~5x
  compute cost) and SoftAdapt (`soft_adapt.py`, caused an earlier MLW-
  weight collapse). Worth a short methodology paragraph explaining *why*
  fixed weights were kept despite the theoretical appeal of adaptive
  weighting — a concrete, defensible negative result.

### 4.5 Experimental protocol and evaluation metric
- Primary metric: `val_vrmse` — variance-normalized RMSE — computed on a
  fixed, seed-permuted 675-simulation held-out set, identical across every
  run for direct comparability. Per-channel breakdowns
  (`val_vrmse_<channel>`) are also logged.
- **Measured noise floor**: repeating the baseline across 3 seeds gave a
  spread of ~1.1% in val_vrmse — used throughout as the statistical bar
  for whether an ablation's result is a real effect (protocol requires
  clearing **2x the noise floor, ~2.2%**, to be reported as a finding).
  This is a genuinely strong methodological point for the thesis to lean
  on — most ablation studies in this space don't establish an explicit
  noise floor.
- **One-variable-at-a-time protocol**: every experiment changes exactly
  one axis from a named baseline config, with the hypothesis, changed
  variable, and kill-criterion written down *before* the run launches
  (worth citing this as good experimental hygiene / reproducibility
  practice, and possibly including the protocol verbatim as an appendix
  or methodology callout — it's in `EXPERIMENTS.md`'s "Protocol going
  forward" section).
- For Stage 2 (DiT), the analogous eval metric is `vae_dit_vrmse` (full
  encode→rollout→decode error) compared against `vae_only_vrmse` (the
  Stage-1 reconstruction floor with no DiT rollout involved) — the ratio
  between them isolates how much error the *DiT* itself is adding on top
  of whatever the VAE already loses, which is the right way to compare
  DiT-stage arms independent of which VAE checkpoint they're built on.

### 4.6 Infrastructure (optional subsection, include if the thesis wants an engineering/reproducibility component)
- Three heterogeneous HPC clusters used across the project: an on-prem
  Intel XPU cluster, an A6000-based cluster, and an H100-based cluster at
  the Leibniz Supercomputing Centre (LRZ) — a genuinely multi-hardware-
  vendor training setup (NVIDIA + Intel), worth mentioning as a
  reproducibility/portability data point.
- A collaborator (external to the thesis) also contributed compute/runs
  on one cluster — worth a brief acknowledgment-style mention if
  appropriate for the thesis's authorship conventions.
- A few concrete, quotable engineering findings if the thesis wants
  color: (1) a DDP + gradient-checkpointing ordering bug that silently
  crashed every multi-GPU smoke test on one cluster until fixed; (2) a
  transient GPU-memory spike at every checkpoint save (traced to an
  unnecessary live-tensor duplication during bf16 optimizer-state
  casting) that looked like GPU contention until root-caused; (3) an
  effective-batch-size accounting bug that silently doubled one cluster's
  per-step data budget relative to every other arm. These are the kind of
  concrete "what actually went wrong and how it was diagnosed" details
  that make an engineering-heavy methodology section credible rather than
  hand-wavy — pick 1–2 if space is tight.

## 5. Results

### 5.1 Stage 1 (VAE) — headline trajectory
Present as a narrative table of the baseline's evolution (each row a
single-variable change from the row above, with the ablation that
justified it):

| Step | Change | val_vrmse | Note |
|---|---|---|---|
| Original baseline (frozen trunk, adapter-only, 15 epochs) | — | ~0.0947 | starting point |
| + whole-encoder-trunk unfreeze | architecture | (real, clear win) | Open Question 3 |
| + epoch budget 15→30 | schedule | −6.09% | Open Question 20 (schedule *shape* didn't matter, only more steps) |
| Confirmed 128x128 baseline | — | **0.078662** | job 524322, `inflate_init="mean"` |
| + 256x256 resolution (confounded with anchor loss) | resolution+anchor | **0.054342** | ~31% drop vs. 128res baseline; resolution-only effect not yet cleanly isolated as of the last update |
| `inflate_init` mean→zeros (parsimony, not a metric win) | init scheme | ~wash (±noise floor) | real ~4.3–4.7% *per-channel* cost to the fresh pressure channel, hidden inside the aggregate wash — worth reporting as a sub-metric finding the aggregate number hides |

State clearly: the -30.9%-labeled "256x256 helps" result from the ledger
is real but was measured under now-superseded settings; a clean,
single-variable resolution-only re-measurement against the *current*
baseline was planned but its completion status should be re-checked in
`EXPERIMENTS.md` before quoting a final number in the thesis (the doc
itself flags this as not-yet-fully-confirmed as of the last update used
here).

### 5.2 Stage 1 — full ablation summary (23 open questions, condensed)
Reproduce as a results table — question, verdict, one-line reason.
(Pull the 23-row list directly from `EXPERIMENTS.md`'s "Open questions, in
priority order" section — it is already thesis-table-ready: seed noise
floor (~1.1%), latent-vs-objective bottleneck (partial), encoder unfreeze
(whole trunk wins), resolution (real gain), schedule-length (yes, real),
encoder LR multiplier (retired on parsimony grounds), epoch budget (yes,
real, this is what actually drove the epoch-count increase), channel
order (matters), decoder/adapter LR multiplier (already near-optimal),
copy-init (null result), log-density (borderline/not adopted), 8-sim
memorization ceiling (doesn't move), RMSE-only ablation (worse — H1/SSIM
are real regularizers not just gradient dilution), H2 curvature (no),
GradNorm (no, unstable + expensive), H1/SSIM/MLW weight retests (no
further effect), KL on/off (free at VAE level, but see Stage 2 caveat),
epoch/schedule-shape sweep (epoch count alone), fresh-channel init (zeros
adopted on parsimony), KL sweep saturation (flat across range tested,
again a VAE-only-metric finding), batch-size-vs-quality (closed, no
quality cost, hardware-only concern).
- **Advisor-design-philosophy note for Discussion**: multiple baseline
  choices here (encoder-LR multiplier, init scheme) were finalized on
  *principled/parsimony* grounds by the thesis advisor even where the
  measured metric gap was inside the noise floor — worth a short
  methodology-philosophy paragraph distinguishing "changed because a
  metric improved" from "changed because it's the more defensible
  default absent evidence otherwise," since both patterns appear
  legitimately in this project's history.

### 5.3 Stage 2 (DiT) — headline result table
Reproduce directly (already in the right shape for a thesis table),
all evaluated on the same 675-sim protocol / comparable subsets:

| Arm | vae_only vrmse (floor) | vae+dit vrmse | ratio to floor |
|---|---|---|---|
| **SDS (self-distillation)** | 0.0621 | **0.3662** | **5.9x — best result in the project** |
| ep20 baseline (no regularizer) | 0.0660 | 0.388–0.405 | 5.9–6.1x |
| anchor_kl1e7 | 0.0693 | 1.037 | 15.0x |
| cosine-schedule, KL 1e-7 (weakest KL) | 0.0578 | 1.085 | 18.8x |
| cosine-schedule, KL 1e-6 | 0.0590 | 1.130 | 19.2x |
| cosine-schedule, KL 1e-5 (strongest KL) | 0.0620 | 1.329 | 21.4x |
| untrained-DiT control (stock transformer, trained VAE) | 0.0598 | 1.355 | 22.7x |
| untrained-DiT + never-finetuned VAE (double floor / sanity check) | 0.472 | 2.921 | 6.2x (off a much worse floor) |

Key findings to narrate:
1. **KL regularization on the VAE, at every weight tested, makes the
   downstream DiT's rollout monotonically worse** (1e-7 weakest/least
   bad → 1e-5 strongest/worst), even though the *same* KL sweep looked
   essentially free/harmless when judged only by Stage-1 reconstruction
   quality (Section 5.2's Open Questions 19/22). This is the thesis's
   central empirical demonstration that Stage-1-only evaluation is
   insufficient for this two-stage pipeline — a regularizer must be
   judged by its effect on the actual downstream consumer.
2. **The weakest-KL arm is still only marginally better than an
   untrained (random-init) DiT control** — meaning KL regularization, at
   the weights tested, doesn't just add a *small* cost, it destroys most
   of the value of DiT fine-tuning altogether. Strong, quotable result.
3. **The SDS self-distillation regularizer solves this**: same
   reconstruction floor as the unregularized baseline (~0.06, no
   reconstruction cost), but the *best* rollout accuracy of any arm,
   including beating the plain baseline outright. This is evidence that
   the fix for "latent shift" isn't to constrain the latent distribution
   indirectly (KL, or the anchor-loss moment/decorrelation terms) but to
   optimize the encoder directly against the thing that actually matters
   (the frozen downstream DiT's own denoising objective).
4. Mention the diagnostic methodology used to characterize the shift
   itself even where a full DiT-probe number wasn't run for every arm:
   per-channel mean/std drift, Wasserstein-1 distributional divergence,
   cross-channel decorrelation, affine-vs-nonlinear displacement
   decomposition, and CKA — all from `latent_space_shift_measure.md`,
   itself a methodological contribution worth summarizing as a general
   protocol (Section 4's loss-function subsection already covers the
   *mitigation* half; this is the *measurement* half).
5. A specific, concrete diagnostic detail worth one sentence: the KL
   arms' worst-performing latent channels overlap heavily across
   different KL weights (the same ~4 channels are consistently the
   hardest), suggesting a structural, not purely regularization-
   strength-dependent, difficulty in specific latent dimensions — flagged
   as future work, not yet resolved.

### 5.4 What to explicitly caveat as "not yet final" in Results
- No clean, single-variable, current-baseline confirmation of the
  256x256-vs-128x128 resolution gain exists yet (the only complete 256res
  run conflates resolution with the anchor loss). Say this plainly rather
  than presenting 0.054342 as a clean resolution effect.
- 512x512 (native simulation resolution) has not produced a confirmed
  result as of the material this document was built from — report it as
  attempted/in-progress, not as a completed ablation, unless the live
  `EXPERIMENTS.md` shows otherwise by the time of writing.
- The lrz_ai (third-cluster) DiT arms have not yet had a real post-bugfix
  training attempt — the sng_pvc results above are the only confirmed DiT
  numbers as of this material's compilation.
- A second SDS variant (distilling from the *stock*, never-fine-tuned
  pretrained transformer rather than this project's own trained DiT, to
  rule out circularity) is implemented but has not been run yet — flag as
  a planned/future experiment, and note this could go either in Results
  ("in progress") or Future Work depending on whether it completes before
  the thesis is finalized.

## 6. Conclusion (recommended structure)
1. Restate the hypothesis (pretrained video-diffusion priors transfer to
   compressible CFD surrogate modeling) and summarize the evidence for
   and against it collected so far.
2. Summarize Stage 1 as a closed, successful result: a systematically
   validated VAE fine-tuning recipe with a large, statistically grounded
   ablation study, ~43% val_vrmse reduction over the naive baseline.
3. Summarize Stage 2 honestly as a promising-but-open result: a working,
   trained DiT achieving a meaningful, non-degenerate rollout (5.9x the
   reconstruction floor with the best recipe), a clearly diagnosed
   failure mode (KL-induced latent shift) affecting the naive
   regularization approach, and a validated, non-obvious fix
   (self-distillation) that the field (per the literature review's
   representation-drift/distillation discussion) would predict could work
   but that hadn't been demonstrated for this specific pretrained-video-
   to-physics transfer setting before.
4. Limitations: single dataset family (Euler_MQ, one PDE class); DiT
   stage not exhaustively tuned/converged at thesis-writing time; latent-
   shift diagnosis is empirical/correlational (KL strength correlates
   with worse rollout) rather than a full causal/theoretical account of
   *why*; only one scalar conditioning variable (gamma) explored, not
   more complex boundary/geometry conditioning; no direct wall-clock/
   compute comparison yet against a from-scratch-trained or classical
   numerical-solver baseline (worth explicitly stating this as a
   limitation if the thesis doesn't have that comparison — it's currently
   about relative ablation quality, not absolute speedup claims).
5. Future work: complete the lrz_ai DiT arms and the resolution-isolated
   256res/512res comparisons; run the untrained-DiT SDS variant to test
   circularity; investigate the structurally-hard latent channels
   identified in the KL sweep; extend beyond a single PDE family/
   parameter; a wall-clock/throughput comparison against a classical CFD
   solver to substantiate the "fast surrogate" motivation quantitatively.

---

## Appendix A — things to verify before the thesis is finalized (do not
guess these; check the live repo state)
- The exact current contents of `EXPERIMENTS.md`'s baseline/results
  section (it changes frequently — this document is a snapshot as of
  2026-09-12; DiT-stage rows especially will likely have moved).
- The arXiv ID (`2603.21210`) cited in this repo's `README.md` for the
  upstream WinDiNet urban-wind-flow paper — this looks anomalous (arXiv
  IDs don't currently reach 2603.xxxxx) and should be independently
  confirmed / corrected before citing it in the thesis's literature
  review or as the direct-predecessor citation.
- Exact meaning/derivation of the "MLW" loss term (`windinet/losses/
  mlw.py`) — referenced here only by its ledger role (net-negative,
  weight 0), not its full mathematical definition; read the source before
  writing the Methodology loss-function subsection.
- Whether the 512x512 VAE run (job 5767882 / its resume) completed by
  thesis-writing time.
- Whether any lrz_ai DiT arms have produced eval numbers by thesis-
  writing time (none did as of this compilation).

## Appendix B — key file/config pointers for whoever drafts figures/tables
- `EXPERIMENTS.md` — current baseline, established facts, open-question
  index (primary Results source).
- `EXPERIMENTS_archive.md` — full write-up/methodology/results detail for
  every individual ablation (use for Results-section deep detail or
  appendix tables).
- `infra.md` — cluster/hardware/throughput detail (Methodology
  infrastructure subsection).
- `latent_space_shift_measure.md` — full latent-shift measurement
  protocol and the anchor-loss derivation (Methodology + Results, Stage
  2/Discussion).
- `README.md` — architecture summary, pipeline commands, the (unverified)
  upstream-paper citation.
- `windinet/losses/`, `windinet/loss_weighting/` — exact loss
  implementations for the Methodology section.
- `configs/finetune_vae/`, `configs/dit/` — exact hyperparameters for
  every named experiment, if the thesis wants to cite specific config
  values verbatim.
