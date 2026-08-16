# Measuring Latent-Space Shift After Encoder Fine-Tuning

A protocol for quantifying how far the VAE latent space moves when you modify and
fine-tune the encoder of a pretrained video diffusion model (LTX-Video), and how
much that shift will cost you in DiT fine-tuning.

Context: 4-channel CFD dataset, encoder stem modified from 3→4 channels, full
encoder unfreeze. Goal: keep the latent shift small so the DiT needs fewer
fine-tuning epochs.

---

## Step 0 — Create a directly comparable "before" encoder

You can't run a 4-channel input through the original 3-channel encoder, so
"before vs after" isn't defined until you fix this. The clean way: checkpoint the
encoder immediately after you modify the stem but *before* any training. If you
used the split-stem (zero-init 4th-channel branch), this checkpoint produces
exactly the pretrained latent — the 4th channel contributes nothing at init. Call
it `E_before`. Your fully fine-tuned encoder is `E_after`. Both accept 4 channels,
both take identical inputs, and the only difference between them is the training
you're trying to measure.

If you didn't use a split stem, the fallback is `E_before(x) = E_pretrained(x[:, :3])`
— run the frozen original on just the first three channels. It's slightly less
clean (the 4th field never enters the reference) but still a valid baseline.

> **Note on mean-init:** if you initialized the 4th channel with the mean of the
> pretrained first-layer weights, the modified-but-untrained checkpoint is
> *already displaced* and is **not** a valid `z_before`. See the final section —
> always use `E_pretrained(x[:, :3])` as the reference in that case.

---

## Step 1 — Assemble a representative probe set

Pick a few hundred held-out samples spanning your parameter ranges (different
geometries, inlet conditions, whatever varies in your dataset). A handful won't do
— latent shift is often uneven across the input distribution, and you want the
average, not one lucky point.

---

## Step 2 — Encode deterministically through both

Critical detail: the LTX-Video VAE is a *variational* encoder — it outputs a
distribution, and `.sample()` adds noise. If you sample, your "shift" will be
polluted by that stochasticity. Take the posterior **mean** (mode), not a sample,
so the only difference between `z_before` and `z_after` is the weight change.

```python
@torch.no_grad()
def encode_mean(E, x):
    out = E(x)                       # x: [B, 4, T, H, W]
    return out.latent_dist.mean      # NOT .sample()  -> [B, 128, T', H', W']

z_before = torch.cat([encode_mean(E_before, xb) for xb in loader])
z_after  = torch.cat([encode_mean(E_after,  xb) for xb in loader])
# both: [N, 128, T', H', W'], paired sample-for-sample
```

Everything below operates on these two paired tensors. Keep two reshapes handy:

```python
# per-channel view: collapse everything except the 128 channels
def chan(z): return z.permute(1, 0, 2, 3, 4).reshape(128, -1)   # [128, M]

# per-position view: each spatiotemporal location is a 128-d vector
def pos(z):  return z.permute(0, 2, 3, 4, 1).reshape(-1, 128)    # [M, 128]
```

---

## Step 3 — Per-channel mean/std drift

*Cheapest and most interpretable; start here.*

The DiT expects each of the 128 channels to be roughly N(0, 1). Check whether
fine-tuning moved them:

```python
mb, sb = chan(z_before).mean(1), chan(z_before).std(1)
ma, sa = chan(z_after ).mean(1), chan(z_after ).std(1)

print("mean drift  |dmu| :", (ma - mb).abs().mean().item(), "max:", (ma - mb).abs().max().item())
print("std ratio  sa/sb  :", (sa / sb).mean().item(), "range:", (sa / sb).min().item(), (sa / sb).max().item())
```

**Reading it:** mean drift under ~0.2 and std ratios inside roughly [0.8, 1.25] is
comfortable — the DiT is denoising toward nearly the same target. A handful of
channels blowing up tells you *where* the shift concentrated, which is more useful
than any single global number.

---

## Step 4 — Per-channel distributional divergence (shape, not just moments)

Two channels can share mean and std but have different distribution shapes.
Wasserstein-1 catches that and is numerically well-behaved:

```python
from scipy.stats import wasserstein_distance

w1 = [wasserstein_distance(chan(z_before)[c].cpu().numpy(),
                           chan(z_after )[c].cpu().numpy()) for c in range(128)]
```

Sort `w1` descending and look at the top few channels — that's your list of "which
channels reorganized most."

---

## Step 5 — Cross-channel correlation structure

*Specific to LTX-Video and worth emphasizing.*

Pretraining decorrelates the 128 channels (their Figure 3 — off-diagonal
correlation goes to near-zero at convergence). If your fine-tuning re-introduced
correlation, you've partly undone what pretraining built, and no amount of DiT
fine-tuning fully fixes that.

```python
def offdiag_corr(z):
    zc = chan(z); zc = zc - zc.mean(1, keepdim=True)
    C = (zc @ zc.T) / zc.shape[1]
    d = torch.sqrt(torch.diag(C).clamp_min(1e-8))
    C = C / (d[:, None] * d[None, :])
    return (C - torch.diag(torch.diag(C))).abs().mean().item()

print("off-diagonal corr  before:", offdiag_corr(z_before), " after:", offdiag_corr(z_after))
```

If "after" is substantially higher, that's a red flag independent of everything
else — it means the anchor / decorrelation regularizer is what you need.

---

## Step 6 — Per-sample displacement, split into coherent vs incoherent

Steps 3–5 are distributional. This asks: for a *specific* input, how far did its
latent move, and is that movement a benign global transform or a genuine reshuffle?
A large but uniform shift is easy for the DiT to absorb; a large input-dependent,
nonlinear shift is not.

```python
X, Y = pos(z_before), pos(z_after)

# raw relative displacement
rel = (Y - X).norm(dim=1) / (X.norm(dim=1) + 1e-6)
print("median relative displacement:", rel.median().item())

# how much of it is explained by a single global affine map?
Xa = torch.cat([X, torch.ones(X.shape[0], 1)], 1)
sol = torch.linalg.lstsq(Xa, Y).solution
resid = (Y - Xa @ sol).norm() / Y.norm()
print("residual after best affine fit:", resid.item())
```

**Reading it:** a **low** affine residual means the shift is mostly a global
rotate/scale/translate — recoverable cheaply, and exactly the case where the
affine-absorption trick (insert a fixed un-shift before the DiT) lets you skip most
DiT retraining. A **high** residual means the encoder learned a genuinely new
nonlinear code and the DiT must relearn it.

---

## Step 7 — CKA (the similarity the DiT actually perceives)

The DiT is roughly invariant to rotation and isotropic scaling of its latent input,
so a raw distance overstates how much it "notices." Centered Kernel Alignment
measures representational similarity modulo exactly those transforms:

```python
def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return ((X.T @ Y).norm()**2 / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()

print("CKA(before, after):", linear_cka(X, Y))   # 1.0 identical; <~0.9 real concern
```

CKA near 1 with a large raw displacement means the shift is mostly the "harmless"
kind — reassuring even if Step 6's raw number looked scary.

---

## Step 8 — The decisive end-to-end test: does the pretrained DiT still denoise these latents?

Every metric above is a proxy. This is the question that actually determines your
DiT epoch count. Take `z_after`, add rectified-flow noise, and run your
*not-yet-fine-tuned* DiT one step — measure velocity-prediction error. Do the same
with `z_before` as the reference.

```python
@torch.no_grad()
def probe(z, dit, cond, ts=(0.1, 0.3, 0.5)):
    for t in ts:
        eps = torch.randn_like(z)
        zt  = (1 - t) * z + t * eps
        v_pred, v_tgt = dit(zt, t, cond), eps - z      # LTX predicts v = eps - z0
        print(f"  t={t}:  {((v_pred - v_tgt)**2).mean().item():.4f}")

print("pretrained DiT on z_before:"); probe(z_before, pretrained_dit, cond)
print("pretrained DiT on z_after :"); probe(z_after,  pretrained_dit, cond)
```

If the error on `z_after` is close to `z_before`, the pretrained DiT still
"understands" your fine-tuned latents — the space wasn't destroyed in any way that
matters, and DiT fine-tuning will converge fast. If it's dramatically worse, the
shift is real and costly, and you'd want the shift-reduction techniques (split
stem, anchor loss) before committing GPU hours to DiT fine-tuning.

---

## How to read the whole set together

Run them in order and stop being worried as soon as the cheap ones clear.

| Observation | Interpretation | Action |
|---|---|---|
| Steps 3 + 5 look pretrained-like **and** Step 8 error ≈ baseline | Benign shift; full encoder fine-tuning was a free win | Fine-tune the DiT normally |
| Steps 3–5 fine **but** Step 6 residual high / Step 7 CKA low | Marginals preserved but the mapping reorganized nonlinearly | DiT needs real fine-tuning; reconstruction gains are legitimate |
| Step 5 correlation rose **or** Step 3 stats drifted far | The one case more DiT training won't rescue | Add the latent-anchor loss to pull the encoder back into the pretrained envelope |
| Step 6 residual low (affine-dominated) | Shift is a global affine transform | Absorb it with a fixed transform before the DiT rather than fine-tuning it away |

**Single scalars to log across runs:** track **CKA (Step 7)** and the **Step-8
velocity-error gap**. Together they tell you almost everything about how much DiT
retraining a given encoder configuration will cost. Watch them as a function of λ
on your anchor loss to find the setting that keeps the latents cheap without giving
up reconstruction.

---

## Reducing the shift: the latent-distribution anchor loss

The metrics above *measure* the shift. This section *reduces* it. It's the most
direct knob on the exact quantity the DiT cares about, and the one referenced
throughout the tables above ("add the latent-anchor loss").

### Why it works

The DiT was trained to denoise toward latents that are, per-channel, approximately
N(0, 1) and decorrelated across the 128 channels — precisely the structure
LTX-Video's uniform-logvar KL loss and the training progression in their Figure 3
produce. If you fine-tune the encoder with reconstruction (and physics) losses
alone, nothing stops it from wandering into a correlated or non-Gaussian regime
that reconstructs well but is *harder for the DiT to denoise no matter how long you
train it*. The anchor loss lets the encoder reorganize freely **within** the
pretrained distributional envelope while forbidding it from leaving that envelope.

You get to keep the reconstruction gains of full encoder fine-tuning, but the
latent stays in the region the DiT already understands — which is exactly what
shrinks the DiT epoch count.

### The loss

```python
def latent_anchor_loss(z):          # z: [B, 128, T, H, W]
    zc = z.permute(1, 0, 2, 3, 4).reshape(128, -1)          # [128, M]

    # (a) per-channel moment matching -> each channel toward N(0, 1)
    mu, std = zc.mean(1), zc.std(1)
    l_moment = mu.pow(2).mean() + (std - 1).pow(2).mean()

    # (b) decorrelation -> keep the 128 channels near-orthogonal
    zc0  = zc - zc.mean(1, keepdim=True)
    corr = (zc0 @ zc0.T) / zc0.shape[1]
    off  = corr - torch.diag(torch.diag(corr))
    l_decorr = off.pow(2).mean()

    return l_moment + l_decorr

# total encoder-stage objective
loss = recon_loss + physics_losses + lam * latent_anchor_loss(z)
```

Two terms, each targeting a specific failure mode the diagnostics flag:

- **Moment matching** keeps per-channel mean/std at 0/1 — directly counters the
  drift Step 3 measures.
- **Decorrelation** keeps the off-diagonal correlation low — directly counters the
  regression Step 5 measures (the LTX-Video-specific one, where fine-tuning
  re-correlates channels that pretraining had decorrelated).

### Choosing λ

Sweep `lam` and read it off the diagnostics rather than guessing:

- Start small (e.g. `1e-3` to `1e-2` relative to a normalized reconstruction loss)
  and increase until Step 3 stats and Step 5 correlation return to pretrained-like
  values.
- Watch reconstruction at the same time. The right λ is the largest value at which
  reconstruction is still essentially unchanged from your full-unfreeze baseline —
  past that point you're trading reconstruction for latent stability, which defeats
  the purpose of having abandoned the color adapter.
- The clean way to tune it: plot **CKA (Step 7)** and the **Step-8 velocity-error
  gap** against λ. You're looking for the knee where the latents become cheap for
  the DiT without reconstruction degrading.

### Stronger variant: distill against a fixed reference

If moment-matching alone still drifts too far, anchor to an actual reference point
rather than just the distribution. Capture a **frozen copy** of the encoder right
after Stage-0 warmup (`E_ref`) and add a distillation term on the native channels'
contribution:

```python
with torch.no_grad():
    z_ref = E_ref(x).latent_dist.mean          # frozen post-warmup reference
l_distill = (z - z_ref).pow(2).mean()

loss = recon_loss + physics_losses + lam * latent_anchor_loss(z) + beta * l_distill
```

This is more restrictive — it pins latents to a specific point, not just a
distribution — so reach for it only if the moment-matching version leaves the
Step-6 residual or Step-7 CKA outside your comfort range. For most cases the
distribution-level anchor (moment + decorrelation) is the better default because it
constrains only what the DiT actually depends on and leaves the encoder maximal
freedom to reconstruct.

### Where it fits in the pipeline

The anchor loss is applied during the **encoder/VAE fine-tuning stage**, alongside
your reconstruction and physics losses — *before* you preprocess the dataset into
latents and fine-tune the DiT. It shrinks the shift at its source so that, by the
time the DiT stage begins, there is less shift for the DiT to absorb. It pairs
naturally with the split-stem (zero-init 4th channel) and bottleneck-adjacent
freezing: the stem and freezing reduce the shift structurally, the anchor loss
regularizes whatever shift remains.

---

## Mean-init vs zero-init for the 4th channel: does the analysis change?

### Is mean-init fine?

Copying the average of the pretrained first-layer weights into the 4th channel
(`W[:, 3] = W[:, :3].mean(dim=1)`) is the standard "inflation" trick and is
perfectly reasonable for reconstruction — the new channel starts as a sensible
"average" feature detector rather than a blank one.

But it has one consequence central to minimizing latent shift: **mean-init does not
preserve the latent at initialization.** The moment you feed a 4-channel input, the
4th channel's nonzero weights inject signal into the latent, so the encoder is
already displaced from the pretrained mapping *before a single training step*.
Zero-init is the opposite — the 4th channel contributes exactly nothing at init, so
the latent is bit-identical to the pretrained 3-channel encoder.

Zero-init is **not** a dead initialization. For a conv, the gradient w.r.t. the
4th-channel weights is `(4th-channel input) × (upstream gradient)`, and the 4th
channel input is nonzero, so those zeroed weights receive gradient and start
learning on step one. You get identical-at-init *and* immediate learning. If your
priority is small latent shift, zero-init is strictly the better starting point: it
begins on the pretrained manifold and moves away only as far as training pushes it,
whereas mean-init *starts* off-manifold and full fine-tuning proceeds from that
already-displaced point.

At convergence with full unfreeze the difference is often small, but mean-init
tends to end further from the pretrained latent than zero-init + anchor loss would.
**If reconstruction is identical either way, prefer zero-init for the shift budget.**

### Does the analysis change? The metrics don't — but your reference does

Every metric (channel stats, W1, correlation, CKA, affine residual, DiT probe) is
computed identically. What changes is the definition of `z_before`, and if you get
that wrong with mean-init you'll mismeasure.

The protocol assumes `z_before` is the *pretrained* latent. With zero-init, the
modified-but-untrained encoder is that reference, so checkpointing right after
modifying the stem gives you a valid `z_before` for free. With mean-init, that same
checkpoint is already displaced by the init, so using it as your reference would
silently hide the init-time jump and report only the post-init drift — you'd
underestimate the true distance.

The fix is to define the reference independently of how you initialized the 4th
channel:

```python
# Reference is ALWAYS the genuine pretrained mapping, not the mean-init checkpoint
@torch.no_grad()
def encode_mean(E, x):  return E(x).latent_dist.mean

z_before = torch.cat([encode_mean(E_pretrained, xb[:, :3]) for xb in loader])  # frozen 3-ch original
z_after  = torch.cat([encode_mean(E_finetuned,  xb)        for xb in loader])  # mean-init, fully FT
```

Run the original frozen 3-channel encoder on the first three channels as the
baseline, regardless of init scheme. This gives a consistent "pretrained space"
anchor that both zero-init and mean-init runs can be compared against on equal
footing.

### Which metrics care and which don't

**Paired measures** — per-sample displacement (Step 6), the affine-residual fit
(Step 6), and CKA (Step 7) — depend entirely on `z_before` being the right
reference. These change with init if you're careless, and must use the corrected
`E_pretrained(x[:, :3])` reference.

**Reference-free measures** don't change with init, because they evaluate `z_after`
directly against what the DiT expects:

- **Steps 3 and 5** read as "how close is `z_after` to per-channel N(0,1) and
  decorrelated" — that target is fixed by pretraining, no `z_before` needed.
- **Step 8** runs the pretrained DiT on `z_after` and measures velocity error —
  completely init-agnostic.

These remain the most trustworthy tools precisely because they sidestep the
reference-definition trap.

### Practical guidance

With mean-init, lean on **Steps 3, 5, and 8** as your primary read (immune to the
init), and only use Steps 6–7 with the corrected `E_pretrained(x[:, :3])`
reference. If you run both init schemes, compare them on **Step 8** — the
pretrained-DiT velocity-error gap — since that's the number that predicts how many
DiT epochs each will cost, and it treats both inits fairly.

**Suggested experiment:** run the Step-8 probe for zero-init and mean-init side by
side on the *same* fine-tuning budget. Expectation is that zero-init (ideally with
the anchor loss holding it) lands closer to the pretrained-DiT baseline error — but
it's cheap to verify and worth knowing for your specific 4-channel setup rather
than taking it on faith.
