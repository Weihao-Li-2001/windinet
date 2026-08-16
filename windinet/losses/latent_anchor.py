"""
Latent-distribution anchor loss.

PhD-advisor-motivated (latent_space_shift_measure.md, 2026-08-16): full
encoder unfreeze lets the VAE reorganize its latent space freely while
chasing reconstruction/physics losses, with nothing stopping it from
wandering into a correlated or non-Gaussian regime that reconstructs well
but is harder for a downstream DiT to denoise, no matter how long the DiT
is trained. This loss keeps the encoder inside the pretrained latent's
distributional envelope (per-channel N(0,1), decorrelated across channels)
while leaving it free to reorganize *within* that envelope. Two terms:

    - moment matching: pulls each channel's batch mean/std toward 0/1.
    - decorrelation: penalizes off-diagonal covariance across channels.

Operates on the RESCALED latents (VaeTrainer._encode's `latents`, i.e. what
scripts/latent_shift_metrics.py and scripts/latent_stats.py already treat as
"should land at mean 0 / std 1 per channel"), not the raw pre-rescale
posterior mean/logvar windinet.losses.kl_divergence operates on. KL
regularizes the encoder's own probabilistic output sample-by-sample against
a literal N(0, I) prior; this instead regularizes the batch-level empirical
statistics of the (rescaled) latents actually consumed downstream, and adds
the cross-channel decorrelation term KL does not provide.

Expected input:
    latents: [B, C, ...] (e.g. [B, C, T, H, W]) -- rescaled latents.
"""

import torch


def latent_anchor_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    Compute the moment-matching + decorrelation anchor loss.

    Args:
        latents:
            Rescaled latents. Shape [B, C, ...].

    Returns:
        Scalar loss: per-channel (mean^2 + (std - 1)^2), averaged over
        channels, plus the mean squared off-diagonal covariance across
        channels.
    """
    c = latents.shape[1]
    zc = latents.movedim(1, 0).reshape(c, -1)

    mu = zc.mean(dim=1)
    std = zc.std(dim=1)
    l_moment = mu.square().mean() + (std - 1.0).square().mean()

    zc0 = zc - zc.mean(dim=1, keepdim=True)
    cov = (zc0 @ zc0.T) / zc0.shape[1]
    off = cov - torch.diag(torch.diag(cov))
    l_decorr = off.square().mean()

    return l_moment + l_decorr
