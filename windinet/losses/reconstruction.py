"""
Reconstruction loss composition for VAE decoder finetuning.

This module only computes individual loss components.

Loss weighting is handled separately by
training.loss_weighting strategies.

Expected input:
    pred, target: [B, C, T, H, W]

Returns:
    Dictionary containing individual losses.
"""


import torch

from .rmse import rmse_loss
from .h1_semi_norm import h1_seminorm_loss
from .h2_semi_norm import h2_seminorm_loss
from .mlw import mlw_loss
from .pcc import pcc_loss
from .vrms import vrms_loss
from .kl_divergence import kl_divergence_loss


def reconstruction_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim_module,
    wavelet: str = "db2",
    spatial_level: int | None = None,
    temporal_level: int | None = None,
    mlw_beta: float = 10.0,
    mlw_eps: float = 1e-6,
    latent_mean: torch.Tensor | None = None,
    latent_logvar: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute reconstruction loss components.

    Components:

        RMSE:
            Pixel-wise reconstruction accuracy

        H1:
            Spatial gradient consistency

        H2:
            Spatial curvature consistency (second derivatives) -- opt-in,
            see loss_weighting.weights in the config

        SSIM:
            Structural similarity

        MLW:
            Multi-level wavelet consistency

        PCC:
            Structural correlation, magnitude-invariant -- opt-in

        VRMS:
            Variance-normalized RMSE (same metric as val_vrmse) -- opt-in

        KL:
            Encoder posterior vs. standard-normal-prior divergence --
            opt-in, only computed if latent_mean/latent_logvar are given

    RMSE/H1/H2/SSIM/MLW/PCC/VRMS are always computed (cheap) and always
    present in the returned dict; whether they influence training is
    entirely up to loss_weighting.weights assigning them a nonzero weight
    (see windinet/config.py LossWeightingConfig) -- same "always compute,
    weight decides" convention MLW already used before this. KL is the one
    exception: it needs the raw encoder distribution, not pred/target, so
    it's only included when the caller has one to give (VaeTrainer.train /
    _evaluate, from VaeTrainer._encode's extra return values).

    Args:
        pred:
            Reconstructed fields.
            Shape [B,C,T,H,W]

        target:
            Ground-truth fields.
            Shape [B,C,T,H,W]

        ssim_module:
            Pre-created SSIMLoss module.

        wavelet:
            Wavelet basis for MLW.

        latent_mean:
            Encoder posterior mean (pre-rescale), for KL. None skips KL.

        latent_logvar:
            Encoder posterior log-variance (pre-rescale), for KL. None
            skips KL.

    Returns:
        Dictionary:

        {
            "rmse": Tensor,
            "h1": Tensor,
            "h2": Tensor,
            "ssim": Tensor,
            "mlw": Tensor,
            "pcc": Tensor,
            "vrms": Tensor,
            "kl": Tensor,   # only if latent_mean/latent_logvar given
        }
    """

    losses = {}

    # ------------------------------------
    # Pixel reconstruction
    # ------------------------------------

    losses["rmse"] = rmse_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Gradient / curvature consistency
    # ------------------------------------

    losses["h1"] = h1_seminorm_loss(
        pred,
        target,
    )

    losses["h2"] = h2_seminorm_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Structural similarity / correlation
    # ------------------------------------

    losses["ssim"] = ssim_module(
        pred,
        target,
    )

    losses["pcc"] = pcc_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Frequency / interface preservation
    # ------------------------------------

    losses["mlw"] = mlw_loss(
        pred,
        target,
        wavelet=wavelet,
        beta=mlw_beta,
        eps=mlw_eps,
        spatial_level=spatial_level,
        temporal_level=temporal_level,
    )


    # ------------------------------------
    # Variance-normalized error
    # ------------------------------------

    losses["vrms"] = vrms_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Latent regularization (opt-in: needs the encoder distribution)
    # ------------------------------------

    if latent_mean is not None and latent_logvar is not None:
        losses["kl"] = kl_divergence_loss(
            latent_mean,
            latent_logvar,
        )


    return losses