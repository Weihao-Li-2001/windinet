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
from .mlw import mlw_loss


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
) -> dict[str, torch.Tensor]:
    """
    Compute reconstruction loss components.

    Components:

        RMSE:
            Pixel-wise reconstruction accuracy

        H1:
            Spatial gradient consistency

        SSIM:
            Structural similarity

        MLW:
            Multi-level wavelet consistency


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

    Returns:
        Dictionary:

        {
            "rmse": Tensor,
            "h1": Tensor,
            "ssim": Tensor,
            "mlw": Tensor
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
    # Gradient consistency
    # ------------------------------------

    losses["h1"] = h1_seminorm_loss(
        pred,
        target,
    )


    # ------------------------------------
    # Structural similarity
    # ------------------------------------

    losses["ssim"] = ssim_module(
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


    return losses