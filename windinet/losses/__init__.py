"""
Loss package for physics-informed VAE training.

This package provides:
    - individual loss components
    - reconstruction loss composition
"""


from .rmse import rmse_loss
from .h1_semi_norm import h1_seminorm_loss
from .h2_semi_norm import h2_seminorm_loss
from .ssim import SSIMLoss
from .mlw import mlw_loss
from .pcc import pcc_loss
from .vrms import vrms_loss
from .kl_divergence import kl_divergence_loss
from .reconstruction import reconstruction_losses


__all__ = [
    "rmse_loss",
    "h1_seminorm_loss",
    "h2_seminorm_loss",
    "SSIMLoss",
    "mlw_loss",
    "pcc_loss",
    "vrms_loss",
    "kl_divergence_loss",
    "reconstruction_losses",
]