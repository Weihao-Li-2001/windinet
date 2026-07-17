"""
Physics-informed loss functions for VAE decoder finetuning.

All loss functions expect:
  - pred, target: [B, 2, T, H, W] (u, v velocity channels)
  - footprint: [B, H, W] where 1=building, 0=fluid
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

import ptwt

class GaussianFilter(nn.Module):
    """Gaussian filter used by SSIM."""

    def __init__(
        self,
        channels: int,
        window_size: int = 11,
        sigma: float = 1.5,
    ):
        super().__init__()

        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        kernel2d = torch.outer(g, g)
        kernel2d /= kernel2d.sum()

        kernel = kernel2d.expand(channels, 1, window_size, window_size)

        self.register_buffer("kernel", kernel)
        self.groups = channels
        self.padding = window_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.kernel,
            padding=self.padding,
            groups=self.groups,
        )
    
def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    gaussian: GaussianFilter,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM map."""

    mu_pred = gaussian(pred)
    mu_gt = gaussian(target)

    mu_pred2 = mu_pred.pow(2)
    mu_gt2 = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred2 = gaussian(pred * pred) - mu_pred2
    sigma_gt2 = gaussian(target * target) - mu_gt2
    sigma_pred_gt = gaussian(pred * target) - mu_pred_gt

    numerator = (
        (2 * mu_pred_gt + c1)
        * (2 * sigma_pred_gt + c2)
    )

    denominator = (
        (mu_pred2 + mu_gt2 + c1)
        * (sigma_pred2 + sigma_gt2 + c2)
    )

    return numerator / (denominator + 1e-8)

def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural Similarity (SSIM) loss.

    Args:
        pred:   [B,C,T,H,W]
        target: [B,C,T,H,W]

    Returns:
        1 - mean(SSIM)
    """

    B, C, T, H, W = pred.shape

    pred = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    target = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    gaussian = GaussianFilter(
        channels=C,
        window_size=window_size,
        sigma=sigma,
    ).to(pred.device)

    ssim_map = _ssim(
        pred,
        target,
        gaussian,
    )

    return 1.0 - ssim_map.mean()

def _wavelet_loss_spatial(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelet: str = "db2",
    level: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Spatial multilevel wavelet loss.

    Args:
        pred, target:
            [B, C, T, H, W]

    Returns:
        Spatial wavelet loss.
    """

    B, C, T, H, W = pred.shape

    pred = pred.permute(0, 2, 1, 3, 4).reshape(B * T * C, H, W)
    target = target.permute(0, 2, 1, 3, 4).reshape(B * T * C, H, W)

    pred_coeffs = ptwt.wavedec2(
        pred,
        wavelet=wavelet,
        level=level,
        axes=(-2, -1),
    )

    target_coeffs = ptwt.wavedec2(
        target,
        wavelet=wavelet,
        level=level,
        axes=(-2, -1),
    )

    loss = 0.0

    # skip approximation coefficients
    for pred_detail, target_detail in zip(pred_coeffs[1:], target_coeffs[1:]):

        for pred_band, target_band in zip(pred_detail, target_detail):

            loss += torch.mean(
                torch.abs(
                    torch.log2(pred_band.abs() + eps)
                    - torch.log2(target_band.abs() + eps)
                )
            )

    return loss

def _wavelet_loss_temporal(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelet: str = "db2",
    level: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Temporal wavelet loss.

    Wavelet transform is applied along the time dimension.
    """

    pred_coeffs = ptwt.wavedec(
        pred,
        wavelet=wavelet,
        level=level,
        axis=2,      # T dimension
    )

    target_coeffs = ptwt.wavedec(
        target,
        wavelet=wavelet,
        level=level,
        axis=2,
    )

    loss = 0.0

    # skip approximation
    for pred_detail, target_detail in zip(pred_coeffs[1:], target_coeffs[1:]):

        loss += torch.mean(
            torch.abs(
                torch.log2(pred_detail.abs() + eps)
                - torch.log2(target_detail.abs() + eps)
            )
        )

    return loss

def mlw_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelet: str = "db2",
    beta: float = 10.0,
    eps: float = 1e-6,
    spatial_level: int | None = None,
    temporal_level: int | None = None,
) -> torch.Tensor:
    """
    Multi-Level Wavelet Loss.

    Based on:

        Wavelet-Based Loss for High-Frequency Interface Dynamics
        https://arxiv.org/abs/2209.02316

    Args:
        pred, target:
            [B, C, T, H, W]

    Returns:
        Scalar MLW loss.
    """

    spatial_loss = _wavelet_loss_spatial(
        pred,
        target,
        wavelet=wavelet,
        level=spatial_level,
        eps=eps,
    )

    temporal_loss = _wavelet_loss_temporal(
        pred,
        target,
        wavelet=wavelet,
        level=temporal_level,
        eps=eps,
    )

    return spatial_loss + beta * temporal_loss

def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ssim_module,
    lambda_rmse: float = 1.0,
    lambda_h1: float = 0.5,
    lambda_ssim: float = 0.2,
    lambda_mlw: float = 0.05,
    wavelet: str = "db2",
    spatial_level: int | None = None,
    temporal_level: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Combined reconstruction loss for BubbleDiNet VAE finetuning.

    Loss:

        L = λ_rmse * RMSE
          + λ_h1   * H1
          + λ_ssim * SSIM
          + λ_mlw  * MLW

    Args:
        pred:
            Reconstructed fields.
            Shape [B, C, T, H, W]

        target:
            Ground-truth fields.
            Shape [B, C, T, H, W]

        ssim_module:
            Pre-created SSIMLoss module.

    Returns:
        Dictionary containing each component and total loss.
    """

    losses = {}

    # ----------------------------------------------------------
    # RMSE
    # ----------------------------------------------------------

    losses["rmse"] = rmse_loss(
        pred,
        target,
    )

    # ----------------------------------------------------------
    # H1 Semi-Norm
    # ----------------------------------------------------------

    losses["h1"] = h1_seminorm_loss(
        pred,
        target,
    )

    # ----------------------------------------------------------
    # SSIM
    # ----------------------------------------------------------

    losses["ssim"] = ssim_module(
        pred,
        target,
    )

    # ----------------------------------------------------------
    # Multi-Level Wavelet
    # ----------------------------------------------------------

    losses["mlw"] = mlw_loss(
        pred,
        target,
        wavelet=wavelet,
        spatial_level=spatial_level,
        temporal_level=temporal_level,
    )

    # ----------------------------------------------------------
    # Total
    # ----------------------------------------------------------

    losses["total"] = (
          lambda_rmse * losses["rmse"]
        + lambda_h1   * losses["h1"]
        + lambda_ssim * losses["ssim"]
        + lambda_mlw  * losses["mlw"]
    )

    return losses

# def _distance_weight_map(fluid_mask: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
#     """Compute w(x) = 1 + alpha * exp(-d(x)^2 / (2*sigma^2)) in fluid, 0 in buildings.

#     fluid_mask: [B, H, W] with 1=fluid, 0=building.
#     """
#     w_maps = []
#     fm_cpu = fluid_mask.detach().to("cpu")
#     for i in range(fm_cpu.shape[0]):
#         fm = fm_cpu[i].numpy().astype(bool)
#         d = distance_transform_edt(fm)
#         w = 1.0 + alpha * np.exp(-(d * d) / (2.0 * sigma * sigma))
#         w *= fm.astype(np.float32)
#         w_maps.append(torch.from_numpy(w))
#     return torch.stack(w_maps, dim=0).to(fluid_mask.device)


# def distance_weighted_mse(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     footprint: torch.Tensor,
#     alpha: float = 2.0,
#     sigma: float = 20.0,
# ) -> torch.Tensor:
#     """MSE weighted by distance to buildings, zero inside buildings.

#     Pixels near building boundaries receive approximately (1 + alpha) times
#     the weight of far-field pixels, with a Gaussian falloff controlled by sigma.

#     Args:
#         pred, target: [B, 2, T, H, W] velocity fields.
#         footprint: [B, H, W] where 1=building, 0=fluid.
#         alpha: wall proximity weight amplification.
#         sigma: Gaussian falloff distance in pixels.
#     """
#     fluid = (1.0 - footprint.float()).clamp(0, 1)
#     w = _distance_weight_map(fluid, alpha=alpha, sigma=sigma)  # [B, H, W]
#     loss = (pred - target).pow(2) * w[:, None, None]  # broadcast over C and T
#     return loss.mean()


# def divergence_loss(pred: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
#     """Penalise violations of incompressibility (du/dx + dv/dy ~ 0).

#     Uses forward finite differences. The loss is masked so that only
#     2x2 stencils entirely within fluid contribute.

#     Args:
#         pred: [B, 2, T, H, W] velocity fields (channel 0=u, 1=v).
#         footprint: [B, H, W] where 1=building, 0=fluid.
#     """
#     if footprint.dim() == 4:
#         footprint = footprint[:, 0]

#     f = (1.0 - footprint.float()).clamp(0, 1)  # 1=fluid

#     u = pred[:, 0]  # [B, T, H, W]
#     v = pred[:, 1]

#     du_dx = F.pad(u[..., :, 1:] - u[..., :, :-1], (0, 1, 0, 0))
#     dv_dy = F.pad(v[..., 1:, :] - v[..., :-1, :], (0, 0, 0, 1))
#     div = du_dx + dv_dy

#     # Valid mask: all corners of the 2x2 stencil must be fluid
#     f00 = f
#     f01 = F.pad(f[:, :, 1:], (0, 1, 0, 0))
#     f10 = F.pad(f[:, 1:, :], (0, 0, 0, 1))
#     f11 = F.pad(f[:, 1:, 1:], (0, 1, 0, 1))
#     mask = (f00 * f01 * f10 * f11)[:, None]  # [B, 1, H, W]

#     return (div.pow(2) * mask).mean()


# def wall_no_penetration_loss(
#     pred: torch.Tensor,
#     footprint: torch.Tensor,
#     band: int = 2,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     """Penalise normal velocity at building walls.

#     Computes the wall normal from the gradient of the building mask,
#     then penalises the dot product (u, v) . n at wall pixels.

#     Args:
#         pred: [B, 2, T, H, W] velocity fields.
#         footprint: [B, H, W] where 1=building, 0=fluid.
#         band: wall mask dilation in pixels (thickens the wall region).
#     """
#     if footprint.dim() == 4:
#         footprint = footprint[:, 0]

#     b = footprint.float()

#     # Wall normal from building mask gradient
#     gx = F.pad(b[:, :, 1:] - b[:, :, :-1], (0, 1, 0, 0))
#     gy = F.pad(b[:, 1:, :] - b[:, :-1, :], (0, 0, 0, 1))
#     nrm = torch.sqrt(gx * gx + gy * gy + eps)
#     nx = gx / nrm
#     ny = gy / nrm

#     # Normal velocity component
#     u, v = pred[:, 0], pred[:, 1]
#     vn = u * nx[:, None] + v * ny[:, None]  # [B, T, H, W]

#     # Wall mask (pixels where gradient is nonzero)
#     wall_mask = (nrm > 0).float()
#     if band > 1:
#         x = wall_mask[:, None]
#         for _ in range(band - 1):
#             x = torch.maximum(x, F.max_pool2d(x, kernel_size=3, stride=1, padding=1))
#         wall_mask = x[:, 0]

#     return (vn.pow(2) * wall_mask[:, None]).mean()


# def physics_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     footprint: torch.Tensor,
#     *,
#     distance_alpha: float = 2.0,
#     distance_sigma: float = 20.0,
#     lambda_div: float = 10.0,
#     lambda_wall: float = 10.0,
#     wall_band: int = 2,
#     warmup_frames: int = 0,
# ) -> dict[str, torch.Tensor]:
#     """Combined physics-informed loss for VAE decoder finetuning.

#     L = L_data + lambda_div * L_div + lambda_wall * L_wall

#     Physics terms (divergence, wall) are only applied to frames after
#     warmup_frames, allowing the model to establish flow patterns before
#     enforcing physical constraints.

#     Args:
#         pred, target: [B, 2, T, H, W] velocity fields.
#         footprint: [B, H, W] where 1=building, 0=fluid.
#         warmup_frames: number of initial frames to skip for physics losses.

#     Returns:
#         dict with keys: total, data, div, wall.
#     """
#     # Flip convention: loss functions expect 1=building
#     fp = 1 - footprint

#     losses = {}
#     losses["data"] = distance_weighted_mse(pred, target, fp, alpha=distance_alpha, sigma=distance_sigma)

#     T = pred.shape[2]
#     zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

#     if lambda_div != 0.0 and T > warmup_frames:
#         losses["div"] = lambda_div * divergence_loss(pred[:, :, warmup_frames:], fp)
#     else:
#         losses["div"] = zero

#     if lambda_wall != 0.0 and T > warmup_frames:
#         losses["wall"] = lambda_wall * wall_no_penetration_loss(pred[:, :, warmup_frames:], fp, band=wall_band)
#     else:
#         losses["wall"] = zero

#     losses["total"] = losses["data"] + losses["div"] + losses["wall"]
#     return losses


# @torch.no_grad()
# def vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
#     """Variance-normalised RMSE: sqrt(MSE / Var(target))."""
#     diff = (pred - target).float()
#     mse = (diff ** 2).mean(dim=(1, 2, 3, 4))
#     var = target.float().var(dim=(1, 2, 3, 4), unbiased=False)
#     return float(torch.sqrt(mse / (var + eps)).mean().item())
