"""
Physics-informed loss functions for VAE decoder finetuning.

Three loss terms enforce physical consistency of decoded wind fields:
  - Distance-weighted MSE: reconstruction loss emphasising near-wall regions
  - Divergence loss: incompressibility constraint (du/dx + dv/dy ~ 0)
  - Wall no-penetration loss: zero normal velocity at building boundaries

All loss functions expect:
  - pred, target: [B, 2, T, H, W] (u, v velocity channels)
  - footprint: [B, H, W] where 1=building, 0=fluid
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _distance_weight_map(fluid_mask: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """Compute w(x) = 1 + alpha * exp(-d(x)^2 / (2*sigma^2)) in fluid, 0 in buildings.

    fluid_mask: [B, H, W] with 1=fluid, 0=building.
    """
    w_maps = []
    fm_cpu = fluid_mask.detach().to("cpu")
    for i in range(fm_cpu.shape[0]):
        fm = fm_cpu[i].numpy().astype(bool)
        d = distance_transform_edt(fm)
        w = 1.0 + alpha * np.exp(-(d * d) / (2.0 * sigma * sigma))
        w *= fm.astype(np.float32)
        w_maps.append(torch.from_numpy(w))
    return torch.stack(w_maps, dim=0).to(fluid_mask.device)


def distance_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    alpha: float = 2.0,
    sigma: float = 20.0,
) -> torch.Tensor:
    """MSE weighted by distance to buildings, zero inside buildings.

    Pixels near building boundaries receive approximately (1 + alpha) times
    the weight of far-field pixels, with a Gaussian falloff controlled by sigma.

    Args:
        pred, target: [B, 2, T, H, W] velocity fields.
        footprint: [B, H, W] where 1=building, 0=fluid.
        alpha: wall proximity weight amplification.
        sigma: Gaussian falloff distance in pixels.
    """
    fluid = (1.0 - footprint.float()).clamp(0, 1)
    w = _distance_weight_map(fluid, alpha=alpha, sigma=sigma)  # [B, H, W]
    loss = (pred - target).pow(2) * w[:, None, None]  # broadcast over C and T
    return loss.mean()


def divergence_loss(pred: torch.Tensor, footprint: torch.Tensor) -> torch.Tensor:
    """Penalise violations of incompressibility (du/dx + dv/dy ~ 0).

    Uses forward finite differences. The loss is masked so that only
    2x2 stencils entirely within fluid contribute.

    Args:
        pred: [B, 2, T, H, W] velocity fields (channel 0=u, 1=v).
        footprint: [B, H, W] where 1=building, 0=fluid.
    """
    if footprint.dim() == 4:
        footprint = footprint[:, 0]

    f = (1.0 - footprint.float()).clamp(0, 1)  # 1=fluid

    u = pred[:, 0]  # [B, T, H, W]
    v = pred[:, 1]

    du_dx = F.pad(u[..., :, 1:] - u[..., :, :-1], (0, 1, 0, 0))
    dv_dy = F.pad(v[..., 1:, :] - v[..., :-1, :], (0, 0, 0, 1))
    div = du_dx + dv_dy

    # Valid mask: all corners of the 2x2 stencil must be fluid
    f00 = f
    f01 = F.pad(f[:, :, 1:], (0, 1, 0, 0))
    f10 = F.pad(f[:, 1:, :], (0, 0, 0, 1))
    f11 = F.pad(f[:, 1:, 1:], (0, 1, 0, 1))
    mask = (f00 * f01 * f10 * f11)[:, None]  # [B, 1, H, W]

    return (div.pow(2) * mask).mean()


def wall_no_penetration_loss(
    pred: torch.Tensor,
    footprint: torch.Tensor,
    band: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalise normal velocity at building walls.

    Computes the wall normal from the gradient of the building mask,
    then penalises the dot product (u, v) . n at wall pixels.

    Args:
        pred: [B, 2, T, H, W] velocity fields.
        footprint: [B, H, W] where 1=building, 0=fluid.
        band: wall mask dilation in pixels (thickens the wall region).
    """
    if footprint.dim() == 4:
        footprint = footprint[:, 0]

    b = footprint.float()

    # Wall normal from building mask gradient
    gx = F.pad(b[:, :, 1:] - b[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(b[:, 1:, :] - b[:, :-1, :], (0, 0, 0, 1))
    nrm = torch.sqrt(gx * gx + gy * gy + eps)
    nx = gx / nrm
    ny = gy / nrm

    # Normal velocity component
    u, v = pred[:, 0], pred[:, 1]
    vn = u * nx[:, None] + v * ny[:, None]  # [B, T, H, W]

    # Wall mask (pixels where gradient is nonzero)
    wall_mask = (nrm > 0).float()
    if band > 1:
        x = wall_mask[:, None]
        for _ in range(band - 1):
            x = torch.maximum(x, F.max_pool2d(x, kernel_size=3, stride=1, padding=1))
        wall_mask = x[:, 0]

    return (vn.pow(2) * wall_mask[:, None]).mean()


def physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    footprint: torch.Tensor,
    *,
    distance_alpha: float = 2.0,
    distance_sigma: float = 20.0,
    lambda_div: float = 10.0,
    lambda_wall: float = 10.0,
    wall_band: int = 2,
    warmup_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Combined physics-informed loss for VAE decoder finetuning.

    L = L_data + lambda_div * L_div + lambda_wall * L_wall

    Physics terms (divergence, wall) are only applied to frames after
    warmup_frames, allowing the model to establish flow patterns before
    enforcing physical constraints.

    Args:
        pred, target: [B, 2, T, H, W] velocity fields.
        footprint: [B, H, W] where 1=building, 0=fluid.
        warmup_frames: number of initial frames to skip for physics losses.

    Returns:
        dict with keys: total, data, div, wall.
    """
    # Flip convention: loss functions expect 1=building
    fp = 1 - footprint

    losses = {}
    losses["data"] = distance_weighted_mse(pred, target, fp, alpha=distance_alpha, sigma=distance_sigma)

    T = pred.shape[2]
    zero = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    if lambda_div != 0.0 and T > warmup_frames:
        losses["div"] = lambda_div * divergence_loss(pred[:, :, warmup_frames:], fp)
    else:
        losses["div"] = zero

    if lambda_wall != 0.0 and T > warmup_frames:
        losses["wall"] = lambda_wall * wall_no_penetration_loss(pred[:, :, warmup_frames:], fp, band=wall_band)
    else:
        losses["wall"] = zero

    losses["total"] = losses["data"] + losses["div"] + losses["wall"]
    return losses


@torch.no_grad()
def vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Variance-normalised RMSE: sqrt(MSE / Var(target))."""
    diff = (pred - target).float()
    mse = (diff ** 2).mean(dim=(1, 2, 3, 4))
    var = target.float().var(dim=(1, 2, 3, 4), unbiased=False)
    return float(torch.sqrt(mse / (var + eps)).mean().item())
