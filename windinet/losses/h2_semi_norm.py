"""
H2 semi-norm loss.

One order past h1_seminorm_loss's first derivatives: measures the
difference between second-order spatial derivatives (curvature) of
prediction and target. Where H1 penalizes gradient (edge/slope) mismatches,
H2 additionally penalizes curvature mismatches -- relevant for shock
fronts, where the second derivative is large and a pure first-derivative
penalty can still miss over-smoothed peaks.

Expected input:
    pred, target: [B, C, T, H, W]

The H2 seminorm is:

    |u|_H2^2 = ||del^2 u||^2

Here we minimize:

    ||del^2 u_pred - del^2 u_target||^2
"""

import torch
import torch.nn.functional as F


def spatial_second_derivatives(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute second-order spatial derivatives via central differences.

    Args:
        x:
            Tensor with shape [B, C, T, H, W]

    Returns:
        dxx:
            Second derivative along width direction.
        dyy:
            Second derivative along height direction.
    """

    # Central second difference: x[i-1] - 2*x[i] + x[i+1]
    dxx = x[..., :, :-2] - 2 * x[..., :, 1:-1] + x[..., :, 2:]
    dyy = x[..., :-2, :] - 2 * x[..., 1:-1, :] + x[..., 2:, :]

    # Keep original tensor size
    dxx = F.pad(
        dxx,
        (1, 1, 0, 0),
    )

    dyy = F.pad(
        dyy,
        (0, 0, 1, 1),
    )

    return dxx, dyy


def h2_seminorm_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute H2 semi-norm loss.

    Penalizes differences between second-order spatial derivatives
    (curvature) -- complements h1_seminorm_loss's first derivatives.

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

    Returns:
        Scalar H2 loss.
    """

    pred_dxx, pred_dyy = spatial_second_derivatives(pred)
    target_dxx, target_dyy = spatial_second_derivatives(target)

    loss_dxx = F.mse_loss(
        pred_dxx,
        target_dxx,
    )

    loss_dyy = F.mse_loss(
        pred_dyy,
        target_dyy,
    )

    return loss_dxx + loss_dyy
