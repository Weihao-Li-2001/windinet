"""
Multi-Level Wavelet Loss (MLW).

The loss compares wavelet coefficients between
prediction and target to preserve high-frequency
structures.

Expected input:
    pred, target: [B, C, T, H, W]

Reference:
    Wavelet-Based Loss for High-Frequency Interface Dynamics
"""

import torch
import ptwt


def _wavelet_loss_spatial(
    pred: torch.Tensor,
    target: torch.Tensor,
    wavelet: str = "db2",
    level: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Spatial multilevel wavelet loss.

    Wavelet decomposition is applied over H/W dimensions.

    Args:
        pred:
            Prediction tensor [B,C,T,H,W]

        target:
            Target tensor [B,C,T,H,W]

    Returns:
        Spatial wavelet coefficient loss.
    """

    B, C, T, H, W = pred.shape


    # Treat each channel of each frame as one image
    pred = (
        pred
        .permute(0, 2, 1, 3, 4)
        .reshape(B * T * C, H, W)
    )

    target = (
        target
        .permute(0, 2, 1, 3, 4)
        .reshape(B * T * C, H, W)
    )


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


    loss = torch.tensor(
        0.0,
        device=pred.device,
        dtype=pred.dtype,
    )


    # Ignore approximation coefficients.
    # Only compare high-frequency details.
    for pred_detail, target_detail in zip(
        pred_coeffs[1:],
        target_coeffs[1:],
    ):

        for pred_band, target_band in zip(
            pred_detail,
            target_detail,
        ):

            loss += torch.mean(
                torch.abs(
                    torch.log2(pred_band.abs() + eps)
                    -
                    torch.log2(target_band.abs() + eps)
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

    Wavelet decomposition is applied along
    the time dimension.

    Args:
        pred:
            Prediction tensor [B,C,T,H,W]

        target:
            Target tensor [B,C,T,H,W]

    Returns:
        Temporal wavelet coefficient loss.
    """


    pred_coeffs = ptwt.wavedec(
        pred,
        wavelet=wavelet,
        level=level,
        axis=2,
    )

    target_coeffs = ptwt.wavedec(
        target,
        wavelet=wavelet,
        level=level,
        axis=2,
    )


    loss = torch.tensor(
        0.0,
        device=pred.device,
        dtype=pred.dtype,
    )


    # Ignore low-frequency approximation
    for pred_detail, target_detail in zip(
        pred_coeffs[1:],
        target_coeffs[1:],
    ):

        loss += torch.mean(
            torch.abs(
                torch.log2(pred_detail.abs() + eps)
                -
                torch.log2(target_detail.abs() + eps)
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

    Combines spatial and temporal wavelet losses:

        L_MLW = L_spatial + beta * L_temporal


    Args:
        pred:
            Prediction tensor [B,C,T,H,W]

        target:
            Ground truth tensor [B,C,T,H,W]

        wavelet:
            Wavelet basis.

        beta:
            Temporal loss weight.

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