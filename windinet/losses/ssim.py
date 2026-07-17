"""
Structural Similarity (SSIM) loss.

Expected input:
    pred, target: [B, C, T, H, W]

Returns:
    1 - SSIM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFilter(nn.Module):
    """
    Fixed Gaussian smoothing kernel used in SSIM.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 11,
        sigma: float = 1.5,
    ):
        super().__init__()

        coords = torch.arange(
            window_size,
            dtype=torch.float32,
        )

        coords -= window_size // 2

        gaussian_1d = torch.exp(
            -(coords ** 2) / (2 * sigma ** 2)
        )

        gaussian_1d /= gaussian_1d.sum()

        gaussian_2d = torch.outer(
            gaussian_1d,
            gaussian_1d,
        )

        gaussian_2d /= gaussian_2d.sum()

        kernel = gaussian_2d.expand(
            channels,
            1,
            window_size,
            window_size,
        )

        self.register_buffer(
            "kernel",
            kernel,
        )

        self.groups = channels
        self.padding = window_size // 2


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

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
    """
    Compute SSIM map.
    """

    mu_pred = gaussian(pred)
    mu_target = gaussian(target)


    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target


    sigma_pred_sq = (
        gaussian(pred * pred)
        - mu_pred_sq
    )

    sigma_target_sq = (
        gaussian(target * target)
        - mu_target_sq
    )

    sigma_pred_target = (
        gaussian(pred * target)
        - mu_pred_target
    )


    numerator = (
        (2 * mu_pred_target + c1)
        *
        (2 * sigma_pred_target + c2)
    )


    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        *
        (sigma_pred_sq + sigma_target_sq + c2)
    )


    return numerator / (denominator + 1e-8)



class SSIMLoss(nn.Module):
    """
    SSIM loss module.

    Loss = 1 - mean(SSIM)
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 11,
        sigma: float = 1.5,
    ):
        super().__init__()

        self.gaussian = GaussianFilter(
            channels=channels,
            window_size=window_size,
            sigma=sigma,
        )


    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        B, C, T, H, W = pred.shape


        # Treat each frame as an independent image
        pred = (
            pred
            .permute(0, 2, 1, 3, 4)
            .reshape(B * T, C, H, W)
        )

        target = (
            target
            .permute(0, 2, 1, 3, 4)
            .reshape(B * T, C, H, W)
        )


        ssim_map = _ssim(
            pred,
            target,
            self.gaussian,
        )


        return 1.0 - ssim_map.mean()