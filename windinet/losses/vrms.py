"""
Variance-normalized RMSE (VRMS) loss.

Same metric already used for evaluation (windinet.training.vae_trainer.vrmse,
"val_vrmse" in metrics.csv), exposed here as a differentiable loss so it can
optionally be optimized directly instead of only monitored. Normalizing by
the target's own variance means a channel with small dynamic range isn't
automatically "cheap" to get right the way plain RMSE would let it be.

Expected input:
    pred, target: [B, C, T, H, W]
"""

import torch


def vrms_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute variance-normalized RMSE, per sample, averaged over the batch.

    VRMSE = sqrt(mean((pred - target)^2) / var(target))

    Args:
        pred:
            Predicted fields.
            Shape [B, C, T, H, W]

        target:
            Ground truth fields.
            Shape [B, C, T, H, W]

        eps:
            Numerical stability term.

    Returns:
        Scalar VRMS loss.
    """

    dims = tuple(range(1, pred.dim()))
    diff = pred - target
    mse = diff.square().mean(dim=dims)
    variance = target.var(dim=dims, unbiased=False)

    return torch.sqrt(mse / (variance + eps)).mean()
