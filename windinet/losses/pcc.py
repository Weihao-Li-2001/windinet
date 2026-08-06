"""
Pearson correlation coefficient (PCC) loss.

Measures linear correlation between prediction and target, per sample,
averaged over the batch -- complementary to RMSE: two fields can have low
RMSE (close everywhere, but structurally uncorrelated noise) or, more
relevantly here, high correlation (same spatial pattern) with a magnitude
offset that RMSE alone penalizes harshly but PCC ignores. Useful as a
sanity check that the model captures spatial *structure*, not just mean
magnitude.

Expected input:
    pred, target: [B, C, T, H, W]

Loss = mean over batch of (1 - PCC), so it is 0 when perfectly correlated
and up to 2 when perfectly anti-correlated.
"""

import torch


def pcc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute 1 - Pearson correlation coefficient, per sample, averaged over
    the batch.

    Each sample's (C, T, H, W) is flattened into one vector before
    computing its correlation with the corresponding target vector --
    mirrors how vrmse (windinet.losses.vrms) reduces per sample before
    averaging over the batch.

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
        Scalar PCC loss.
    """

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)

    covariance = (pred_centered * target_centered).sum(dim=1)
    pred_norm = pred_centered.square().sum(dim=1).sqrt()
    target_norm = target_centered.square().sum(dim=1).sqrt()

    pcc = covariance / (pred_norm * target_norm + eps)

    return (1.0 - pcc).mean()
