"""
Objective functions for inverse building layout optimization.

The default objective is Pedestrian Wind Comfort (PWC), which penalises
dangerous, uncomfortable, and stagnant wind conditions.

Extending with custom objectives
---------------------------------
To add a new objective, define a function with the following signature::

    def my_objective(u, v, mask, **kwargs) -> dict[str, torch.Tensor]:
        '''
        Args:
            u, v: (T, H, W) velocity components in m/s.
            mask: (H, W) spatial mask (1 = active fluid pixel).
            **kwargs: any additional parameters.

        Returns:
            dict with at least a "total" key (scalar loss to minimize)
            and any extra metric keys for logging.
        '''
        speed = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
        ...
        return {"total": loss, "my_metric": some_value}

Then replace the ``pwc_loss`` call in ``inverse/optimize.py`` (search for
"# --- objective ---") with your function. The optimizer will minimise
``out["total"]`` and log all other keys.

The helper ``masked_mean`` is available for computing spatial averages
restricted to fluid pixels.
"""

import torch


def masked_mean(x, mask, eps=1e-6):
    """Mean of *x* over spatial dims, restricted to *mask* > 0."""
    masked_x = x * mask
    return masked_x.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + eps)


def pwc_loss(
    u,
    v,
    mask,
    tau=1.0,
    danger_threshold=15.0,
    comfort_threshold=5.0,
    stagnation_threshold=0.5,
    w_danger=10.0,
    w_comfort=1.0,
    w_stagnation=0.3,
):
    """Pedestrian Wind Comfort (PWC) loss based on temporal exceedance.

    Three sigmoid-smoothed exceedance fractions penalise:
      - **Danger**: fraction of (pixel, time) where speed > danger_threshold
      - **Comfort**: fraction where speed > comfort_threshold
      - **Stagnation**: fraction where speed < stagnation_threshold

    Parameters
    ----------
    u, v : (T, H, W) or (B, T, H, W)
        Wind velocity components in m/s.
    mask : (H, W) or broadcastable
        Spatial mask (1 = active fluid pixel).
    tau : float
        Sigmoid temperature (lower = sharper).
    danger_threshold, comfort_threshold, stagnation_threshold : float
        Speed thresholds in m/s.
    w_danger, w_comfort, w_stagnation : float
        Loss weights for each term.

    Returns
    -------
    dict with keys: total, mean_speed, danger, comfort, stagnation.
    """
    while mask.dim() < u.dim():
        mask = mask.unsqueeze(0)

    speed = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
    mean_speed = masked_mean(speed, mask)

    e_danger = masked_mean(torch.sigmoid((speed - danger_threshold) / tau), mask)
    e_comfort = masked_mean(torch.sigmoid((speed - comfort_threshold) / tau), mask)
    e_stagnation = masked_mean(torch.sigmoid((stagnation_threshold - speed) / tau), mask)

    total = (
        w_danger * e_danger.mean()
        + w_comfort * e_comfort.mean()
        + w_stagnation * e_stagnation.mean()
    )

    return {
        "total": total,
        "mean_speed": mean_speed.mean(),
        "danger": e_danger.mean(),
        "comfort": e_comfort.mean(),
        "stagnation": e_stagnation.mean(),
    }
