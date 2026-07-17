"""
Fixed loss weighting strategy.

Uses manually specified constant weights.

This serves as the baseline strategy:

    L = sum_i w_i * L_i

where w_i remain unchanged during training.
"""


from typing import Dict

from .base import LossWeightingStrategy



class FixedWeighting(LossWeightingStrategy):
    """
    Constant loss weighting.

    Example:

        weights = {
            "rmse": 1.0,
            "h1": 0.5,
            "ssim": 0.2,
            "mlw": 0.05,
        }
    """


    def __init__(
        self,
        weights: Dict[str, float],
    ):
        """
        Args:
            weights:
                Dictionary mapping loss names to fixed weights.
        """

        super().__init__(
            loss_names=list(weights.keys())
        )

        self.weights = {
            k: float(v)
            for k, v in weights.items()
        }


    def get_weights(self) -> Dict[str, float]:
        """
        Return current fixed weights.
        """

        return self.weights.copy()


    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float] | None = None,
    ) -> None:
        """
        Fixed weighting does not update.

        Arguments are accepted for interface compatibility.
        """

        return None