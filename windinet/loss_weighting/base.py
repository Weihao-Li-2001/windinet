"""
Base interface for loss weighting strategies.

A weighting strategy receives individual loss components:

{
    "rmse": loss_rmse,
    "h1": loss_h1,
    "ssim": loss_ssim,
    "mlw": loss_mlw,
}

and produces weights:

{
    "rmse": w_rmse,
    "h1": w_h1,
    "ssim": w_ssim,
    "mlw": w_mlw,
}

The final objective:

    L = sum_i w_i * L_i

is assembled by the trainer.
"""

from abc import ABC, abstractmethod
from typing import Dict


class LossWeightingStrategy(ABC):
    """
    Abstract base class for adaptive loss weighting.
    """


    def __init__(
        self,
        loss_names: list[str],
    ):
        """
        Args:
            loss_names:
                Names of controlled loss components.
        """

        self.loss_names = loss_names


    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """
        Return current loss weights.

        Returns:
            {
                loss_name: weight
            }
        """
        pass


    @abstractmethod
    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float] | None = None,
    ) -> None:
        """
        Update loss weights.

        Args:
            losses:
                Current loss values.

                Example:
                {
                    "rmse": 0.2,
                    "h1": 3.1
                }


            grad_norms:
                Optional gradient statistics.

                Required by methods such as GradNorm.
        """
        pass