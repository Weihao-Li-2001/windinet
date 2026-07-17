"""
SoftAdapt loss weighting.

Adaptive loss weighting based on the relative
change rate of each loss component.

Reference:
    SoftAdapt: Techniques for Adaptive Loss Weighting
    of Neural Networks with Multi-Part Loss Functions
"""


from typing import Dict, Optional

import torch

from .base import LossWeightingStrategy



class SoftAdapt(LossWeightingStrategy):
    """
    SoftAdapt adaptive weighting.

    Losses with slower improvement receive larger weights.

    Example:

        losses =
        {
            "rmse": value,
            "h1": value,
            "ssim": value,
            "mlw": value,
        }

    """

    def __init__(
        self,
        loss_names: list[str],
        temperature: float = 0.1,
        epsilon: float = 1e-8,
        normalize: bool = True,
    ):
        """
        Args:

            loss_names:
                Controlled loss components.

            temperature:
                Softmax temperature.

            epsilon:
                Numerical stability.

            normalize:
                If True:

                    sum(weights)=number of losses

                Otherwise:

                    sum(weights)=1

        """

        super().__init__(
            loss_names=loss_names
        )


        self.temperature = float(
            temperature
        )

        self.epsilon = float(
            epsilon
        )

        self.normalize = normalize


        # Store previous epoch losses

        self.previous_losses: Dict[str, float] = {}


        # Current weights

        self.weights = {

            name: 1.0

            for name in loss_names
        }



    def get_weights(
        self,
    ) -> Dict[str, float]:
        """
        Return current weights.
        """

        return self.weights.copy()



    @staticmethod
    def _stable_softmax(
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x - x.max()

        exp_x = torch.exp(x)

        return exp_x / (
            exp_x.sum()
            +
            1e-12
        )



    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Optional[
            Dict[str, float]
        ] = None,
    ) -> None:
        """
        Update SoftAdapt weights.

        grad_norms is ignored.
        It is only included for interface compatibility
        with GradNorm.

        """

        current_losses = {

            k:
            float(losses[k])

            for k in self.loss_names

            if k in losses
        }



        # First epoch:
        # only initialize history

        if len(self.previous_losses) == 0:

            self.previous_losses = (
                current_losses.copy()
            )

            return



        rates = {}


        for k in current_losses:


            if k not in self.previous_losses:
                continue


            previous = max(
                abs(
                    self.previous_losses[k]
                ),
                self.epsilon,
            )


            current = current_losses[k]


            # Relative loss change
            #
            # negative:
            # loss decreases
            #
            # positive:
            # loss increases

            rates[k] = (
                current - self.previous_losses[k]
            ) / previous



        if len(rates) == 0:
            return



        rate_tensor = torch.tensor(
            [
                rates[k]
                for k in rates
            ],
            dtype=torch.float32,
        )



        # Larger rate -> larger weight
        #
        # Meaning:
        # loss decreasing slowly
        # receives more attention

        weights = self._stable_softmax(
            rate_tensor
            /
            self.temperature
        )



        if self.normalize:

            weights *= len(weights)



        for idx, k in enumerate(rates):

            self.weights[k] = float(
                weights[idx]
            )



        # Update history

        self.previous_losses = (
            current_losses.copy()
        )