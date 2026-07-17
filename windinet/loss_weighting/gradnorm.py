"""
GradNorm loss weighting.

Reference:
    GradNorm: Gradient Normalization for Adaptive Loss Balancing
    (Chen et al., 2018)

Epoch-level implementation.

Maintains dynamic weights for multiple loss components:

    L = sum_i w_i * L_i
"""


from typing import Dict, Optional

import torch

from .base import LossWeightingStrategy



class GradNorm(LossWeightingStrategy):
    """
    Adaptive loss weighting using GradNorm.
    """


    def __init__(
        self,
        loss_names: list[str],
        alpha: float = 1.5,
        weight_lr: float = 0.025,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
        epsilon: float = 1e-12,
    ):
        """
        Args:

            loss_names:
                Controlled loss components.

            alpha:
                Strength of inverse training rate balancing.

            weight_lr:
                Learning rate for weight update.

        """

        super().__init__(
            loss_names=loss_names
        )


        self.alpha = alpha
        self.weight_lr = weight_lr

        self.min_weight = min_weight
        self.max_weight = max_weight

        self.epsilon = epsilon


        # L_i(0)
        self.initial_losses = {}


        # Current weights
        n = len(loss_names)

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



    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update GradNorm weights.

        Args:

            losses:
                Current unweighted losses.

            grad_norms:
                Gradient norms:

                {
                    loss_name: ||grad(w_i L_i)||
                }

        """

        if grad_norms is None:
            return


        keys = [
            k for k in self.loss_names
            if k in losses and k in grad_norms
        ]


        if len(keys) < 2:
            return



        # --------------------------------------------------
        # Initialize L_i(0)
        # --------------------------------------------------

        for k in keys:

            if k not in self.initial_losses:

                self.initial_losses[k] = max(
                    float(losses[k]),
                    self.epsilon,
                )



        # --------------------------------------------------
        # Relative inverse training rate
        #
        # r_i =
        # (L_i/L_i0) /
        # mean(L_j/L_j0)
        # --------------------------------------------------

        progress = {}

        for k in keys:

            progress[k] = (
                float(losses[k])
                /
                self.initial_losses[k]
            )


        mean_progress = sum(
            progress.values()
        ) / len(progress)


        if mean_progress < self.epsilon:
            return


        relative_rate = {

            k:
            progress[k] / mean_progress

            for k in keys
        }



        # --------------------------------------------------
        # Gradient target
        #
        # G_i* = G_avg * r_i^alpha
        # --------------------------------------------------

        G = {

            k:
            max(
                float(grad_norms[k]),
                self.epsilon,
            )

            for k in keys
        }


        G_avg = sum(
            G.values()
        ) / len(G)


        target = {

            k:
            G_avg *
            (
                relative_rate[k]
                **
                self.alpha
            )

            for k in keys
        }



        # --------------------------------------------------
        # Update weights
        #
        # minimize:
        #
        # sum |w_i*g_i-G_i*|
        #
        # gradient:
        #
        # sign(w_i*g_i-G_i*)*g_i
        #
        # --------------------------------------------------

        for k in keys:

            w = self.weights[k]

            grad = (
                torch.sign(
                    torch.tensor(
                        w * G[k]
                        -
                        target[k]
                    )
                )
                *
                G[k]
            )


            new_w = (
                w
                -
                self.weight_lr
                *
                float(grad)
            )


            self.weights[k] = max(
                self.min_weight,
                min(
                    self.max_weight,
                    new_w,
                ),
            )



        # --------------------------------------------------
        # Normalize:
        #
        # sum(w_i)=N
        #
        # --------------------------------------------------

        weight_sum = sum(
            self.weights[k]
            for k in keys
        )


        if weight_sum > self.epsilon:

            scale = len(keys) / weight_sum


            for k in keys:

                self.weights[k] *= scale