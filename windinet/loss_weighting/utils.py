"""
Utilities for adaptive loss weighting.

Contains gradient statistics required by
methods such as GradNorm.
"""


from typing import Dict, Iterable

import torch
import torch.nn as nn



def compute_grad_norms(
    losses: Dict[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute gradient norm for each loss component.

    For each loss:

        G_i = || grad_theta(L_i) ||_2


    Args:
        losses:
            Individual loss components.

            Example:

            {
                "rmse": loss_rmse,
                "h1": loss_h1,
                "ssim": loss_ssim,
                "mlw": loss_mlw,
            }


        parameters:
            Trainable model parameters.

            Usually decoder.parameters()


        epsilon:
            Numerical stability.


    Returns:

        {
            "rmse": gradient norm,
            "h1": gradient norm,
            ...
        }

    """


    # Convert generator to list because
    # autograd.grad needs reusable parameters

    parameters = [
        p for p in parameters
        if p.requires_grad
    ]


    if len(parameters) == 0:
        raise ValueError(
            "No trainable parameters provided."
        )


    grad_norms = {}



    for name, loss in losses.items():

        if not torch.is_tensor(loss):
            raise TypeError(
                f"Loss {name} is not a torch.Tensor."
            )


        gradients = torch.autograd.grad(
            outputs=loss,
            inputs=parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )


        total_norm = torch.tensor(
            0.0,
            device=loss.device,
        )


        for grad in gradients:

            if grad is not None:

                total_norm += (
                    grad.detach()
                    .norm(2)
                    ** 2
                )


        total_norm = torch.sqrt(
            total_norm + epsilon
        )


        grad_norms[name] = (
            float(total_norm.item())
        )


    return grad_norms