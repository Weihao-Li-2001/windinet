"""
Factory for constructing loss weighting strategies.
"""


from .fixed import FixedWeighting
from .base import LossWeightingStrategy
from .gradnorm import GradNorm
from .soft_adapt import SoftAdapt


def build_loss_weighting(
    config,
) -> LossWeightingStrategy:
    """
    Build loss weighting strategy from configuration.

    Expected config example:

        strategy: fixed

        weights:
            rmse: 1.0
            h1: 0.5
            ssim: 0.2
            mlw: 0.05


    Args:
        config:
            Configuration object.

    Returns:
        Initialized loss weighting strategy.
    """


    strategy = config.strategy.lower()


    if strategy == "fixed":

        return FixedWeighting(
            weights=config.weights,
        )


    elif strategy == "gradnorm":

        return GradNorm(
            loss_names=config.loss_names,
            alpha=config.alpha,
            weight_lr=config.weight_lr,
        )
    

    elif strategy == "softadapt":

        return SoftAdapt(
            loss_names=config.loss_names,
            temperature=config.temperature,
        )

    elif strategy == "uncertainty":

        raise NotImplementedError(
            "Uncertainty weighting is not implemented yet."
        )


    else:

        raise ValueError(
            f"Unknown loss weighting strategy: {strategy}"
        )