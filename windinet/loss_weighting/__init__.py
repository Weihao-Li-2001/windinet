"""
Loss weighting strategies.

Available strategies:
    - FixedWeighting
    - GradNorm (future)
    - UncertaintyWeighting (future)
"""


from .base import LossWeightingStrategy
from .fixed import FixedWeighting
from .gradnorm import GradNorm
from .soft_adapt import SoftAdapt

__all__ = [
    "LossWeightingStrategy",
    "FixedWeighting",
    "GradNorm",
    "SoftAdapt",
]