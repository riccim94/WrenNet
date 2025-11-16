"""Loss utilities for Birds Distillation Edge."""

from .distillation_loss import (
    DistillationLoss,
    AdaptiveDistillationLoss,
)
from .focal_loss import (
    FocalLoss,
    FocalDistillationLoss,
    AdaptiveFocalDistillationLoss,
)

__all__ = [
    "DistillationLoss",
    "AdaptiveDistillationLoss",
    "FocalLoss",
    "FocalDistillationLoss",
    "AdaptiveFocalDistillationLoss",
]
