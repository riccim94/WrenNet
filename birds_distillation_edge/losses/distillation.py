"""Knowledge distillation loss wrappers."""

from .distillation_loss import (
    AdaptiveDistillationLoss,
    DistillationLoss,
)
from .focal_loss import (
    AdaptiveFocalDistillationLoss,
    FocalDistillationLoss,
    FocalLoss,
)

__all__ = [
    "AdaptiveDistillationLoss",
    "DistillationLoss",
    "AdaptiveFocalDistillationLoss",
    "FocalDistillationLoss",
    "FocalLoss",
]
