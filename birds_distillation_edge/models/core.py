"""
Core model re-exports for Birds Distillation Edge.

The legacy `models.py` and `modules.py` files have been moved into this
package to ease the transition toward a fully namespaced layout.
"""

from .legacy_models import Improved_Phi_GRU_ATT  # noqa: F401
from .legacy_modules import (  # noqa: F401
    AttentionLayer,
    FocusedAttention,
    HighwayGRU,
    LightConsonantEnhancer,
    MatchboxNetSkip,
    StatefulGRU,
    StatefulRNNLayer,
)

__all__ = [
    "Improved_Phi_GRU_ATT",
    "AttentionLayer",
    "FocusedAttention",
    "HighwayGRU",
    "LightConsonantEnhancer",
    "MatchboxNetSkip",
    "StatefulGRU",
    "StatefulRNNLayer",
]
