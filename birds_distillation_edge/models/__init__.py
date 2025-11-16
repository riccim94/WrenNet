"""
Model architectures and building blocks for Birds Distillation Edge.
"""

from . import core as _core
from . import legacy_models as _legacy_models
from . import legacy_modules as _legacy_modules
from .core import *  # noqa: F401,F403
from .legacy_models import *  # noqa: F401,F403
from .legacy_modules import *  # noqa: F401,F403

def _public_symbols(module):
    if hasattr(module, "__all__"):
        return list(module.__all__)  # type: ignore[attr-defined]
    return [name for name in dir(module) if not name.startswith("_")]

__all__ = sorted(
    set(
        _public_symbols(_core)
        + _public_symbols(_legacy_models)
        + _public_symbols(_legacy_modules)
    )
)
