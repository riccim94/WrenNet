"""Compatibility shim for legacy module imports.

Provides access to the building blocks now hosted inside
``birds_distillation_edge.models.legacy_modules``.
"""

from birds_distillation_edge.models.legacy_modules import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
