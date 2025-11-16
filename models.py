"""Compatibility shim for legacy imports.

Re-exports symbols from ``birds_distillation_edge.models`` while the
codebase transitions to the new package structure.
"""

from birds_distillation_edge.models import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
