"""Command-line entry points for Birds Distillation Edge."""

import importlib

__all__ = ["train", "distill", "benchmark", "extract_soft_labels"]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__} has no attribute {name!r}")
