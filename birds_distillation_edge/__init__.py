"""Birds Distillation Edge package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("birds_distillation_edge")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__", "BirdsDataModule"]


def __getattr__(name: str):
    if name == "BirdsDataModule":
        from .datamodule import BirdsDataModule as _BirdsDataModule

        return _BirdsDataModule
    raise AttributeError(f"module {__name__} has no attribute {name!r}")
