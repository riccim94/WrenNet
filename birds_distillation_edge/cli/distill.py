"""Convenience CLI for distillation training."""

from __future__ import annotations

from typing import Iterable

from .train import main as train_main

__all__ = ["main"]


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv) if argv is not None else []
    has_config = any(arg.startswith("--config-name") for arg in args)
    if not has_config:
        args = ["--config-name", "distillation", *args]
    train_main(args)


if __name__ == "__main__":
    main()
