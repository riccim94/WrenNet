"""Logging helpers for Birds Distillation Edge."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level_name: str, default: int = logging.INFO) -> None:
    """Configure the root logger level."""
    level = getattr(logging, level_name.upper(), default)
    logging.getLogger().setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
