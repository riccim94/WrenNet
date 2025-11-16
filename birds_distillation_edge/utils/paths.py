"""Path utilities for Birds Distillation Edge."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_timestamped_dir(base_dir: Path, prefix: str | None = None) -> Path:
    ensure_dir(base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if prefix:
        name = f"{prefix}_{timestamp}"
    else:
        name = timestamp
    target = base_dir / name
    return ensure_dir(target)


def create_subdirs(root: Path, subdirs: Iterable[str]) -> dict[str, Path]:
    root = ensure_dir(root)
    return {name: ensure_dir(root / name) for name in subdirs}


def _normalise_path(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value.expanduser().resolve()
    return Path(value).expanduser().resolve()


def resolve_data_path(
    config_value: str | Path | None,
    *,
    env_var: str | None = None,
    fallbacks: Sequence[str | Path] | None = None,
) -> Tuple[Path, bool, list[Path]]:
    """
    Resolve a dataset path by combining config value, optional environment variable and additional fallbacks.

    Returns a tuple of (resolved_path, exists_flag, tried_candidates).
    """

    candidates: list[Path] = []

    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            candidates.append(_normalise_path(env_value))

    if config_value:
        candidates.append(_normalise_path(config_value))

    if fallbacks:
        candidates.extend(_normalise_path(fb) for fb in fallbacks if fb)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        unique_candidates.append(candidate)
        seen.add(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate, True, unique_candidates

    fallback = unique_candidates[0] if unique_candidates else _normalise_path(Path.cwd())
    return fallback, False, unique_candidates


__all__ = [
    "ensure_dir",
    "make_timestamped_dir",
    "create_subdirs",
    "resolve_data_path",
]
