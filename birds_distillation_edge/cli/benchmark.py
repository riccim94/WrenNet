"""Benchmark CLI wrapper."""

from __future__ import annotations

import sys
from typing import Iterable


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv) if argv is not None else []
    sys.argv = [sys.argv[0], *args]
    from benchmark.run_benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
