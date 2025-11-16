"""Wrapper script that lives next to the convert-and-quantize pipeline.

It simply forwards all the relevant arguments to the canonical
``birds_distillation_edge.cli.export_onnx`` entry-point so that the ONNX export
can be triggered from this directory (or via Docker helpers) without having to
remember the module path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birds_distillation_edge.cli import export_onnx as base_export_cli  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained Lightning checkpoint to ONNX using the "
        "convert-and-quantize helper."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the Lightning checkpoint (.ckpt) to export.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the ONNX file.",
    )
    parser.add_argument(
        "--config-name",
        default="base",
        help="Hydra config file to load (default: base).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to forward to the exporter.",
    )
    parser.add_argument(
        "--input-format",
        choices=("waveform", "mel"),
        default="mel",
        help="Input format expected by the exported model (default: mel).",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help=(
            "Optional Hydra-style overrides forwarded verbatim to "
            "birds_distillation_edge.cli.export_onnx (e.g. data.batch_size=8)."
        ),
    )
    return parser.parse_args()


def _sanitize_overrides(overrides: Sequence[str]) -> list[str]:
    items = list(overrides)
    if items and items[0] == "--":
        items = items[1:]
    return items


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    forwarded_args = [
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--config-name",
        args.config_name,
        "--opset",
        str(args.opset),
        "--input-format",
        args.input_format,
        * _sanitize_overrides(args.overrides),
    ]
    base_export_cli.main(forwarded_args)


if __name__ == "__main__":
    main()
