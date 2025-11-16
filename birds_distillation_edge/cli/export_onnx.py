"""Utility to export a trained Lightning checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import torchaudio

from ..experiment import BirdsExperiment
from ..utils import ensure_dir
from ..models.export_wrappers import PostMelExportWrapper
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _load_config(config_name: str, overrides: Sequence[str]) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the Lightning checkpoint (.ckpt) to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports") / "model.onnx",
        help="Destination ONNX file (default: exports/model.onnx).",
    )
    parser.add_argument(
        "--config-name",
        default="base",
        help="Hydra config file name (without .yaml).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra-style overrides (e.g. data.batch_size=8).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use (default: 17).",
    )
    parser.add_argument(
        "--input-format",
        choices=("waveform", "mel"),
        default="waveform",
        help=(
            "Export either the full waveform model or the post-mel variant. "
            "Use 'mel' to avoid torchaudio operators in the ONNX graph."
        ),
    )
    return parser.parse_args()


def _infer_num_classes_from_checkpoint(checkpoint_path: Path) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    fc_weight = state_dict.get("model.fc.weight")
    if isinstance(fc_weight, torch.Tensor) and fc_weight.ndim == 2:
        return int(fc_weight.shape[0])
    return None


def _load_experiment(cfg, checkpoint_path: Path) -> BirdsExperiment:
    inferred_classes = _infer_num_classes_from_checkpoint(checkpoint_path)
    experiment: BirdsExperiment = BirdsExperiment.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        num_classes_override=inferred_classes,
    )
    experiment.eval()
    experiment.to("cpu")
    return experiment


def _build_dummy_input(cfg) -> torch.Tensor:
    sample_rate = int(cfg.data.sample_rate)
    clip_duration = float(cfg.data.clip_duration)
    num_samples = max(1, int(sample_rate * clip_duration))
    return torch.zeros(1, 1, num_samples, dtype=torch.float32)


def _compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    params,
) -> torch.Tensor:
    n_fft = int(getattr(params, "n_fft", 400))
    hop_length = int(getattr(params, "hop_length", 160))
    n_mels = int(getattr(params, "n_mel_bins", 64))
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    mel = mel_spec(waveform)
    mel = amplitude_to_db(mel)
    return mel


def export_onnx(
    cfg: DictConfig,
    checkpoint_path: Path,
    output_path: Path,
    opset: int,
    input_format: str,
) -> None:
    experiment = _load_experiment(cfg, checkpoint_path)
    model = experiment.model.eval()
    ensure_dir(output_path.parent)

    if input_format == "waveform":
        dummy_input = _build_dummy_input(cfg)
        input_names = ["input_waveform"]
        dynamic_axes = {"input_waveform": {0: "batch", 2: "time"}, "logits": {0: "batch"}}
        export_model = model
    else:
        dummy_waveform = _build_dummy_input(cfg)
        sample_rate = int(cfg.data.sample_rate)
        log_mel = _compute_log_mel(dummy_waveform, sample_rate, cfg.model.params)
        wrapper = PostMelExportWrapper(model)
        dummy_input = log_mel
        input_names = ["log_mel"]
        dynamic_axes = {"log_mel": {0: "batch", 3: "time"}, "logits": {0: "batch"}}
        export_model = wrapper

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    print(f"Exported ONNX model to {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args()
    cfg = _load_config(args.config_name, args.overrides)
    struct_original = OmegaConf.is_struct(cfg.model.params)
    if struct_original:
        OmegaConf.set_struct(cfg.model.params, False)
    inferred_classes = _infer_num_classes_from_checkpoint(args.checkpoint.resolve())
    if inferred_classes is None:
        cfg.model.params.pop("num_classes", None)
    else:
        cfg.model.params.num_classes = inferred_classes
    if struct_original:
        OmegaConf.set_struct(cfg.model.params, True)
    export_onnx(
        cfg,
        args.checkpoint.resolve(),
        args.output.resolve(),
        args.opset,
        args.input_format,
    )


if __name__ == "__main__":
    main()
