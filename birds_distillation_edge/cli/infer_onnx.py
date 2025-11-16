"""Run inference with an exported ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from .train import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference helper for ONNX-exported models.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the ONNX file.")
    parser.add_argument("--audio-path", type=Path, required=True, help="Path to the audio file to classify.")
    parser.add_argument(
        "--config-name",
        default="base",
        help="Hydra config file name (without .yaml). Used to retrieve preprocessing settings.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra-style overrides (e.g. data.sample_rate=32000).",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of class names overriding the config.",
    )
    parser.add_argument(
        "--input-format",
        choices=("waveform", "mel"),
        default="waveform",
        help="Match this with the ONNX export (default: waveform).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many classes to display when printing probabilities (default: 5).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the probability vector as JSON.",
    )
    return parser.parse_args()


def load_audio(
    audio_path: Path,
    sample_rate: int,
    clip_duration: float,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    target_len = int(sample_rate * clip_duration)
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    elif waveform.shape[1] < target_len:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform.unsqueeze(0)


def build_class_names(cfg, logits_size: int, class_map_file: Path | None) -> list[str]:
    if class_map_file and class_map_file.exists():
        with class_map_file.open("r", encoding="utf-8") as handle:
            names = json.load(handle)
            if isinstance(names, list) and len(names) == logits_size:
                return [str(name) for name in names]
            print("WARNING: --class-map file does not match logits size. Falling back to config.")

    cfg_classes = list(cfg.data.allowed_bird_classes)
    if cfg_classes and len(cfg_classes) + 1 == logits_size:
        cfg_classes = cfg_classes + ["No_Birds"]
    if cfg_classes and len(cfg_classes) == logits_size:
        return cfg_classes
    return [f"class_{i}" for i in range(logits_size)]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)


def run_inference(
    model_path: Path,
    input_tensor: torch.Tensor,
) -> np.ndarray:
    session = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_tensor.numpy().astype(np.float32)})[0]
    return logits[0]


def compute_log_mel(
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args()
    cfg = load_config(args.config_name, args.overrides)

    sample_rate = int(cfg.data.sample_rate)
    clip_duration = float(cfg.data.clip_duration)

    audio_tensor = load_audio(args.audio_path, sample_rate, clip_duration)
    if args.input_format == "mel":
        audio_tensor = compute_log_mel(audio_tensor, sample_rate, cfg.model.params)
    logits = run_inference(args.model_path, audio_tensor)

    probs = softmax(logits)
    class_names = build_class_names(cfg, logits_size=probs.size, class_map_file=args.class_map)
    top_k = min(args.top_k, probs.size)
    top_indices = np.argsort(probs)[-top_k:][::-1]

    predicted_idx = int(np.argmax(probs))
    predicted_name = class_names[predicted_idx]
    print(f"Hard label: {predicted_name} (index {predicted_idx})")

    print("\nSoft labels (top probabilities):")
    for idx in top_indices:
        print(f"  {class_names[idx]:<20} {probs[idx]:.4f}")

    if args.save_json:
        payload = [{"class": name, "probability": float(prob)} for name, prob in zip(class_names, probs)]
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved soft labels to {args.save_json}")


if __name__ == "__main__":
    main()
