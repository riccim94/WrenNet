"""Soft label extraction CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from tqdm import tqdm

from ..utils import configure_logging, get_logger, ensure_dir, make_timestamped_dir

log = get_logger(__name__)


class SoftLabelExtractor:
    """Generate soft labels from BirdNET predictions for target species."""

    def __init__(
        self,
        species_list_path: Path,
        confidence_threshold: float = 0.05,
    ) -> None:
        self.species_list_path = species_list_path
        self.confidence_threshold = confidence_threshold
        self.analyzer = Analyzer(custom_species_list_path=str(species_list_path))
        self.target_species = self._load_target_species(species_list_path)
        self.species_to_idx = {species: idx for idx, species in enumerate(self.target_species)}
        self.num_classes = len(self.target_species)
        log.info("Loaded %d target species", self.num_classes)

    def _load_target_species(self, species_list_path: Path) -> list[str]:
        dataset_path = species_list_path.parent / ".." / "bird_sound_dataset_processed"
        dataset_path = dataset_path.resolve()
        actual_species: list[str] = []
        if dataset_path.exists():
            log.info("Scanning dataset for species directories: %s", dataset_path)
            for item in dataset_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    actual_species.append(item.name.replace('_', ' '))
        actual_species.append("non-bird")
        return actual_species

    def extract_soft_labels(self, audio_path: Path) -> np.ndarray:
        try:
            recording = Recording(self.analyzer, str(audio_path), min_conf=self.confidence_threshold)
            recording.analyze()
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("Failed to analyze %s: %s", audio_path, exc)
            return self._default_soft_labels(error=True)

        soft_labels = np.zeros(self.num_classes, dtype=np.float32)
        non_bird_idx = self.species_to_idx.get("non-bird")
        found_target = False

        for detection in recording.detections:
            species = detection['scientific_name']
            confidence = detection['confidence']
            if species in self.species_to_idx:
                idx = self.species_to_idx[species]
                soft_labels[idx] = max(soft_labels[idx], confidence)
                found_target = True
            elif non_bird_idx is not None:
                soft_labels[non_bird_idx] = max(soft_labels[non_bird_idx], confidence * 0.1)

        if not found_target and recording.detections and non_bird_idx is not None:
            soft_labels[non_bird_idx] = max(soft_labels[non_bird_idx], 0.3)
        if not recording.detections and non_bird_idx is not None:
            soft_labels[non_bird_idx] = max(soft_labels[non_bird_idx], 0.1)
        return soft_labels

    def _default_soft_labels(self, error: bool = False) -> np.ndarray:
        labels = np.zeros(self.num_classes, dtype=np.float32)
        non_bird_idx = self.species_to_idx.get("non-bird")
        if non_bird_idx is not None:
            labels[non_bird_idx] = 0.1 if error else 0.0
        return labels

    def process_dataset(
        self,
        dataset_path: Path,
        output_path: Path,
        max_files_per_class: Optional[int] = None,
    ) -> None:
        ensure_dir(output_path)
        soft_labels_data: dict[str, list[float]] = {}
        failed_files: list[str] = []

        for species_dir in dataset_path.iterdir():
            if not species_dir.is_dir() or species_dir.name.startswith('.'):
                continue
            files = [f for f in species_dir.iterdir() if f.is_file()]
            if max_files_per_class:
                files = files[:max_files_per_class]
            for file_path in tqdm(files, desc=f"Processing {species_dir.name}"):
                soft = self.extract_soft_labels(file_path)
                if soft is None:
                    failed_files.append(str(file_path))
                    continue
                soft_labels_data[file_path.stem] = soft.tolist()

        metadata = {
            "num_classes": self.num_classes,
            "target_species": self.target_species,
            "confidence_threshold": self.confidence_threshold,
        }

        (output_path / "soft_labels.json").write_text(json.dumps(soft_labels_data, indent=2))
        (output_path / "soft_labels_metadata.json").write_text(json.dumps(metadata, indent=2))
        if failed_files:
            log.warning("Failed to process %d files", len(failed_files))
            (output_path / "failed_files.txt").write_text("\n".join(failed_files))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract soft labels using BirdNET")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory")
    parser.add_argument("output", type=Path, help="Output directory for soft labels")
    parser.add_argument(
        "--species-list",
        type=Path,
        default=Path("distillation/species.txt"),
        help="Path to species.txt",
    )
    parser.add_argument("--confidence", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--max-files", type=int, default=None, help="Max files per class (debug)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    extractor = SoftLabelExtractor(args.species_list, args.confidence)
    output_dir = make_timestamped_dir(args.output)
    extractor.process_dataset(args.dataset, output_dir, args.max_files)


if __name__ == "__main__":
    main()
