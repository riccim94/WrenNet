"""
Lightning DataModule for Birds Distillation Edge.

This initial implementation focuses on the supervised training path by
leveraging the existing combined dataset factory to merge bird samples
with optional "no birds" segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import lightning as L
import logging
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
import torch

from .data.dataset_factory import create_combined_dataset
from .distillation.datasets import DistillationBirdSoundDataset
from .utils.paths import resolve_data_path

logger = logging.getLogger(__name__)


class BirdsDataModule(L.LightningDataModule):
    """
    Minimal Lightning DataModule wrapping the legacy dataset factory.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        data_cfg = cfg.data
        self.batch_size: int = int(data_cfg.batch_size)
        self.num_workers: int = int(data_cfg.num_workers)
        self.pin_memory: bool = bool(data_cfg.pin_memory)
        self.persistent_workers: bool = bool(data_cfg.get("persistent_workers", False))
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._metadata: Optional[DatasetMetadata] = None
        self.use_distillation: bool = bool(getattr(cfg, "distillation", {}).get("enabled", False))

        bird_dir, bird_exists, bird_candidates = resolve_data_path(
            data_cfg.get("bird_data_dir"),
            env_var="BDE_BIRD_DATASET_DIR",
            fallbacks=(
                Path.cwd() / "bird_sound_dataset",
                Path.cwd() / "data/bird_sound_dataset",
                Path("/app/bird_sound_dataset"),
            ),
        )
        esc50_dir, esc50_exists, esc50_candidates = resolve_data_path(
            data_cfg.get("esc50_dir"),
            env_var="BDE_ESC50_DIR",
            fallbacks=(
                Path.cwd() / "esc-50/ESC-50-master",
                Path.cwd() / "ESC-50-master",
                Path.cwd() / "data/esc50",
                Path("/app/ESC-50-master"),
            ),
        )
        no_birds_dir, no_birds_exists, no_birds_candidates = resolve_data_path(
            data_cfg.get("pregenerated_no_birds_dir"),
            env_var="BDE_NO_BIRDS_DIR",
            fallbacks=(
                Path.cwd() / "augmented_dataset/no_birds",
                Path.cwd() / "augmented_dataset",
                Path("/app/augmented_dataset/no_birds"),
            ),
        )

        self.bird_data_dir = bird_dir
        self.esc50_dir = esc50_dir
        self.pregenerated_no_birds_dir = no_birds_dir
        self._bird_dir_exists = bird_exists
        self._esc50_dir_exists = esc50_exists
        self._no_birds_dir_exists = no_birds_exists
        self._bird_candidates = bird_candidates
        self._esc50_candidates = esc50_candidates
        self._no_birds_candidates = no_birds_candidates

        self.cfg.data.bird_data_dir = str(self.bird_data_dir)
        self.cfg.data.esc50_dir = str(self.esc50_dir)
        self.cfg.data.pregenerated_no_birds_dir = str(self.pregenerated_no_birds_dir)

        if not self._bird_dir_exists:
            logger.warning(
                "Bird dataset directory not found. Tried: %s. Set BDE_BIRD_DATASET_DIR or adjust config.",
                ", ".join(str(path) for path in self._bird_candidates),
            )

        if not self._esc50_dir_exists:
            logger.warning(
                "ESC-50 directory not found. Tried: %s. Set BDE_ESC50_DIR or adjust config.",
                ", ".join(str(path) for path in self._esc50_candidates),
            )

        if bool(data_cfg.load_pregenerated_no_birds) and not self._no_birds_dir_exists:
            logger.warning(
                "Pregenerated no-birds directory not found. Tried: %s. Set BDE_NO_BIRDS_DIR or disable loading.",
                ", ".join(str(path) for path in self._no_birds_candidates),
            )

        if self.use_distillation:
            logger.info("Distillation data path enabled.")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepare datasets for the requested stage.
        """
        if stage in (None, "fit"):
            self._train_dataset = self._build_combined_dataset(subset="training")
            self._val_dataset = self._build_combined_dataset(subset="validation")

        if stage in (None, "test"):
            self._test_dataset = self._build_combined_dataset(subset="testing")

        self._metadata = self._extract_metadata(self._train_dataset)
        if self._metadata:
            self._metadata.train_samples = self._dataset_len(self._train_dataset)
            self._metadata.val_samples = self._dataset_len(self._val_dataset)
            self._metadata.test_samples = self._dataset_len(self._test_dataset)
            self._log_split_stats(self._metadata)

    def train_dataloader(self) -> DataLoader:
        return self._create_loader(self._train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_loader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_loader(self._test_dataset, shuffle=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_combined_dataset(self, subset: str) -> Dataset:
        data_cfg = self.cfg.data
        allowed_classes = list(data_cfg.allowed_bird_classes) or None
        num_no_bird_samples = data_cfg.get("num_no_bird_samples", 100)

        if self.use_distillation:
            try:
                return self._build_distillation_dataset(subset, allowed_classes, num_no_bird_samples)
            except Exception as exc:
                logger.warning("Falling back to supervised dataset: %s", exc)

        try:
            dataset = create_combined_dataset(
                bird_data_dir=data_cfg.bird_data_dir,
                esc50_dir=data_cfg.esc50_dir,
                allowed_bird_classes=allowed_classes,
                target_sr=data_cfg.sample_rate,
                clip_duration=data_cfg.clip_duration,
                subset=subset,
                num_no_bird_samples=num_no_bird_samples,
                esc50_no_bird_ratio=data_cfg.esc50_no_bird_ratio,
                use_augmentation=bool(data_cfg.get("use_augmentation", subset == "training")),
                load_pregenerated_no_birds=bool(data_cfg.load_pregenerated_no_birds),
                pregenerated_no_birds_dir=data_cfg.pregenerated_no_birds_dir,
                validation_split=data_cfg.validation_split,
                test_split=data_cfg.test_split,
                split_seed=self.cfg.project.seed,
                empty_segment_cache_path=data_cfg.get("empty_segment_cache_path"),
                refresh_empty_segment_cache=bool(data_cfg.get("refresh_empty_segment_cache", False)),
            )
            return dataset
        except FileNotFoundError as exc:
            logger.warning(
                "Falling back to synthetic dataset because of missing files: %s",
                exc,
            )
            return self._build_dummy_dataset()

    def _create_loader(self, dataset: Optional[Dataset], shuffle: bool, drop_last: bool = False) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset has not been initialised. Call setup() first.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=drop_last,
        )

    def get_metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            raise RuntimeError(
                "Dataset metadata unavailable. Ensure `setup('fit')` has been called."
            )
        return self._metadata

    def _build_dummy_dataset(self) -> Dataset:
        tensor = torch.zeros(16, 1, int(self.cfg.data.sample_rate * self.cfg.data.clip_duration))
        labels = torch.arange(16, dtype=torch.long) % 2
        return torch.utils.data.TensorDataset(tensor, labels)

    def _build_distillation_dataset(
        self,
        subset: str,
        allowed_classes: Optional[list[str]],
        num_no_bird_samples: int,
    ) -> Dataset:
        dist_cfg = self.cfg.distillation
        soft_labels_path = Path(dist_cfg.soft_labels_path)
        if not soft_labels_path.exists():
            raise FileNotFoundError(f"Soft labels path not found: {soft_labels_path}")

        dataset_kwargs = {
            "bird_data_dir": self.cfg.data.bird_data_dir,
            "esc50_dir": self.cfg.data.esc50_dir,
            "allowed_bird_classes": allowed_classes,
            "target_sr": self.cfg.data.sample_rate,
            "clip_duration": self.cfg.data.clip_duration,
            "subset": subset,
            "validation_split": self.cfg.data.validation_split,
            "test_split": self.cfg.data.test_split,
            "split_seed": self.cfg.project.seed,
            "use_augmentation": bool(self.cfg.data.get("use_augmentation", subset == "training")),
            "load_pregenerated_no_birds": self.cfg.data.load_pregenerated_no_birds,
            "pregenerated_no_birds_dir": self.cfg.data.pregenerated_no_birds_dir,
            "num_no_bird_samples": num_no_bird_samples,
            "esc50_no_bird_ratio": self.cfg.data.esc50_no_bird_ratio,
        }

        logger.info(
            "Creating distillation dataset from %s (subset=%s)",
            soft_labels_path,
            subset,
        )

        dist_dataset = DistillationBirdSoundDataset(
            soft_labels_path=str(soft_labels_path),
            **dataset_kwargs,
        )
        num_classes = getattr(dist_dataset, "num_classes", None)
        if num_classes is None or num_classes == 0:
            if allowed_classes:
                num_classes = len(allowed_classes) + 1
            else:
                num_classes = 1

        dist_wrapped = _DistillationWrapper(dist_dataset)

        if subset != "training":
            sup_dataset = create_combined_dataset(
                bird_data_dir=self.cfg.data.bird_data_dir,
                esc50_dir=self.cfg.data.esc50_dir,
                allowed_bird_classes=allowed_classes,
                target_sr=self.cfg.data.sample_rate,
                clip_duration=self.cfg.data.clip_duration,
                subset=subset,
                num_no_bird_samples=num_no_bird_samples,
                esc50_no_bird_ratio=self.cfg.data.esc50_no_bird_ratio,
                use_augmentation=False,
                load_pregenerated_no_birds=bool(self.cfg.data.load_pregenerated_no_birds),
                pregenerated_no_birds_dir=self.cfg.data.pregenerated_no_birds_dir,
                validation_split=self.cfg.data.validation_split,
                test_split=self.cfg.data.test_split,
                split_seed=self.cfg.project.seed,
                empty_segment_cache_path=self.cfg.data.get("empty_segment_cache_path"),
                refresh_empty_segment_cache=bool(self.cfg.data.get("refresh_empty_segment_cache", False)),
            )
            return _SupervisedWrapper(sup_dataset, num_classes)

        datasets: list[Dataset] = [dist_wrapped]

        hybrid_ratio = float(dist_cfg.get("hybrid_ratio", 0.0))
        use_hybrid = bool(dist_cfg.get("use_hybrid_dataset", False))
        if use_hybrid and hybrid_ratio > 0.0:
            hybrid_ratio = max(0.0, min(hybrid_ratio, 1.0))
            sup_dataset = create_combined_dataset(
                bird_data_dir=self.cfg.data.bird_data_dir,
                esc50_dir=self.cfg.data.esc50_dir,
                allowed_bird_classes=allowed_classes,
                target_sr=self.cfg.data.sample_rate,
                clip_duration=self.cfg.data.clip_duration,
                subset=subset,
                num_no_bird_samples=num_no_bird_samples,
                esc50_no_bird_ratio=self.cfg.data.esc50_no_bird_ratio,
                use_augmentation=bool(self.cfg.data.get("use_augmentation", subset == "training")),
                load_pregenerated_no_birds=bool(self.cfg.data.load_pregenerated_no_birds),
                pregenerated_no_birds_dir=self.cfg.data.pregenerated_no_birds_dir,
                validation_split=self.cfg.data.validation_split,
                test_split=self.cfg.data.test_split,
                split_seed=self.cfg.project.seed,
                empty_segment_cache_path=self.cfg.data.get("empty_segment_cache_path"),
                refresh_empty_segment_cache=bool(self.cfg.data.get("refresh_empty_segment_cache", False)),
            )
            total_dist = len(dist_dataset)
            sup_len = int(hybrid_ratio * total_dist)
            if sup_len > 0:
                if sup_len >= len(sup_dataset):
                    sup_subset = sup_dataset
                else:
                    gen = torch.Generator().manual_seed(int(self.cfg.project.seed))
                    indices = torch.randperm(len(sup_dataset), generator=gen)[:sup_len]
                    sup_subset = Subset(sup_dataset, indices.tolist())
                datasets.append(_SupervisedWrapper(sup_subset, num_classes))

        if len(datasets) == 1:
            return dist_wrapped
        return ConcatDataset(datasets)

    def _extract_metadata(self, dataset: Optional[Dataset]) -> DatasetMetadata:
        """
        Infer number of classes and related information from the dataset.
        """
        default_allowed = list(self.cfg.data.allowed_bird_classes) or []
        fallback_num_classes = len(default_allowed) if default_allowed else self._get_model_num_classes()
        fallback_num_classes = max(1, fallback_num_classes)
        if dataset is None:
            num_classes = fallback_num_classes
            if not default_allowed:
                num_classes = max(2, num_classes)
            return DatasetMetadata(
                num_classes=num_classes,
                has_no_birds=False,
                allowed_classes=default_allowed,
                distillation_enabled=False,
            )

        inspected = set()
        num_classes: Optional[int] = None
        allowed_classes = default_allowed.copy()
        has_no_birds = False
        distillation_active = self.use_distillation

        def inspect(ds: Dataset) -> None:
            nonlocal num_classes, has_no_birds, allowed_classes
            if id(ds) in inspected:
                return
            inspected.add(id(ds))

            if hasattr(ds, "num_classes"):
                try:
                    value = int(getattr(ds, "num_classes"))
                    num_classes = max(num_classes or 0, value)
                except Exception:
                    pass
            if hasattr(ds, "get_num_classes"):
                try:
                    value = int(ds.get_num_classes())  # type: ignore[attr-defined]
                    num_classes = max(num_classes or 0, value)
                except Exception:
                    pass
            if hasattr(ds, "allowed_classes") and not allowed_classes:
                try:
                    allowed_classes = list(getattr(ds, "allowed_classes"))
                except Exception:
                    pass
            if hasattr(ds, "class_to_idx"):
                try:
                    class_keys = list(getattr(ds, "class_to_idx").keys())
                    if not allowed_classes:
                        allowed_classes = class_keys
                    if any("no" in key.lower() and "bird" in key.lower() for key in class_keys):
                        has_no_birds = True
                except Exception:
                    pass
            module_name = ds.__class__.__name__.lower()
            if "nobird" in module_name or "no_bird" in module_name:
                has_no_birds = True

            if hasattr(ds, "datasets"):
                for child in getattr(ds, "datasets"):  # type: ignore[attr-defined]
                    inspect(child)
            if hasattr(ds, "dataset"):
                inspect(getattr(ds, "dataset"))
            if hasattr(ds, "base_dataset"):
                inspect(getattr(ds, "base_dataset"))
            if hasattr(ds, "base"):
                inspect(getattr(ds, "base"))

        inspect(dataset)

        if num_classes is None or num_classes <= 0:
            num_classes = fallback_num_classes

        if has_no_birds and allowed_classes and num_classes <= len(allowed_classes):
            num_classes = len(allowed_classes) + 1

        if num_classes <= 1:
            num_classes = max(2, fallback_num_classes or 2)

        return DatasetMetadata(
            num_classes=num_classes,
            has_no_birds=has_no_birds,
            allowed_classes=allowed_classes,
            distillation_enabled=distillation_active,
        )

    def _dataset_len(self, dataset: Optional[Dataset]) -> int:
        if dataset is None:
            return 0
        try:
            return len(dataset)
        except TypeError:
            return 0

    def _log_split_stats(self, metadata: DatasetMetadata) -> None:
        logger.info(
            "Dataset splits - train: %s, val: %s, test: %s, classes: %s (no_birds=%s)",
            metadata.train_samples,
            metadata.val_samples,
            metadata.test_samples,
            metadata.num_classes,
            metadata.has_no_birds,
        )

    def _get_model_num_classes(self) -> int:
        model_cfg = getattr(self.cfg, "model", None)
        params = getattr(model_cfg, "params", None) if model_cfg else None
        candidates = []

        def _extract(value):
            try:
                return int(value)
            except Exception:
                return None

        if params is not None:
            if hasattr(params, "num_classes"):
                candidates.append(_extract(getattr(params, "num_classes")))
            if hasattr(params, "get"):
                candidates.append(_extract(params.get("num_classes")))
            if isinstance(params, dict):
                candidates.append(_extract(params.get("num_classes")))
        if model_cfg is not None:
            if hasattr(model_cfg, "num_classes"):
                candidates.append(_extract(getattr(model_cfg, "num_classes")))

        for candidate in candidates:
            if candidate is not None and candidate > 0:
                return max(2, candidate)
        return 2


@dataclass
class DatasetMetadata:
    num_classes: int
    has_no_birds: bool
    allowed_classes: list[str]
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    distillation_enabled: bool = False


class _DistillationWrapper(Dataset):
    def __init__(self, base: DistillationBirdSoundDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        audio, label, soft = self.base[idx]
        mask = torch.tensor(True, dtype=torch.bool)
        return audio, label, soft, mask


class _SupervisedWrapper(Dataset):
    def __init__(self, base: Dataset, num_classes: int):
        self.base = base
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        audio, label = self.base[idx]
        soft = torch.zeros(self.num_classes, dtype=torch.float32)
        mask = torch.tensor(False, dtype=torch.bool)
        return audio, label, soft, mask


__all__ = ["BirdsDataModule"]
