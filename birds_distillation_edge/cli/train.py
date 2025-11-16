"""Command-line entry point for Lightning training."""

from __future__ import annotations

__all__ = ["main", "resolve_config_dir", "load_config", "build_trainer"]

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import lightning as L
from hydra import compose, initialize_config_dir
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from ..datamodule import BirdsDataModule
from ..experiment import BirdsExperiment
from lightning.pytorch.loggers import CSVLogger

from ..utils import configure_logging, ensure_dir

log = logging.getLogger(__name__)


def resolve_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def load_config(config_name: str, overrides: Sequence[str]) -> DictConfig:
    config_dir = resolve_config_dir()
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def build_trainer(trainer_cfg: DictConfig, project_cfg: DictConfig | None = None) -> L.Trainer:
    trainer_kwargs = {
        key: value
        for key, value in OmegaConf.to_container(trainer_cfg, resolve=True).items()
        if value is not None
    }
    if "logger" not in trainer_kwargs:
        run_name = "training"
        if project_cfg:
            run_name = str(project_cfg.get("name", run_name))
        log_dir = ensure_dir(Path.cwd() / "logs" / "lightning")
        trainer_kwargs["logger"] = CSVLogger(save_dir=str(log_dir), name=run_name)
    log.info("Initialising Trainer with args: %s", trainer_kwargs)
    return L.Trainer(**trainer_kwargs)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Birds Distillation Edge trainer")
    parser.add_argument(
        "--config-name",
        default="base",
        help="Configuration file name (without .yaml) located in the config directory.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra-style overrides (e.g. data.batch_size=16).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config_name, args.overrides)
    log.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.project.get("seed") is not None:
        seed = int(cfg.project.seed)
        log.info("Seeding training run with seed=%s", seed)
        seed_everything(seed, workers=True)

    log_level_name = str(cfg.logging.level).upper()
    configure_logging(log_level_name)

    datamodule = BirdsDataModule(cfg)
    datamodule.setup(stage="fit")
    metadata = datamodule.get_metadata()

    if cfg.model.params.get("num_classes") != metadata.num_classes:
        log.info(
            "Adjusting model.num_classes from %s to dataset-derived value %s",
            cfg.model.params.get("num_classes"),
            metadata.num_classes,
        )
        cfg.model.params.num_classes = metadata.num_classes

    if metadata.distillation_enabled:
        log.info("Distillation mode active (soft labels: %s)", cfg.distillation.soft_labels_path)

    experiment = BirdsExperiment(cfg, num_classes_override=metadata.num_classes)
    trainer = build_trainer(cfg.trainer, cfg.project)

    log.info("Starting training with Lightning Trainer.")
    trainer.fit(model=experiment, datamodule=datamodule)

    testing_cfg = getattr(cfg, "testing", None)
    if testing_cfg and testing_cfg.get("run_after_train", False):
        log.info("Running test loop after training as requested by configuration.")
        trainer.test(model=experiment, datamodule=datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
