"""
LightningModule implementing the supervised training workflow.

This module is the first step in porting the legacy training script to
PyTorch Lightning. It currently supports standard cross-entropy
training; distillation-specific behaviours will be layered on top in a
subsequent iteration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import MulticlassAccuracy
import numpy as np

from .losses.distillation import DistillationLoss, FocalDistillationLoss, FocalLoss
from .metrics.classification import ClassificationMetrics
from .utils import save_classification_report

logger = logging.getLogger(__name__)


def _to_container(cfg_section: Any) -> Dict[str, Any]:
    """Safely convert an OmegaConf section to a plain dictionary."""
    if cfg_section is None:
        return {}
    return OmegaConf.to_container(cfg_section, resolve=True, enum_to_str=True) or {}


class BirdsExperiment(L.LightningModule):
    """
    Lightning module handling supervised bird classification training.
    """

    def __init__(self, cfg: DictConfig, num_classes_override: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg

        # Persist configuration for checkpointing/logging.
        self.save_hyperparameters(
            _to_container(cfg)
        )

        # Distillation feature flags.
        self.distillation_enabled: bool = bool(cfg.losses.distillation.enabled)
        self.distillation_config = cfg.losses.distillation if self.distillation_enabled else None
        self._distillation_warning_emitted = False
        self.distillation_loss_fn: Optional[nn.Module] = None
        self.focal_enabled: bool = bool(cfg.losses.focal.enabled)

        # Instantiate model via Hydra to keep compatibility with existing configs.
        model_kwargs = _to_container(cfg.model.params)
        if num_classes_override is not None:
            model_kwargs["num_classes"] = int(num_classes_override)
        model_target = cfg.model.target
        model_conf: Dict[str, Any] = {"_target_": model_target, **model_kwargs}
        self.model: nn.Module = instantiate(model_conf)

        self.num_classes: int = int(model_kwargs.get("num_classes", 1))

        # Loss function and metrics.
        classification_cfg = cfg.losses.classification
        self.loss_fn = self._build_classification_loss(classification_cfg)

        if self.distillation_enabled and self.distillation_config:
            self.distillation_loss_fn = self._build_distillation_loss(self.distillation_config)

        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.class_names = self._build_class_names(model_kwargs.get("num_classes", self.num_classes))
        self.classification_metrics = ClassificationMetrics(self.class_names)
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []

    # --------------------------------------------------------------------- #
    # Lightning API
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        self._maybe_warn_distillation()
        audio, labels, soft_labels, mask = self._prepare_batch(batch)
        logits = self(audio)
        loss = self._compute_loss(logits, labels, soft_labels, mask, stage="train")

        acc = self.train_accuracy(logits, labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        self.log(
            "train/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        self._maybe_warn_distillation()
        audio, labels, soft_labels, mask = self._prepare_batch(batch)
        logits = self(audio)
        loss = self._compute_loss(logits, labels, soft_labels, mask, stage="val")
        preds = torch.argmax(logits, dim=1)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(labels.detach().cpu())

        acc = self.val_accuracy(logits, labels)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        self.log(
            "val/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        self._maybe_warn_distillation()
        audio, labels, soft_labels, mask = self._prepare_batch(batch)
        logits = self(audio)
        loss = self._compute_loss(logits, labels, soft_labels, mask, stage="test")
        preds = torch.argmax(logits, dim=1)
        self._test_preds.append(preds.detach().cpu())
        self._test_targets.append(labels.detach().cpu())

        acc = self.test_accuracy(logits, labels)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        self.log(
            "test/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def configure_optimizers(self):
        optimizer = self._build_optimizer()
        scheduler_cfg = self.cfg.get("scheduler")

        if scheduler_cfg is None or scheduler_cfg.name in (None, "null", ""):
            return optimizer

        scheduler_params = _to_container(scheduler_cfg.get("params"))
        name = scheduler_cfg.name

        if name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_params
            )
            monitor = scheduler_cfg.get("monitor", "val/loss")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        raise ValueError(f"Unsupported scheduler configured: {name}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _prepare_batch(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask: Optional[torch.Tensor]
        soft: Optional[torch.Tensor]

        if len(batch) == 4:
            audio, labels, soft, mask = batch
            mask = mask.to(self.device, non_blocking=True).view(-1).bool()
            soft = soft.to(self.device, non_blocking=True).float()
        elif len(batch) == 3:
            audio, labels, soft = batch
            mask = torch.ones(labels.shape[0], device=labels.device, dtype=torch.bool)
            soft = soft.to(self.device, non_blocking=True).float()
        elif len(batch) == 2:
            audio, labels = batch
            soft = None
            mask = None
        else:
            raise ValueError("Unexpected batch structure received")

        audio = audio.to(self.device, non_blocking=True).float()
        labels = labels.to(self.device, non_blocking=True).long()
        if mask is not None and mask.device != self.device:
            mask = mask.to(self.device, non_blocking=True)
        return audio, labels, soft, mask

    def _build_classification_loss(self, classification_cfg: DictConfig) -> nn.Module:
        if self.focal_enabled:
            focal_cfg = self.cfg.losses.focal
            params = _to_container(focal_cfg.get("params"))
            alpha = params.get("alpha", 1.0)
            gamma = float(focal_cfg.get("gamma", 2.0))
            reduction = params.get("reduction", "mean")
            ignore_index = params.get("ignore_index", -100)
            return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index)

        name = classification_cfg.name.lower()
        if name == "cross_entropy":
            params = _to_container(classification_cfg.get("params"))
            return nn.CrossEntropyLoss(**params)
        raise ValueError(f"Unsupported classification loss: {classification_cfg.name}")

    def _build_distillation_loss(self, distillation_cfg: DictConfig) -> nn.Module:
        alpha = float(distillation_cfg.get("alpha", 0.5))
        temperature = float(distillation_cfg.get("temperature", 4.0))

        if self.focal_enabled:
            focal_cfg = self.cfg.losses.focal
            params = _to_container(focal_cfg.get("params"))
            class_weights = params.get("alpha")
            gamma = float(focal_cfg.get("gamma", 2.0))
            return FocalDistillationLoss(
                alpha=alpha,
                gamma=gamma,
                temperature=temperature,
                class_weights=class_weights,
            )

        return DistillationLoss(alpha=alpha, temperature=temperature)
    def _build_class_names(self, inferred_classes: int) -> List[str]:
        names = list(self.cfg.data.allowed_bird_classes)
        if not names:
            names = [f"class_{i}" for i in range(inferred_classes)]
        while len(names) < inferred_classes:
            names.append(f"class_{len(names)}")
        return names[:inferred_classes]
    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        self._log_classification_metrics("val", preds, targets)

    def on_test_epoch_start(self) -> None:
        self._test_preds.clear()
        self._test_targets.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        preds = torch.cat(self._test_preds).numpy()
        targets = torch.cat(self._test_targets).numpy()
        self._log_classification_metrics("test", preds, targets)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optim_cfg = self.cfg.optimizers
        main_cfg = optim_cfg.main
        lr = float(main_cfg.lr)
        weight_decay = float(main_cfg.get("weight_decay", 0.0))

        breakpoint_lr = optim_cfg.get("breakpoint_lr")
        transition_lr = optim_cfg.get("transition_width_lr")
        breakpoint_lr = float(breakpoint_lr) if breakpoint_lr is not None else None
        transition_lr = float(transition_lr) if transition_lr is not None else None

        param_groups = self._split_parameter_groups(
            lr,
            breakpoint_lr,
            transition_lr,
        )

        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        return optimizer

    def _split_parameter_groups(
        self,
        fallback_lr: float,
        breakpoint_lr: float | None,
        transition_lr: float | None,
    ) -> list[Dict[str, Any]]:
        groups: list[Dict[str, Any]] = []
        other_params = []
        breakpoint_params = []
        transition_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "combined_log_linear_spec.breakpoint" in name:
                breakpoint_params.append(param)
            elif "combined_log_linear_spec.transition_width" in name:
                transition_params.append(param)
            else:
                other_params.append(param)

        if other_params:
            groups.append({"params": other_params, "lr": fallback_lr})
        if breakpoint_params:
            groups.append(
                {
                    "params": breakpoint_params,
                    "lr": breakpoint_lr if breakpoint_lr is not None else fallback_lr,
                }
            )
        if transition_params:
            groups.append(
                {
                    "params": transition_params,
                    "lr": transition_lr if transition_lr is not None else fallback_lr,
                }
            )

        if not groups:
            raise RuntimeError("No trainable parameters found for optimizer.")
        return groups

    def _maybe_warn_distillation(self) -> None:
        if (
            self.distillation_enabled
            and self.distillation_loss_fn is None
            and not self._distillation_warning_emitted
        ):
            logger.warning(
                "Distillation requested but loss function is unavailable. Proceeding with "
                "supervised loss only."
            )
            self._distillation_warning_emitted = True

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device)
        sup_loss = torch.tensor(0.0, device=logits.device)

        if mask is not None and soft_labels is not None and self.distillation_loss_fn is not None:
            mask = mask.view(-1).bool()
            dist_idx = torch.nonzero(mask, as_tuple=True)[0]
            sup_idx = torch.nonzero(~mask, as_tuple=True)[0]

            if dist_idx.numel() > 0:
                dist_total, dist_hard, dist_soft = self.distillation_loss_fn(
                    logits[dist_idx], labels[dist_idx], soft_labels[dist_idx]
                )
                total_loss = total_loss + dist_total
                self.log(
                    f"{stage}/dist_hard_loss",
                    dist_hard,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=dist_idx.numel(),
                )
                self.log(
                    f"{stage}/dist_soft_loss",
                    dist_soft,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=dist_idx.numel(),
                )

            if sup_idx.numel() > 0:
                sup_loss = self.loss_fn(logits[sup_idx], labels[sup_idx])
                total_loss = total_loss + sup_loss
                self.log(
                    f"{stage}/sup_loss",
                    sup_loss,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=sup_idx.numel(),
                )
            if dist_idx.numel() > 0:
                return total_loss
            if sup_idx.numel() > 0:
                return sup_loss

        # Fallback to standard loss (entire batch)
        base_loss = self.loss_fn(logits, labels)
        if mask is not None:
            self.log(
                f"{stage}/sup_loss",
                base_loss,
                on_epoch=True,
                prog_bar=False,
                batch_size=labels.size(0),
            )
        return base_loss

    def _log_classification_metrics(self, stage: str, preds: np.ndarray, targets: np.ndarray) -> None:
        metrics = self.classification_metrics.compute(preds, targets)
        logger.info("%s classification report:\n%s", stage.upper(), metrics.report)
        logger.info("%s confusion matrix:\n%s", stage.upper(), metrics.confusion)
        save_classification_report(
            metrics.report,
            metrics.confusion,
            metrics.class_names,
            self._get_output_dir(),
            stage,
        )

    def _get_output_dir(self) -> Path:
        if self.trainer is not None:
            if getattr(self.trainer, "log_dir", None):
                return Path(self.trainer.log_dir)
            if self.trainer.logger and getattr(self.trainer.logger, "log_dir", None):
                return Path(self.trainer.logger.log_dir)
        return Path.cwd()
