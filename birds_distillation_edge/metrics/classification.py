"""Classification metrics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class ClassificationReport:
    report: str
    confusion: np.ndarray
    class_names: List[str]


class ClassificationMetrics:
    """Compute classification report and confusion matrix."""

    def __init__(self, class_names: Iterable[str]):
        self.class_names = list(class_names)

    def compute(self, preds: np.ndarray, targets: np.ndarray) -> ClassificationReport:
        labels = np.arange(len(self.class_names))
        report = classification_report(
            targets,
            preds,
            labels=labels,
            target_names=self.class_names,
            zero_division=0,
        )
        cm = confusion_matrix(targets, preds, labels=labels)
        return ClassificationReport(report=report, confusion=cm, class_names=self.class_names)
