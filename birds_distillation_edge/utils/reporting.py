"""Reporting utilities (classification reports, confusion matrices)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_classification_report(report: str, confusion: np.ndarray, class_names: Iterable[str], output_dir: Path, stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{stage}_classification_report.txt"
    cm_path = output_dir / f"{stage}_confusion_matrix.png"

    report_path.write_text(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{stage.title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
