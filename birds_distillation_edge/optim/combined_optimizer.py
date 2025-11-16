"""
Combined optimizer utilities for Birds Distillation Edge.

These classes mirror the functionality previously provided in
`distillation/optimizer/combined_optimizer.py`, enabling multiple
optimizers with selective activation (e.g., for distillation-specific
parameters).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from torch.optim import Optimizer


class CombinedOptimizer:
    """Wrapper managing multiple `torch.optim.Optimizer` instances."""

    def __init__(self, main_optimizer: Optimizer, filter_optimizer: Optimizer) -> None:
        self.optimizers: Dict[str, Optimizer] = {
            "main": main_optimizer,
            "filter": filter_optimizer,
        }
        self.active_optimizers: List[str] = ["main", "filter"]

    def set_active_optimizers(self, active_keys: Iterable[str]) -> None:
        self.active_optimizers = [key for key in active_keys if key in self.optimizers]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for key in self.active_optimizers:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for key in self.active_optimizers:
            self.optimizers[key].step()

    def state_dict(self) -> Dict[str, dict]:
        return {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: Dict[str, dict]) -> None:
        for key, value in state_dict.items():
            if key in self.optimizers:
                self.optimizers[key].load_state_dict(value)

    @property
    def param_groups(self):
        return self.optimizers["main"].param_groups

    @property
    def defaults(self):
        return self.optimizers["main"].defaults
