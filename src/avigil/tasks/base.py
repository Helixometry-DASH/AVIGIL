"""Task interface.

A Task owns everything that differs between experiments *except* the model:
which records are used, what the label means, the loss, and which metrics are
reported. The trainer is task-agnostic and only calls these methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class Task(ABC):
    name: str

    @abstractmethod
    def build_dataset(self, split: str, train: bool) -> Dataset: ...

    @abstractmethod
    def build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def compute_loss(
        self, model: torch.nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    @abstractmethod
    def predict(
        self, model: torch.nn.Module, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]: ...

    @abstractmethod
    def evaluate(self, collected: dict[str, Any]) -> dict[str, float]: ...

    @property
    def monitor(self) -> tuple[str, str]:
        """(metric name, "min" | "max") used for checkpoint selection."""
        return ("eer", "min")

    @property
    def collate(self):
        """Optional custom collate_fn; None uses torch's default.

        Localization needs one because the per-clip segment lists are ragged.
        """
        return None

    @property
    def collate_pads(self) -> bool:
        """True if ``collate`` pads ragged items and reports a validity mask.

        A collate that calls ``torch.stack`` needs every item the same shape, so
        a task without a padding collate must evaluate full-length clips one at
        a time. The video task pads; the audio localization task does not.
        """
        return False

