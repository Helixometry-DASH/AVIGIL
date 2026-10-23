"""The model contract.

The trainer, the evaluator and the metrics depend only on this interface, so
the architecture can be swapped without touching them.

A model takes a batch and returns per-frame forgery logits. Returning logits
rather than probabilities keeps ``BCEWithLogitsLoss`` numerically stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AVLocalizer(nn.Module, ABC):
    """Base class for audio-visual temporal forgery localizers."""

    @abstractmethod
    def forward(
        self, video: torch.Tensor, audio: torch.Tensor, valid: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            video: ``(B, T, 3, H, W)`` uint8 or float
            audio: ``(B, S)`` float32 at 16 kHz, ``S == T / 25 * 16000``
            valid: ``(B, T)`` 1 for decoded frames, 0 for padding

        Returns a dict with at least:
            ``frame_logits``: ``(B, T)`` per-frame forged logit
        and optionally:
            ``clip_logit``:   ``(B,)``   clip-level real/fake logit
            ``modality_logits``: ``(B, 4)`` real / visual / audio / both
            ``boundary``:     ``(B, T, 2)`` start/end offsets for regression heads
        """
        raise NotImplementedError

    @property
    def frame_rate(self) -> float:
        return 25.0
