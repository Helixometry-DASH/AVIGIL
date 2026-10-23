"""Prediction heads: dense frame/start/end, plus an auxiliary clip logit."""

from __future__ import annotations

import torch
from torch import nn

from .decoder import _groups


class LocalizationHead(nn.Module):
    """Three time-aligned logits per frame, and one clip-level logit.

    Separate 1x1 convs rather than a single 3-channel conv: the frame task is
    region segmentation while start/end are sparse peak detection, and keeping
    them separate lets each head's bias initialise to its own prior.

    GroupNorm, not BatchNorm -- see the note in decoder.py.
    """

    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(dim, hidden, 3, padding=1), nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.frame = nn.Conv1d(hidden, 1, 1)
        self.start = nn.Conv1d(hidden, 1, 1)
        self.end = nn.Conv1d(hidden, 1, 1)
        self.clip = nn.Linear(hidden, 1)
        # start/end are sparse (~2 peaks in 265 frames): bias toward negative
        for head in (self.start, self.end):
            nn.init.constant_(head.bias, -4.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """``x``: (B, T, dim) -> frame/start/end (B, T), clip_logit (B,).

        ``mask`` (B, T) is 1 on real frames. The clip head pools over time, so
        an unmasked mean over a padded batch mixes padded features into the
        clip decision in proportion to the pad fraction (0.147 on a random
        full-length batch of 8).
        """
        h = self.trunk(x.transpose(1, 2))                    # (B, hidden, T)
        if mask is None:
            pooled = h.mean(dim=2)
        else:
            m = mask[:, None, :].to(h.dtype)
            pooled = (h * m).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)
        return {
            "frame_logits": self.frame(h).squeeze(1),
            "start_logits": self.start(h).squeeze(1),
            "end_logits": self.end(h).squeeze(1),
            "clip_logit": self.clip(pooled).squeeze(-1),
            # pre-head features, for a downstream fusion model to read. Adding a
            # key is backward compatible: every existing caller indexes by name.
            "features": h,
        }
