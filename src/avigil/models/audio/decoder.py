"""Temporal FPN decoder: coarse semantics at fine resolution.

The coarse block-6 path is upsampled by the exact grid ratio and added to a
projection of the block-4 skip, then refined by dilated residual blocks.
Addition rather than concatenation keeps the channel count fixed across
variants.

Normalisation is GroupNorm, which carries no running statistics, so the layer
behaves identically in training and in evaluation and is independent of batch
size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int, target: int = 8) -> int:
    """Largest divisor of ``channels`` not exceeding ``target``."""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class TemporalBlock(nn.Module):
    """Dilated residual temporal conv -- widens context without downsampling."""

    def __init__(self, dim: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.1) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation),
            nn.GroupNorm(_groups(dim), dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TemporalDecoder(nn.Module):
    """Fuse a coarse feature sequence with an optional fine-resolution skip.

    Coarse features are upsampled to the skip's grid (block 4, 15.19 ms) by
    linear interpolation, then added to a projection of the skip. Addition
    rather than concat keeps the channel count fixed so A0 and A1 have nearly
    identical decoder capacity -- otherwise "higher resolution helps" could
    just be "more parameters help".
    """

    def __init__(
        self,
        coarse_dim: int,
        dim: int = 128,
        skip_dim: int | None = None,
        n_blocks: int = 4,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        coarse_scale: int = 1,
    ) -> None:
        super().__init__()
        # ratio of the coarse hop to the output hop, e.g. 3**7 / 3**5 = 9 for
        # block 6 -> block 4. Upsampling by this fixed factor rather than to an
        # absolute length is what makes the decoder batch-shape independent.
        self.coarse_scale = int(coarse_scale)
        self.proj_coarse = nn.Conv1d(coarse_dim, dim, 1)
        self.proj_skip = nn.Conv1d(skip_dim, dim, 1) if skip_dim else None
        dil = (dilations * n_blocks)[:n_blocks]
        self.blocks = nn.Sequential(*[TemporalBlock(dim, 3, d, dropout) for d in dil])
        self.out_dim = dim

    def forward(
        self, coarse: torch.Tensor, skip: torch.Tensor | None = None,
        out_len: int | None = None,
    ) -> torch.Tensor:
        """``coarse``: (B, T_c, C). ``skip``: (B, T_f, C_s). -> (B, T_out, dim)"""
        h = self.proj_coarse(coarse.transpose(1, 2))         # (B, dim, T_c)
        target = out_len or (skip.size(1) if skip is not None else coarse.size(1))
        h = self._to_grid(h, target)
        if skip is not None and self.proj_skip is not None:
            s = self.proj_skip(skip.transpose(1, 2))
            h = h + self._fit(s, target)
        return self.blocks(h).transpose(1, 2)                # (B, T_out, dim)

    def _to_grid(self, h: torch.Tensor, target: int) -> torch.Tensor:
        """Upsample by the fixed grid ratio, then fit to ``target``.

        The scale factor is fixed rather than derived from ``target``. With a
        ragged batch ``target`` is the padded width, so a derived factor would
        resample every position -- not just the padded tail -- from a
        different source location. A fixed factor makes position ``i``
        independent of the batch it is in.
        """
        if self.coarse_scale > 1:
            h = F.interpolate(h, scale_factor=float(self.coarse_scale),
                              mode="linear", align_corners=False)
        return self._fit(h, target)

    @staticmethod
    def _fit(x: torch.Tensor, target: int) -> torch.Tensor:
        """Trim or edge-replicate to exactly ``target`` frames."""
        n = x.size(-1)
        if n == target:
            return x
        if n > target:
            return x[..., :target]
        return F.pad(x, (0, target - n), mode="replicate")
