"""Time-preserving graph reasoning over the coarse audio features.

The temporal node set is **never pooled**: top-k pooling both drops nodes and
reorders the survivors by score, after which node *i* no longer corresponds to
a known timestamp, and the whole point here is that it does. Temporal node *t*
therefore stays at timestamp *t* through every layer.

The spectral node set is pooled, since collapsing frequency costs no temporal
information and that axis has no ordering to preserve.
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import GraphAttention, GraphPool, HeterogeneousGraphAttention


class TimePreservingGAT(nn.Module):
    """Graph attention over temporal nodes, with nothing pooled.

    Input and output are both ``(B, T, D)`` with T and node order intact.
    Attention is dense over T, so cost is O(T^2 * D); it is applied at the
    coarse block-6 grid (T = 29 for a 4 s crop) while output resolution comes
    from the block-4 skip.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 2.0) -> None:
        super().__init__()
        self.gat = GraphAttention(in_dim, out_dim, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gat(x)                      # (B, T, out_dim), T preserved


class HSGALLoc(nn.Module):
    """Two-stage heterogeneous spectral-temporal attention.

    Each stage attends over the temporal and spectral node sets jointly and
    carries a master node forward. Between the stages only the spectral set is
    pooled, so every temporal node survives with its index, and the two stages'
    outputs are summed residually.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 100.0,
                 spectral_pool: float = 0.5) -> None:
        super().__init__()
        self.stage1 = HeterogeneousGraphAttention(in_dim, out_dim, temperature=temperature)
        self.stage2 = HeterogeneousGraphAttention(out_dim, out_dim, temperature=temperature)
        self.pool_spectral = GraphPool(spectral_pool, out_dim, 0.3)
        self.master = nn.Parameter(torch.randn(1, 1, in_dim))

    def forward(
        self, temporal: torch.Tensor, spectral: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = temporal.size(0)
        master = self.master.expand(b, -1, -1)
        t1, s1, m1 = self.stage1(temporal, spectral, master=master)
        s1 = self.pool_spectral(s1)                 # spectral axis only
        t2, s2, m2 = self.stage2(t1, s1, master=m1)
        return t1 + t2, s1 + s2, m1 + m2            # temporal length untouched
