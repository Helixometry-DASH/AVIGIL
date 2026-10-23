"""Multi-scale audio encoder.

A fixed mel-spaced filterbank followed by six residual blocks, each pooling
the time axis by 3, returning selected intermediate feature maps rather than
only the last one:

    block 4    hop 3**5 = 243 samples    15.19 ms per frame
    block 6    hop 3**7 = 2187 samples   136.69 ms per frame

Block 4 is the resolution the localization decoder predicts on; block 6 carries
the coarse context the graph layers reason over.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ResidualBlock, SincFilterBank


class MultiScaleEncoder(nn.Module):
    """Filterbank + 6 residual blocks, exposing selected intermediate taps."""

    def __init__(
        self,
        filts: list | None = None,
        first_conv: int = 128,
        taps: tuple[int, ...] = (4, 6),
        freeze: bool = False,
    ) -> None:
        super().__init__()
        filts = filts or [70, [1, 32], [32, 32], [32, 64], [64, 64]]

        self.conv_time = SincFilterBank(out_channels=filts[0], kernel_size=first_conv,
                                        in_channels=1)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(nb_filts=filts[1], first=True),
                ResidualBlock(nb_filts=filts[2]),
                ResidualBlock(nb_filts=filts[3]),
                ResidualBlock(nb_filts=filts[4]),
                ResidualBlock(nb_filts=filts[4]),
                ResidualBlock(nb_filts=filts[4]),
            ]
        )
        self.taps = tuple(sorted(taps))
        self.channels = {1: filts[1][1], 2: filts[2][1], 3: filts[3][1],
                         4: filts[4][1], 5: filts[4][1], 6: filts[4][1]}

        if freeze:
            self.requires_grad_(False)

    def forward(self, audio: torch.Tensor) -> dict[int, torch.Tensor]:
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        x = self.conv_time(audio.unsqueeze(1))          # (B, 70, N - kernel + 1)
        x = F.max_pool2d(torch.abs(x.unsqueeze(1)), (3, 3))
        x = self.selu(self.first_bn(x))
        out: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in self.taps:
                out[i] = x                               # (B, C, freq, T)
        return out
