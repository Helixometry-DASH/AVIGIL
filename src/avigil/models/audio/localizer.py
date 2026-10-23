"""Audio forgery localization.

A fixed mel-spaced filterbank and a residual CNN produce features at two time
scales; graph attention reasons over the coarse scale with the temporal axis
preserved; and a temporal FPN decoder predicts frame, start and end logits on
the fine 15.19 ms grid, plus an auxiliary clip logit.

Four variants, each adding one component to the previous:

    a0  coarse block-6 features -> decoder          (136.69 ms source grid)
    a1  + block-4 skip                              (15.19 ms source grid)
    a2  + time-preserving temporal graph attention
    a3  + spectral graph attention + heterogeneous stage   <- released model
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import register
from .decoder import TemporalDecoder
from .encoder import MultiScaleEncoder
from .graph import HSGALLoc, TimePreservingGAT
from .grid import SINC_KERNEL, hop_samples, n_frames
from .heads import LocalizationHead


class AudioLocalizer(nn.Module):
    def __init__(
        self,
        variant: str = "a1",
        fine_block: int = 4,
        coarse_block: int = 6,
        dim: int = 128,
        gat_dim: int = 64,
        decoder_blocks: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        filts: list | None = None,
        first_conv: int = 128,
    ) -> None:
        super().__init__()
        if variant not in ("a0", "a1", "a2", "a3"):
            raise ValueError(f"variant must be a0|a1|a2|a3, got {variant!r}")
        self.variant = variant
        self.fine_block = int(fine_block)
        self.coarse_block = int(coarse_block)
        self.use_skip = variant in ("a1", "a2", "a3")
        self.use_tgat = variant in ("a2", "a3")
        self.use_hsgal = variant == "a3"

        taps = (coarse_block,) if not self.use_skip else (fine_block, coarse_block)
        self.encoder = MultiScaleEncoder(
            filts=filts, first_conv=first_conv, taps=taps, freeze=freeze_encoder,
        )
        enc_ch = self.encoder.channels[coarse_block]
        fine_ch = self.encoder.channels[fine_block]

        coarse_dim = enc_ch
        if self.use_tgat:
            self.tgat = TimePreservingGAT(enc_ch, gat_dim)
            coarse_dim = gat_dim
        if self.use_hsgal:
            self.sgat = TimePreservingGAT(enc_ch, gat_dim)
            self.hsgal = HSGALLoc(gat_dim, gat_dim)
            coarse_dim = gat_dim

        # the coarse path is upsampled by the exact grid ratio (3**(c+1)/3**(f+1),
        # i.e. 9 for block 6 -> block 4) so the result does not depend on how
        # much padding its batch happens to carry
        scale = (hop_samples(coarse_block) // hop_samples(fine_block)) if self.use_skip else 1
        self.decoder = TemporalDecoder(
            coarse_dim=coarse_dim, dim=dim,
            skip_dim=fine_ch if self.use_skip else None,
            n_blocks=decoder_blocks, dropout=dropout,
            coarse_scale=scale,
        )
        self.head = LocalizationHead(dim, hidden=dim, dropout=dropout)

    # ---- helpers ----------------------------------------------------------
    def output_frames(self, n_samples: int) -> int:
        """Length of the prediction sequence, i.e. the target grid length."""
        block = self.fine_block if self.use_skip else self.coarse_block
        return n_frames(n_samples, block)

    @property
    def output_block(self) -> int:
        return self.fine_block if self.use_skip else self.coarse_block

    # ---- forward ----------------------------------------------------------
    @staticmethod
    def _grid_mask(lengths: torch.Tensor, block: int, width: int) -> torch.Tensor:
        """(B,) audio-sample counts -> (B, width) bool mask on block's grid."""
        n = ((lengths - (SINC_KERNEL - 1)) // hop_samples(block)).clamp(0, width)
        idx = torch.arange(width, device=lengths.device)[None, :]
        return idx < n[:, None]

    def forward(self, audio: torch.Tensor,
                frame_mask: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """``lengths`` (B,) is the true audio-sample count per item.

        Supplying it is what makes a ragged batch safe. A convolution maps an
        all-zero padded position to its bias rather than to zero, and the
        decoder's dilated convolutions have a receptive field of 31 frames
        (~0.47 s), so without ``lengths`` those values would reach the valid
        tail of each clip. Given ``lengths``, the encoder features are masked
        on both taps' grids before the decoder sees them.
        """
        feats = self.encoder(audio)
        coarse = feats[self.coarse_block]                   # (B, C, F, T_c)
        if lengths is not None:
            cm = self._grid_mask(lengths, self.coarse_block, coarse.shape[-1])
            coarse = coarse * cm[:, None, None, :]

        # temporal nodes: collapse frequency, keep every time index
        temporal = torch.max(torch.abs(coarse), dim=2).values.transpose(1, 2)

        if self.use_hsgal:
            # spectral nodes: collapse time (no timeline to preserve there)
            spectral = torch.max(torch.abs(coarse), dim=3).values.transpose(1, 2)
            t = self.tgat(temporal)
            s = self.sgat(spectral)
            temporal, _spectral, _master = self.hsgal(t, s)
        elif self.use_tgat:
            temporal = self.tgat(temporal)

        skip = None
        if self.use_skip:
            fine = feats[self.fine_block]                   # (B, C, F, T_f)
            if lengths is not None:
                fm = self._grid_mask(lengths, self.fine_block, fine.shape[-1])
                fine = fine * fm[:, None, None, :]
            skip = torch.max(torch.abs(fine), dim=2).values.transpose(1, 2)

        out_len = self.output_frames(audio.shape[-1])
        h = self.decoder(temporal, skip=skip, out_len=out_len)
        return self.head(h, frame_mask)


def _build(variant: str):
    def fn(**kwargs) -> AudioLocalizer:
        kwargs.setdefault("variant", variant)
        return AudioLocalizer(**kwargs)
    return fn


for _v in ("a0", "a1", "a2", "a3"):
    register(f"aloc_{_v}")(_build(_v))
