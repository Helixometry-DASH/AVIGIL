"""Visual forgery localization on cached VideoMAE V2 features.

The backbone runs offline (``scripts/extract_features.py``) and writes one fp16
vector set per 80 ms tubelet: channel 0 is the mean over all 196 patch tokens,
channels 1: are a 4x4 spatial region grid. This module consumes that cache.

Four variants, each adding one component to the previous:

    v0  global feature channel, predictions at the 80 ms feature grid
    v1  + x2 upsampling decoder, predictions at the 40 ms video-frame grid
    v2  + spatial attention pooling over the 4x4 region tokens
    v3  + temporal graph attention                          <- released model

Spatial attention pooling lets the model select which part of the frame to
read, which matters because a lip-sync manipulation is confined to the mouth
while the global mean spreads it over all sixteen regions.

Normalization is per-position channel normalization (:class:`ChannelNorm`),
which reduces over channels at each time step and never over time. Clips run
2.4-20.0 s, so batches are padded; normalizing over channels only keeps the
output at every valid position exactly invariant to how much padding its batch
carries and to the sequence length. Padded positions are re-zeroed after every
operation that has a bias, for the same reason.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register

EMBED = 768


class ChannelNorm(nn.Module):
    """LayerNorm over channels of a ``(B, C, T)`` tensor.

    Reduces over ``C`` at each time step, never over ``T``. That makes the
    whole model's output at a valid position independent of how much padding
    its batch carries -- a property GroupNorm does not have and which
    variable-length localization needs.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SpatialAttnPool(nn.Module):
    """Learned multi-head pooling over the region tokens of one time step.

    The cache holds, per 80 ms position, a global vector plus a 4x4 grid of
    region vectors. A fixed mean over regions is what the global channel
    already is, so pooling only earns its parameters if it can *select*: a
    query learns which regions to read, and a forgery confined to one
    sixteenth of the frame can dominate the output instead of being averaged
    into fifteen genuine regions.
    """

    def __init__(self, dim: int = EMBED, out_dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (out_dim // heads) ** -0.5
        self.query = nn.Parameter(torch.randn(heads, out_dim // heads) * 0.02)
        self.key = nn.Linear(dim, out_dim)
        self.value = nn.Linear(dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, regions: torch.Tensor) -> torch.Tensor:
        # regions (B, T, V, D) -> (B, T, out_dim)
        b, t, v, _ = regions.shape
        k = self.key(regions).view(b, t, v, self.heads, -1)
        val = self.value(regions).view(b, t, v, self.heads, -1)
        att = torch.einsum("btvhd,hd->btvh", k, self.query) * self.scale
        att = att.softmax(dim=2)
        out = torch.einsum("btvh,btvhd->bthd", att, val)
        return self.norm(out.reshape(b, t, -1))


class TemporalBlock(nn.Module):
    """Residual dilated conv block: the trunk's unit of temporal context."""

    def __init__(self, dim: int, dilation: int, kernel: int = 3) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation)
        self.norm1 = ChannelNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.norm2 = ChannelNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(x + h)


class TemporalGraphAttention(nn.Module):
    """Temporal graph attention: a local band plus strided global references.

    Every temporal node keeps its timestamp -- nothing is pooled or dropped,
    which is what localization requires. Edges are a +/- ``band`` neighbourhood (manipulation boundaries
    are local) union a strided set of global reference nodes (comparing a
    suspect region against distant genuine ones).
    """

    def __init__(self, dim: int, heads: int = 4, band: int = 4, n_global: int = 8) -> None:
        super().__init__()
        self.heads, self.band, self.n_global = heads, band, n_global
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _edges(self, t: int, lengths: torch.Tensor, device) -> torch.Tensor:
        """Per-clip ``(B, T, T)`` adjacency: local band union strided globals.

        The stride comes from each clip's own valid length, not from the padded
        width of its batch. Deriving it from ``t`` made the set of global
        reference nodes depend on which other clips shared the batch, which
        moved a clip's logits by 0.034 between a padded and an unpadded run.
        """
        idx = torch.arange(t, device=device)
        local = ((idx[None, :] - idx[:, None]).abs() <= self.band)[None]
        stride = (lengths // max(1, self.n_global)).clamp_min(1)[:, None]
        pos = idx[None, :].expand(lengths.shape[0], t)
        glob = (pos % stride == 0) & (pos < lengths[:, None])
        return local | glob[:, None, :]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x (B, T, C)
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        shape = (b, t, self.heads, c // self.heads)
        q, k, v = (z.view(*shape).transpose(1, 2) for z in (q, k, v))
        lengths = (
            mask.sum(1) if mask is not None
            else torch.full((b,), t, device=x.device, dtype=torch.long)
        )
        allow = self._edges(t, lengths, x.device)
        if mask is not None:
            allow = allow & mask[:, None, :]
        # a padded query row would otherwise have every key masked out, and
        # softmax over an all -inf row is NaN; its output is discarded anyway
        eye = torch.eye(t, dtype=torch.bool, device=x.device)[None]
        allow = allow | eye
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None])
        return x + self.proj(out.transpose(1, 2).reshape(b, t, c))


class VideoLocalizer(nn.Module):
    """Cached-feature visual localizer. See the module docstring for variants."""

    def __init__(
        self,
        variant: str = "v2",
        dim: int = 256,
        n_blocks: int = 4,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        fine_blocks: int = 2,
        heads: int = 4,
        band: int = 4,
        n_global: int = 8,
        in_dim: int = EMBED,
    ) -> None:
        super().__init__()
        if variant not in ("v0", "v1", "v2", "v3"):
            raise ValueError(f"variant must be v0|v1|v2|v3, got {variant!r}")
        self.variant = variant
        self.use_spatial = variant in ("v2", "v3")
        self.use_graph = variant == "v3"
        self.upsample = 1 if variant == "v0" else 2

        if self.use_spatial:
            self.spatial = SpatialAttnPool(in_dim, dim, heads)
        else:
            self.spatial = None
            self.proj_in = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))

        # temporal difference: an explicit first difference, so a boundary is
        # visible to the first conv instead of having to be derived by it
        self.diff = nn.Conv1d(dim * 2, dim, 1)
        self.trunk = nn.ModuleList(
            TemporalBlock(dim, dilations[i % len(dilations)]) for i in range(n_blocks)
        )
        self.graph = TemporalGraphAttention(dim, heads, band, n_global) if self.use_graph else None

        if self.upsample == 2:
            self.up = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
            self.fine = nn.ModuleList(TemporalBlock(dim, 1) for _ in range(fine_blocks))
        else:
            self.up, self.fine = None, None

        self.head_frame = nn.Conv1d(dim, 1, 1)
        self.head_start = nn.Conv1d(dim, 1, 1)
        self.head_end = nn.Conv1d(dim, 1, 1)
        self.head_clip = nn.Linear(dim, 1)
        # boundaries are ~2% of positions; start near "no boundary here"
        nn.init.constant_(self.head_start.bias, -4.0)
        nn.init.constant_(self.head_end.bias, -4.0)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """``feats`` (B, T, V, 768) from the cache; ``mask`` (B, T) True where valid."""
        # cache channel 0 is the mean over all 196 patches; channels 1: are the
        # 4x4 region grid, which only the spatial variants read
        h = (
            self.spatial(feats[:, :, 1:])
            if self.use_spatial
            else self.proj_in(feats[:, :, 0])
        )

        if mask is not None:
            h = h * mask[..., None]

        x = h.transpose(1, 2)                      # (B, C, T)
        shifted = F.pad(x[:, :, :-1], (1, 0))
        x = self.diff(torch.cat([x, x - shifted], dim=1))
        # Re-zero padded positions after every op that has a bias. A conv maps
        # an all-zero padded position to its *bias*, not to zero, and the next
        # conv would read that nonzero value into the valid tail, so the mask
        # is re-applied rather than only applied to the input.
        if mask is not None:
            x = x * mask[:, None, :]

        for blk in self.trunk:
            x = blk(x)
            if mask is not None:
                x = x * mask[:, None, :]
        if self.graph is not None:
            x = self.graph(x.transpose(1, 2), mask).transpose(1, 2)
            if mask is not None:
                x = x * mask[:, None, :]

        clip_mask = mask[:, None, :].float() if mask is not None else torch.ones_like(x[:, :1])
        pooled = (x * clip_mask).sum(-1) / clip_mask.sum(-1).clamp_min(1.0)

        if self.up is not None:
            fine_mask = (
                mask.repeat_interleave(2, dim=1)[:, None, :] if mask is not None else None
            )
            y = self.up(x)
            # FPN by addition, not concatenation: v1 then differs from v0 by
            # the upsampling path alone, so the ladder measures resolution and
            # not a width change.
            y = y + F.interpolate(h.transpose(1, 2), size=y.shape[-1], mode="nearest")
            if fine_mask is not None:
                y = y * fine_mask
            for blk in self.fine:
                y = blk(y)
                if fine_mask is not None:
                    y = y * fine_mask
        else:
            y = x

        return {
            "frame_logits": self.head_frame(y).squeeze(1),
            "start_logits": self.head_start(y).squeeze(1),
            "end_logits": self.head_end(y).squeeze(1),
            "clip_logit": self.head_clip(pooled).squeeze(-1),
            # pre-head features on the 40 ms output grid, for the fusion model
            "features": y,
        }


def _build(variant: str):
    def fn(**kwargs) -> VideoLocalizer:
        kwargs.setdefault("variant", variant)
        return VideoLocalizer(**kwargs)
    return fn


for _v in ("v0", "v1", "v2", "v3"):
    register(f"vloc_{_v}")(_build(_v))
