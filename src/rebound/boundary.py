"""Timestamp-aware shared transformer with continuous start/end outputs."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .evidence import DirectedChangeProfile


@dataclass
class BoundaryConfig:
    evidence_dimension: int = 32
    width: int = 512
    heads: int = 8
    layers: int = 12
    attention_seconds: float = .5
    windows: tuple = (.08, .16, .32)
    anchors: int = 8
    checkpoint_blocks: bool = True
    offset_range: float = 1.0


class TimestampBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n1, self.n2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width))
        self.relative_slope = nn.Parameter(torch.zeros(cfg.heads))

    def forward(self, x, times, valid):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, self.cfg.heads, d // self.cfg.heads).permute(2, 0, 3, 1, 4)
        difference = times[:, None, :] - times[:, :, None]
        allow = (difference.abs() <= self.cfg.attention_seconds) & valid[:, None, :]
        fallback = torch.arange(t, device=x.device) == 0
        allow = allow | ((~valid)[:, :, None] & fallback[None, None])
        bias = self.relative_slope[None, :, None, None] * difference[:, None] / self.cfg.attention_seconds
        bias = bias.masked_fill(~allow[:, None], float('-inf')).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff(self.n2(x))
        return x.masked_fill(~valid[..., None], 0)


class SharedContinuousBoundary(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or BoundaryConfig()
        cfg = self.cfg
        self.profile = DirectedChangeProfile(cfg.windows, cfg.anchors)
        self.input = nn.Linear(cfg.evidence_dimension + len(cfg.windows) * 5, cfg.width)
        self.blocks = nn.ModuleList([TimestampBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.heads = nn.Linear(cfg.width, 3)
        self.offsets = nn.Linear(cfg.width, 2)
        with torch.no_grad():
            self.heads.bias[1:].fill_(-4.)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    def forward(self, z, times, valid):
        if times.ndim == 1:
            times = times[None].expand(len(z), -1)
        profile = self.profile(z.float(), times.float(), valid)
        x = self.input(torch.cat((z, profile.to(z.dtype)), -1))
        x = x.masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            if self.training and self.cfg.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(block, x, times, valid, use_reentrant=False)
            else:
                x = block(x, times, valid)
        x = self.norm(x)
        frame, start, end = self.heads(x).float().unbind(-1)
        start_offset, end_offset = (self.cfg.offset_range * self.offsets(x).float().tanh()).unbind(-1)
        if times.shape[1] < 2:
            raise ValueError('At least two evidence timestamps are required')
        hop = torch.cat((times[:, 1:] - times[:, :-1], times[:, -1:] - times[:, -2:-1]), -1)
        return {'frame_logits': frame, 'start_logits': start, 'end_logits': end,
                'start_offset': start_offset, 'end_offset': end_offset,
                'start_time': times + hop * start_offset, 'end_time': times + hop * end_offset,
                'times': times, 'valid': valid, 'features': x, 'change_profile': profile}

    def freeze(self):
        self.eval().requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        return self
