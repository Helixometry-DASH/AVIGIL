import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DenseCrossModalBoundary(nn.Module):
    """Dense 768-dim Multi-head Cross-Attention + Temporal Conv BiGRU."""
    def __init__(self, in_dim=768, hidden_dim=256, conv_dim=128, dropout=0.1):
        super().__init__()
        self.register_buffer('mean_a', torch.zeros(in_dim))
        self.register_buffer('scale_a', torch.ones(in_dim))
        self.register_buffer('mean_v', torch.zeros(in_dim))
        self.register_buffer('scale_v', torch.ones(in_dim))

        self.proj_a = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.cross_a2v = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.cross_v2a = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)

        feature_width = hidden_dim * 4 + 1  # 256*4 + 1 = 1025
        self.conv = nn.Sequential(
            nn.Dropout1d(dropout),
            nn.Conv1d(feature_width, conv_dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, 5, padding=8, dilation=4), nn.GELU()
        )
        self.gru = nn.GRU(conv_dim, conv_dim // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(conv_dim)
        self.heads = nn.Conv1d(conv_dim, 3, 1)
        self.offsets = nn.Conv1d(conv_dim, 2, 1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)

    @torch.no_grad()
    def fit_standardization(self, real_a_list, real_v_list):
        all_a = torch.cat(real_a_list, dim=0)
        all_v = torch.cat(real_v_list, dim=0)
        self.mean_a.copy_(all_a.mean(0))
        self.scale_a.copy_(all_a.std(0).clamp_min(1e-3))
        self.mean_v.copy_(all_v.mean(0))
        self.scale_v.copy_(all_v.std(0).clamp_min(1e-3))

    def forward(self, a, v, times, valid):
        # Normalize
        a_norm = ((a - self.mean_a) / self.scale_a).masked_fill(~valid[..., None], 0)
        v_norm = ((v - self.mean_v) / self.scale_v).masked_fill(~valid[..., None], 0)

        # Linear projections
        pa = self.proj_a(a_norm)
        pv = self.proj_v(v_norm)

        # Cross-Attention
        key_padding_mask = ~valid
        a_recon, _ = self.cross_a2v(query=pa, key=pv, value=pv, key_padding_mask=key_padding_mask)
        v_recon, _ = self.cross_v2a(query=pv, key=pa, value=pa, key_padding_mask=key_padding_mask)

        # Discrepancy signals
        diff_a = pa - a_recon
        diff_v = pv - v_recon
        cos = (F.normalize(pa, dim=-1) * F.normalize(pv, dim=-1)).sum(-1, keepdim=True)
        feat = torch.cat([diff_a, diff_v, diff_a.abs(), diff_v.abs(), cos], dim=-1)
        feat = feat.masked_fill(~valid[..., None], 0)

        # Conv + BiGRU
        c = self.conv(feat.transpose(1, 2)).transpose(1, 2)
        g, _ = self.gru(c)
        h = self.norm(c + g).transpose(1, 2)

        frame, start, end = self.heads(h).unbind(1)
        so, eo = self.offsets(h).tanh().unbind(1)
        hop = torch.cat((times[:, 1:] - times[:, :-1], times[:, -1:] - times[:, -2:-1]), -1)

        return {
            'frame_logits': frame, 'start_logits': start, 'end_logits': end,
            'start_offset': so, 'end_offset': eo,
            'start_time': times + hop * so, 'end_time': times + hop * eo,
            'times': times, 'valid': valid
        }

