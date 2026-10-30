import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class RealSyncNet(nn.Module):
    """Self-supervised contrastive audio-visual sync detector trained ONLY on Real clips."""
    def __init__(self, in_dim=768, embed_dim=128, temp=0.07):
        super().__init__()
        self.temp = temp
        self.proj_a = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # Learnable temperature and scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temp))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward_features(self, a, v):
        """a, v: (B, T, 768) -> norm_a, norm_v: (B, T, embed_dim)"""
        za = self.proj_a(a)
        zv = self.proj_v(v)
        za = F.normalize(za, p=2, dim=-1)
        zv = F.normalize(zv, p=2, dim=-1)
        return za, zv

    def sync_score(self, a, v):
        """Frame-level synchrony score in [0, 1]. High = sync (Real), Low = desync (Forged)."""
        za, zv = self.forward_features(a, v)
        cos = (za * zv).sum(dim=-1)  # (B, T)
        scale = self.logit_scale.exp().clamp(max=100.0)
        prob = torch.sigmoid(cos * scale + self.bias)
        return prob, cos

