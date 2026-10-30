"""Genuine-only predictive towers, independent of source-forgery experts."""
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class GenuineConfig:
    feature_dim: int = 512
    width: int = 768
    heads: int = 12
    predictor_layers: int = 6
    covariance_rank: int = 16
    attention_window: int = 64
    mask_block: int = 8
    mask_folds: int = 2
    checkpoint_blocks: bool = True

    def dictionary(self):
        return asdict(self)

def positional(t, d, device, dtype):
    positions = torch.arange(t, device=device).float()[:, None]
    freq = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.) / d))
    phase = positions * freq
    result = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[:, :d]
    return result.to(dtype)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.n2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, valid, window=0):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        allowed = valid[:, None, None, :].expand(b, 1, t, t)
        if window:
            position = torch.arange(t, device=x.device)
            local = (position[:, None] - position[None, :]).abs() <= window
            allowed = allowed & local[None, None]
        # Torch 2.4 CUDA SDPA backward is not finite for all-masked rows.
        # Padded queries attend one valid key and are zeroed after the block.
        fallback = torch.arange(t, device=x.device) == 0
        allowed = allowed | ((~valid)[:, None, :, None] & fallback[None, None, None, :])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff(self.n2(x))
        return x.masked_fill(~valid[..., None], 0)


class Stack(nn.Module):
    def __init__(self, cfg, layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg.width, cfg.heads) for _ in range(layers)])
        self.use_checkpoint = cfg.checkpoint_blocks

    def forward(self, x, valid, window=0):
        for block in self.blocks:
            if self.training and self.use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(block, x, valid, window, use_reentrant=False)
            else:
                x = block(x, valid, window)
        return x


class PredictiveTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.project = nn.Linear(cfg.feature_dim, cfg.width)
        self.mask_token = nn.Parameter(torch.zeros(cfg.width))
        self.stack = Stack(cfg, cfg.predictor_layers)
        self.norm = nn.LayerNorm(cfg.width)
        d, r = cfg.feature_dim, cfg.covariance_rank
        self.mean = nn.Linear(cfg.width, d)
        self.log_diag = nn.Linear(cfg.width, d)
        self.factor = nn.Linear(cfg.width, d * r) if r else None

    def forward(self, context, valid, hidden=None):
        x = self.project(context)
        if hidden is not None:
            x = torch.where(hidden[..., None], self.mask_token[None, None, :], x)
        x = x + positional(x.shape[1], x.shape[-1], x.device, x.dtype)
        x = self.norm(self.stack(x, valid, self.cfg.attention_window))
        mean = self.mean(x)
        diag = self.log_diag(x).float().clamp(-5., 5.).exp()
        factor = self.factor(x).float().reshape(*mean.shape, self.cfg.covariance_rank) if self.factor else None
        if factor is not None:
            factor = factor / math.sqrt(self.cfg.feature_dim)
        return mean.float(), diag, factor


def gaussian_nll(target, mean, diag, factor):
    with torch.autocast(device_type=target.device.type, enabled=False):
        return _gaussian_nll(target, mean, diag, factor)


def _gaussian_nll(target, mean, diag, factor):
    delta = target.float() - mean.float()
    if factor is None:
        return .5 * ((delta.square() / diag).sum(-1) + diag.log().sum(-1) + target.shape[-1] * math.log(2 * math.pi))
    distribution = torch.distributions.LowRankMultivariateNormal(mean.float(), factor.float(), diag.float(), validate_args=False)
    return -distribution.log_prob(target.float())


def whiten(target, mean, diag, factor):
    """W Sigma W^T=I using a rank-r correction, without d-by-d factorization."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        return _whiten(target, mean, diag, factor)


def _whiten(target, mean, diag, factor):
    w = (target.float() - mean.float()) / diag.sqrt()
    if factor is None:
        return w
    b = factor.float() / diag.sqrt()[..., None]
    eigenvalues, vectors = torch.linalg.eigh(b.transpose(-1, -2) @ b)
    root = (1 + eigenvalues.clamp_min(0)).sqrt()
    coeff = 1 / (root * (root + 1))
    projected = (b.transpose(-1, -2) @ w[..., None])
    correction = b @ (vectors @ (coeff[..., None] * (vectors.transpose(-1, -2) @ projected)))
    return w - correction.squeeze(-1)


class GenuinePredictors(nn.Module):
    names = ('audio_self', 'video_self', 'audio_cross', 'video_cross')

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.towers = nn.ModuleDict({name: PredictiveTower(cfg) for name in self.names})
        for name in ('audio_cross', 'video_cross'):
            self.towers[name].mask_token.requires_grad_(False)

    def forward(self, features, valid, step=0):
        cfg = self.cfg
        t = features.shape[2]
        positions = torch.arange(t, device=features.device)
        hidden = (((positions // cfg.mask_block) % cfg.mask_folds) == (step % cfg.mask_folds))[None].expand(len(features), -1) & valid
        losses = []
        for m, label in enumerate(('audio', 'video')):
            for kind in ('self', 'cross'):
                context = features[:, m if kind == 'self' else 1 - m]
                mean, diag, factor = self.towers[label + '_' + kind](context, valid, hidden if kind == 'self' else None)
                nll = gaussian_nll(features[:, m], mean, diag, factor) / cfg.feature_dim
                selected = hidden if kind == 'self' else valid
                losses.append(nll[selected].mean())
        return torch.stack(losses).mean()

    def evidence(self, features, valid):
        """features [B,2,T,d], audio then video; deterministic block masks."""
        cfg = self.cfg
        b, _, t, d = features.shape
        residuals, surprises = [], []
        positions = torch.arange(t, device=features.device)
        for m, label in enumerate(('audio', 'video')):
            rs, ns = [], []
            for kind in ('self', 'cross'):
                target = features[:, m]
                context = features[:, m if kind == 'self' else 1 - m]
                tower = self.towers[label + '_' + kind]
                residual = torch.zeros_like(target, dtype=torch.float32)
                surprise = torch.zeros((b, t), device=target.device)
                # Cross predictor has no target input, so needs only one pass.
                folds = cfg.mask_folds if kind == 'self' else 1
                for fold in range(folds):
                    selected = (((positions // cfg.mask_block) % folds) == fold)[None].expand(b, -1) & valid
                    mean, diag, factor = tower(context, valid, selected if kind == 'self' else None)
                    residual = torch.where(selected[..., None], whiten(target, mean, diag, factor), residual)
                    nll = gaussian_nll(target, mean, diag, factor) / d
                    surprise = torch.where(selected, nll, surprise)
                rs.append(residual.masked_fill(~valid[..., None], 0))
                ns.append(surprise.masked_fill(~valid, 0))
            residuals.append(torch.stack(rs, 1))
            surprises.append(torch.stack(ns, 1))
        return torch.stack(residuals, 1), torch.stack(surprises, 1)


