"""AVIGIL: joint audio-visual forgery localization.

Runs the audio and the visual branch over the same time span, aligns their
feature sequences onto one grid, and predicts jointly from the pair. The two
unimodal heads keep predicting their own modality alongside the joint head, so
every forward pass returns three sets of intervals: audio, visual, and joint.

The joint head predicts on the **audio** grid (15.19 ms), the finer of the
two, and the visual sequence is interpolated up to meet it. See
:mod:`avigil.models.fusion.align`.

Four outputs per clip:

    frame / start / end logits   dense, on the audio grid
    clip logit                   P(the clip contains any forgery)
    modality logits              real / visual-only / audio-only / both

Variants, each adding one component:

    m3  aligned features concatenated, then a temporal conv trunk
    m4  + bidirectional cross-modal attention
    m5  + reliability gating on the cross-modal messages   <- released model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register
from ..video.model import ChannelNorm, TemporalBlock
from .align import align_audio_to_video, align_video_to_audio

N_MODALITY = 4          # real / visual-only / audio-only / both


class CrossModalAttention(nn.Module):
    """One direction of cross-modal attention, with an optional reliability gate.

    The gate guards against modality interference. When a clip is manipulated
    in one modality only, the other modality is genuine and should not pull the
    representation toward "fake". The gate is a learned scalar per time step
    applied to the incoming cross-modal message, so the model can scale that
    message down to zero and attend to its own modality alone.
    """

    def __init__(self, dim: int, heads: int = 4, gated: bool = True) -> None:
        super().__init__()
        self.heads, self.gated = heads, gated
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim * 3, dim // 4), nn.GELU(),
                                      nn.Linear(dim // 4, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # x, y: (B, T, C) already on the same grid
        b, t, c = x.shape
        q = self.q(self.norm_q(x)).view(b, t, self.heads, -1).transpose(1, 2)
        k, v = self.kv(self.norm_kv(y)).chunk(2, dim=-1)
        k = k.view(b, t, self.heads, -1).transpose(1, 2)
        v = v.view(b, t, self.heads, -1).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :] | torch.eye(
                t, dtype=torch.bool, device=x.device)[None, None]
        msg = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        msg = self.proj(msg.transpose(1, 2).reshape(b, t, c))
        if self.gated:
            g = torch.sigmoid(self.gate(torch.cat([x, y, msg], dim=-1)))
            msg = g * msg
        return x + msg


class AVIGIL(nn.Module):
    """Joint audio-visual localizer over the two unimodal branches."""

    def __init__(
        self,
        audio_branch: nn.Module,
        video_branch: nn.Module,
        variant: str = "m5",
        dim: int = 256,
        audio_dim: int = 128,
        video_dim: int = 256,
        n_blocks: int = 4,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        heads: int = 4,
        fuse_grid: str = "audio",
        freeze_branches: bool = True,
    ) -> None:
        super().__init__()
        if variant not in ("m3", "m4", "m5"):
            raise ValueError(f"variant must be m3|m4|m5, got {variant!r}")
        if fuse_grid not in ("audio", "video"):
            raise ValueError(f"fuse_grid must be audio|video, got {fuse_grid!r}")
        self.variant, self.fuse_grid = variant, fuse_grid
        self.use_cross = variant in ("m4", "m5")
        self.use_gate = variant == "m5"

        self.audio_branch, self.video_branch = audio_branch, video_branch
        if freeze_branches:
            for p in self.audio_branch.parameters():
                p.requires_grad_(False)
            for p in self.video_branch.parameters():
                p.requires_grad_(False)
        self.frozen = freeze_branches

        # +3 channels per modality: the branch's own frame/start/end logits are
        # its forensic verdict and are cheaper to read than to re-derive
        self.proj_a = nn.Conv1d(audio_dim + 3, dim, 1)
        self.proj_v = nn.Conv1d(video_dim + 3, dim, 1)
        self.norm_a, self.norm_v = ChannelNorm(dim), ChannelNorm(dim)

        if self.use_cross:
            self.a2v = CrossModalAttention(dim, heads, gated=self.use_gate)
            self.v2a = CrossModalAttention(dim, heads, gated=self.use_gate)

        self.fuse = nn.Conv1d(dim * 2, dim, 1)
        self.trunk = nn.ModuleList(
            TemporalBlock(dim, dilations[i % len(dilations)]) for i in range(n_blocks)
        )
        self.head_frame = nn.Conv1d(dim, 1, 1)
        self.head_start = nn.Conv1d(dim, 1, 1)
        self.head_end = nn.Conv1d(dim, 1, 1)
        self.head_clip = nn.Linear(dim, 1)
        self.head_modality = nn.Linear(dim, N_MODALITY)
        nn.init.constant_(self.head_start.bias, -4.0)
        nn.init.constant_(self.head_end.bias, -4.0)

    def branches(self, audio, video_feats, video_mask):
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            oa = self.audio_branch(audio)
            ov = self.video_branch(video_feats, video_mask)
        return oa, ov

    def forward(self, audio, video_feats, video_mask=None) -> dict:
        oa, ov = self.branches(audio, video_feats, video_mask)

        def stack(o):
            return torch.stack([o["frame_logits"], o["start_logits"], o["end_logits"]], dim=1)

        out = self.fuse_assembled(
            oa["features"], stack(oa), ov["features"], stack(ov)
        )
        out["audio"], out["video"] = oa, ov
        return out

    def fuse_assembled(self, a_feats, a_logits, v_feats, v_logits) -> dict:
        """Fuse already-computed branch outputs.

        Separated from :meth:`forward` so that evaluation can slide the audio
        encoder over a whole recording, reassemble its features on the full
        frame grid, and then fuse **once over that entire timeline**. Running
        the fusion per window instead would cap its temporal context at the
        4.0375 s encoder window, which is the opposite of the point: the graph
        is supposed to receive the assembled sequence spanning the recording.
        """
        a = torch.cat([a_feats, a_logits], dim=1)                  # (B, 128+3, T_a)
        v = torch.cat([v_feats, v_logits], dim=1)                  # (B, 256+3, T_v)

        if self.fuse_grid == "audio":
            t_out = a.shape[-1]
            v = align_video_to_audio(v, t_out)
        else:
            t_out = v.shape[-1]
            a = align_audio_to_video(a, t_out)

        ha = self.norm_a(self.proj_a(a)).transpose(1, 2)            # (B, T, C)
        hv = self.norm_v(self.proj_v(v)).transpose(1, 2)
        if self.use_cross:
            ha, hv = self.a2v(ha, hv), self.v2a(hv, ha)

        x = self.fuse(torch.cat([ha, hv], dim=-1).transpose(1, 2))  # (B, C, T)
        for blk in self.trunk:
            x = blk(x)
        pooled = x.mean(dim=2)

        return {
            "frame_logits": self.head_frame(x).squeeze(1),
            "start_logits": self.head_start(x).squeeze(1),
            "end_logits": self.head_end(x).squeeze(1),
            "clip_logit": self.head_clip(pooled).squeeze(-1),
            "modality_logits": self.head_modality(pooled),
        }


def _build(variant: str):
    def fn(**kwargs) -> AVIGIL:
        kwargs.setdefault("variant", variant)
        return AVIGIL(**kwargs)
    return fn


for _v in ("m3", "m4", "m5"):
    register(f"avigil_{_v}")(_build(_v))
