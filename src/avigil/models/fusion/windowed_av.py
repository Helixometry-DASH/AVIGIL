"""Windowed audio encoding for the multimodal model.

Crop for computation, retain the whole recording for prediction. The audio
encoder is a fixed-window model, so it is slid over the recording in 4.0375 s
windows and its outputs are reassembled on the recording's own frame grid.

The **fusion** then runs once over that whole assembled timeline rather than
per window, so the joint head's temporal context spans the recording instead
of a single 4 s window.

The video side needs nothing here: the backbone already ran on 16-frame
windows with an 8-frame hop at extraction time, and those were blended into
one continuous 80 ms sequence spanning the clip.
"""

from __future__ import annotations

import numpy as np
import torch

from ...engine.windowed import edge_weights, window_frame_starts
from ..audio.grid import hop_samples, n_frames

LOGIT_KEYS = ("frame_logits", "start_logits", "end_logits")


@torch.no_grad()
def assemble_audio(
    branch,
    audio: torch.Tensor,
    block: int = 4,
    window: int = 64600,
    hop_frames: int = 132,
    batch_windows: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slide ``branch`` over one recording; return ``(features, logits)``.

    ``features`` is ``(1, C, T)`` and ``logits`` ``(1, 3, T)`` on the full
    audio frame grid, blended over overlaps with a triangular weight that is
    lowest at window edges -- a frame near an edge saw only half its
    convolutional context.
    """
    device = audio.device
    n = int(audio.shape[-1])
    total = max(n_frames(n, block), 1)
    t_win = n_frames(window, block)
    starts = [p * hop_samples(block) for p in window_frame_starts(total, t_win, hop_frames)]
    wts = edge_weights(t_win, device)

    feats = logits = None
    wsum = torch.zeros(total, device=device)
    for i in range(0, len(starts), batch_windows):
        chunk = starts[i : i + batch_windows]
        pieces = []
        for s in chunk:
            seg = audio[..., s : s + window].reshape(-1)
            if seg.numel() < window:
                seg = torch.nn.functional.pad(seg, (0, window - seg.numel()))
            pieces.append(seg)
        out = branch(torch.stack(pieces))
        f = out["features"]                                   # (B, C, t_win)
        lg = torch.stack([out[k] for k in LOGIT_KEYS], dim=1)  # (B, 3, t_win)
        if feats is None:
            feats = torch.zeros(1, f.shape[1], total, device=device, dtype=f.dtype)
            logits = torch.zeros(1, 3, total, device=device, dtype=lg.dtype)
        for b, s in enumerate(chunk):
            g0 = s // hop_samples(block)
            g1 = min(g0 + t_win, total)
            m = g1 - g0
            if m <= 0:
                continue
            w = wts[:m]
            feats[0, :, g0:g1] += f[b, :, :m] * w
            logits[0, :, g0:g1] += lg[b, :, :m] * w
            wsum[g0:g1] += w
    d = wsum.clamp_min(1e-6)
    return feats / d, logits / d


@torch.no_grad()
def _audio_clip(branch, audio, block, window, hop_frames, batch_windows) -> torch.Tensor:
    """Max clip logit over the audio windows -- any suspicious window counts."""
    n = int(audio.shape[-1])
    total = max(n_frames(n, block), 1)
    t_win = n_frames(window, block)
    starts = [p * hop_samples(block)
              for p in window_frame_starts(total, t_win, hop_frames)]
    best = None
    for i in range(0, len(starts), batch_windows):
        pieces = []
        for s in starts[i : i + batch_windows]:
            seg = audio[..., s : s + window].reshape(-1)
            if seg.numel() < window:
                seg = torch.nn.functional.pad(seg, (0, window - seg.numel()))
            pieces.append(seg)
        lg = branch(torch.stack(pieces))["clip_logit"]
        m = lg.max()
        best = m if best is None else torch.maximum(best, m)
    return best


@torch.no_grad()
def predict_av_full(
    model,
    audio: torch.Tensor,
    video_feats: torch.Tensor,
    video_mask: torch.Tensor | None = None,
    block: int = 4,
    window: int = 64600,
    hop_frames: int = 132,
    batch_windows: int = 16,
) -> dict[str, np.ndarray]:
    """One whole recording -> joint per-frame probabilities on the audio grid.

    ``audio`` is ``(1, S)`` and ``video_feats`` ``(1, P, V, 768)`` for the same
    recording.
    """
    a_feats, a_logits = assemble_audio(
        model.audio_branch, audio, block, window, hop_frames, batch_windows
    )
    ov = model.video_branch(video_feats, video_mask)
    v_logits = torch.stack([ov[k] for k in LOGIT_KEYS], dim=1)
    # the audio branch's clip verdict, from the most suspicious window
    oa_clip = _audio_clip(model.audio_branch, audio, block, window, hop_frames, batch_windows)
    out = model.fuse_assembled(a_feats, a_logits, ov["features"], v_logits)
    res = {k: torch.sigmoid(out[k].float())[0].cpu().numpy() for k in LOGIT_KEYS}
    # the two branches' own verdicts, each on its own grid. The fused result is
    # the union view ("is anything forged here"); these say WHICH modality, and
    # they are what a caller needs to report audio and video separately.
    ap = torch.sigmoid(a_logits.float())[0].cpu().numpy()
    vp = torch.sigmoid(v_logits.float())[0].cpu().numpy()
    res["audio_branch"] = {k: ap[i] for i, k in enumerate(LOGIT_KEYS)}
    res["video_branch"] = {k: vp[i] for i, k in enumerate(LOGIT_KEYS)}
    res["audio_clip"] = float(torch.sigmoid(oa_clip))
    res["video_clip"] = float(torch.sigmoid(ov["clip_logit"].float())[0])
    res["clip"] = float(torch.sigmoid(out["clip_logit"].float())[0])
    res["modality"] = out["modality_logits"][0].float().cpu().numpy()
    res["n_windows"] = len(
        window_frame_starts(max(n_frames(int(audio.shape[-1]), block), 1),
                            n_frames(window, block), hop_frames)
    )
    return res
