"""Timestamp-aware alignment between the audio and video feature grids.

The two branches run at different rates and neither is a multiple of the other:

    audio (block 4)            hop 243 samples @ 16 kHz = 15.1875 ms = 65.84 Hz
    video (upsample 2)         hop 2 frames    @ 25 fps = 40.0 ms    = 25.00 Hz

Fusion happens on the **audio** grid, the finer of the two, so the joint head
can express a boundary to 15.19 ms. The coarser modality is interpolated up:
video 25 Hz -> audio 65.84 Hz.

Alignment is by timestamp, never by array index, so it is unaffected by how
much padding a tensor carries. Audio frame ``j`` is centred at
``(j * 243 + 64) / 16000`` seconds, where 64 is the SincConv group delay.
"""

from __future__ import annotations

import torch

from ..audio.grid import SINC_DELAY, hop_samples
from ..video.vgrid import hop_seconds


def audio_frame_times(n_frames: int, block: int = 4, sample_rate: int = 16000,
                      device=None) -> torch.Tensor:
    """Centre time, in seconds, of each audio feature frame."""
    j = torch.arange(n_frames, dtype=torch.float32, device=device)
    return (j * hop_samples(block) + SINC_DELAY) / sample_rate


def video_frame_times(n_frames: int, upsample: int = 2, device=None) -> torch.Tensor:
    """Centre time, in seconds, of each video output frame."""
    k = torch.arange(n_frames, dtype=torch.float32, device=device)
    return (k + 0.5) * hop_seconds(upsample)


def resample_to_grid(
    feats: torch.Tensor,
    src_times: torch.Tensor,
    dst_times: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linearly interpolate ``(B, C, T_src)`` onto ``dst_times``.

    Linear rather than nearest: at 65.84 -> 25 Hz each destination step spans
    about 2.6 source frames, so nearest-neighbour would discard roughly 60% of
    the audio evidence and alias the boundary peaks it is meant to carry.

    Positions outside the source span are clamped to the end values and should
    be masked off by the caller.
    """
    b, c, t_src = feats.shape
    if t_src == 0:
        return feats.new_zeros(b, c, dst_times.numel())
    if mask is not None:
        feats = feats * mask[:, None, :]

    pos = torch.searchsorted(src_times.contiguous(), dst_times.contiguous())
    hi = pos.clamp(1, t_src - 1) if t_src > 1 else pos.clamp(0, 0)
    lo = (hi - 1).clamp(0, t_src - 1)
    t_lo, t_hi = src_times[lo], src_times[hi]
    w = torch.where(t_hi > t_lo, (dst_times - t_lo) / (t_hi - t_lo), torch.zeros_like(dst_times))
    w = w.clamp(0.0, 1.0).view(1, 1, -1)
    return feats[:, :, lo] * (1.0 - w) + feats[:, :, hi] * w


def align_audio_to_video(
    audio_feats: torch.Tensor,
    n_video_frames: int,
    block: int = 4,
    upsample: int = 2,
    sample_rate: int = 16000,
    audio_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B, C, T_audio)`` -> ``(B, C, n_video_frames)`` on the 25 Hz video grid.

    Used when ``fuse_grid="video"``; the default fuses on the audio grid.
    """
    dev = audio_feats.device
    src = audio_frame_times(audio_feats.shape[-1], block, sample_rate, dev)
    dst = video_frame_times(n_video_frames, upsample, dev)
    return resample_to_grid(audio_feats, src, dst, audio_mask)


def align_video_to_audio(
    video_feats: torch.Tensor,
    n_audio_frames: int,
    block: int = 4,
    upsample: int = 2,
    sample_rate: int = 16000,
    video_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B, C, T_video)`` -> ``(B, C, n_audio_frames)`` on the 15.19 ms grid.

    This is the default direction. Interpolating the coarser modality up adds
    no information, but it keeps the joint head on the grid that can express
    the audio branch's 7.3 ms boundaries.
    """
    dev = video_feats.device
    src = video_frame_times(video_feats.shape[-1], upsample, dev)
    dst = audio_frame_times(n_audio_frames, block, sample_rate, dev)
    return resample_to_grid(video_feats, src, dst, video_mask)
