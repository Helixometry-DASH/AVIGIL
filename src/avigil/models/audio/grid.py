"""Exact sample <-> frame arithmetic for the audio output grid.

Targets and predictions must live on the same grid, so the mapping is derived
from the network's convolution arithmetic rather than assumed.

The encoder's temporal path for an input of N samples:

    SincConv (kernel 129, valid)      N - 128
    max_pool2d((3, 3))                floor(. / 3)
    residual block k = 1..6           floor(. / 3)     (convs keep T, pool /3)

so block k has ``floor((N - 128) / 3**(k+1))`` frames. The effective hop is
``3**(k+1)`` samples and the SincConv group delay is 64 samples (half of
129-1). A 64,600-sample input gives 265 frames at block 4 and 29 at block 6.
"""

from __future__ import annotations

import numpy as np

SINC_KERNEL = 129          # 128 is forced odd so the filters stay symmetric
SINC_DELAY = 64            # (SINC_KERNEL - 1) // 2
SAMPLE_RATE = 16000


def hop_samples(block: int) -> int:
    """Samples between adjacent frames at the output of residual ``block``."""
    return 3 ** (block + 1)


def n_frames(n_samples: int, block: int) -> int:
    """Frame count at the output of residual ``block`` for ``n_samples`` input."""
    return (n_samples - (SINC_KERNEL - 1)) // hop_samples(block)


def frame_seconds(block: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Grid spacing in seconds (block 4 -> 0.01519 s, block 6 -> 0.13669 s)."""
    return hop_samples(block) / sample_rate


def time_to_frame(t: float, block: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Continuous frame index for a timestamp, accounting for SincConv delay."""
    return (t * sample_rate - SINC_DELAY) / hop_samples(block)


def frame_to_time(f: float, block: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Inverse of :func:`time_to_frame`; frame centres in seconds."""
    return (f * hop_samples(block) + SINC_DELAY) / sample_rate


def segments_to_targets(
    segments: list[tuple[float, float]],
    n_samples: int,
    block: int = 4,
    sample_rate: int = SAMPLE_RATE,
    boundary_sigma: float = 1.5,
    clip_start: float = 0.0,
) -> dict[str, np.ndarray]:
    """Build frame / start / end targets plus an edge-truncation mask.

    ``segments`` are in seconds, relative to the start of this crop.

    The masks matter: when a crop cuts through a forged region the region is
    still fake, but the crop edge is NOT a real transition. Supervising a
    boundary there teaches the model that crop edges are manipulation
    boundaries. ``start_valid`` / ``end_valid`` are zeroed within
    ``boundary_sigma`` frames of a truncated edge.

    Boundary targets are Gaussian-softened rather than one-hot: at a 15.2 ms
    grid an exact-frame target is both unlearnable and needlessly harsh.
    """
    T = n_frames(n_samples, block)
    frame = np.zeros(T, dtype=np.float32)
    start = np.zeros(T, dtype=np.float32)
    end = np.zeros(T, dtype=np.float32)
    start_valid = np.ones(T, dtype=np.float32)
    end_valid = np.ones(T, dtype=np.float32)

    clip_end = clip_start + n_samples / sample_rate
    idx = np.arange(T, dtype=np.float32)

    for s, e in segments:
        fs = time_to_frame(s, block, sample_rate)
        fe = time_to_frame(e, block, sample_rate)
        lo = int(np.clip(np.floor(fs + 0.5), 0, T))
        hi = int(np.clip(np.floor(fe + 0.5), 0, T))
        if hi <= lo:
            hi = min(lo + 1, T)
        frame[lo:hi] = 1.0

        # a boundary is real only if it lies strictly inside the crop
        if s > clip_start + 1e-6:
            start = np.maximum(start, np.exp(-0.5 * ((idx - fs) / boundary_sigma) ** 2))
        else:
            start_valid[: max(1, int(boundary_sigma * 2))] = 0.0
        if e < clip_end - 1e-6:
            end = np.maximum(end, np.exp(-0.5 * ((idx - fe) / boundary_sigma) ** 2))
        else:
            end_valid[-max(1, int(boundary_sigma * 2)) :] = 0.0

    return {
        "frame": frame,
        "start": start,
        "end": end,
        "start_valid": start_valid,
        "end_valid": end_valid,
    }
