"""Video grid arithmetic.

At 25 fps a video frame is 40 ms and a VideoMAE tubelet (2 frames) is 80 ms.
The x2 upsampling decoder brings predictions from the 80 ms feature grid onto
the 40 ms frame grid.

Segment boundaries in the metadata are given in seconds and need not be
frame-aligned, so the decoder estimates each boundary as the
probability-weighted centroid of the boundary head rather than snapping it to
a grid point. That yields a continuous time estimate; see
``proposals._soft_index``.
"""

from __future__ import annotations

import numpy as np

FPS = 25.0
TUBELET = 2


def hop_seconds(upsample: int = 2) -> float:
    """Output grid spacing. ``upsample=2`` -> 0.04 s, ``upsample=1`` -> 0.08 s."""
    return TUBELET / (FPS * upsample)


def n_out(n_positions: int, upsample: int = 2) -> int:
    return int(n_positions) * int(upsample)


def time_to_frame(t: float, upsample: int = 2) -> float:
    return float(t) / hop_seconds(upsample)


def frame_to_time(f: float, upsample: int = 2) -> float:
    return float(f) * hop_seconds(upsample)


def iou_ceiling(length: float, upsample: int = 2) -> float:
    """Best IoU a grid-snapped prediction can reach for a segment of ``length``."""
    g = hop_seconds(upsample)
    return length / (length + g)


def segments_to_targets(
    segments,
    n_positions: int,
    upsample: int = 2,
    boundary_sigma: float = 1.5,
    clip_start: float = 0.0,
) -> dict[str, np.ndarray]:
    """Dense frame / start / end targets on the uniform output grid.

    ``segments`` are seconds relative to the start of this crop. Output index
    ``k`` covers ``[k*hop, (k+1)*hop)``, matching ``labels.frame_mask`` so the
    video task and the metadata helpers cannot drift apart.

    ``start_valid`` / ``end_valid`` are zeroed near a crop edge that truncated
    a segment: the region is still forged there, but the crop edge is not a
    manipulation boundary and must not be supervised as one.
    """
    T = n_out(n_positions, upsample)
    hop = hop_seconds(upsample)
    frame = np.zeros(T, dtype=np.float32)
    start = np.zeros(T, dtype=np.float32)
    end = np.zeros(T, dtype=np.float32)
    start_valid = np.ones(T, dtype=np.float32)
    end_valid = np.ones(T, dtype=np.float32)

    clip_end = clip_start + T * hop
    idx = np.arange(T, dtype=np.float32)
    guard = max(1, int(boundary_sigma * 2))

    for s, e in segments:
        # float64 and round-half-up, explicitly. A third of all visual
        # boundaries land exactly on a half grid step (68.1% align to 20 ms but
        # only 34.9% to a 40 ms frame), so with plain round() the mask edge is
        # decided by banker's rounding and by whether the caller passed float32
        # or float64 -- the same segment would quantise differently depending on
        # dtype. floor(x + 0.5) is one deterministic rule for every boundary.
        fs, fe = float(s) / hop, float(e) / hop
        lo = int(np.clip(np.floor(fs + 0.5), 0, T))
        hi = int(np.clip(np.floor(fe + 0.5), 0, T))
        if hi <= lo:                      # segment shorter than one grid step
            hi = min(lo + 1, T)
        frame[lo:hi] = 1.0

        if s > clip_start + 1e-6:
            start = np.maximum(start, np.exp(-0.5 * ((idx - fs) / boundary_sigma) ** 2))
        else:
            start_valid[:guard] = 0.0
        if e < clip_end - 1e-6:
            end = np.maximum(end, np.exp(-0.5 * ((idx - fe) / boundary_sigma) ** 2))
        else:
            end_valid[-guard:] = 0.0

    return {
        "frame": frame,
        "start": start,
        "end": end,
        "start_valid": start_valid,
        "end_valid": end_valid,
    }
