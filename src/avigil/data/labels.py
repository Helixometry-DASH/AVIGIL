"""Label construction.

Each metadata record describes one clip:

    file           str          path to the clip, relative to the dataset root
    split          str          which split the clip belongs to
    video_frames   int          number of video frames in the clip
    fake_periods   [[s, e]]     manipulated intervals, in seconds
    modify_audio   bool         whether the audio was manipulated
    modify_video   bool         whether the video was manipulated
    original       str | null   the real clip this one was derived from

Three label views are derived from one such record:

  frame_mask  (T,)  float32  1.0 where the frame falls inside a forged segment
  segments    (N,2) float32  forged intervals in *seconds*, sorted
  modality    int            0 real, 1 visual-only, 2 audio-only, 3 both

The frame grid is taken from ``video_frames`` rather than from a duration
times a frame rate, because a container's reported duration can disagree with
its video track's by a frame or two and converting with it would shift every
label.
"""

from __future__ import annotations

import numpy as np

MODALITY_REAL = 0
MODALITY_VISUAL = 1
MODALITY_AUDIO = 2
MODALITY_BOTH = 3
MODALITY_NAMES = ("real", "visual_only", "audio_only", "both")


def modality_label(record: dict) -> int:
    v, a = bool(record["modify_video"]), bool(record["modify_audio"])
    if not v and not a:
        return MODALITY_REAL
    if v and not a:
        return MODALITY_VISUAL
    if a and not v:
        return MODALITY_AUDIO
    return MODALITY_BOTH


def sorted_segments(record: dict) -> np.ndarray:
    """Forged intervals in seconds, shape (N, 2), sorted by start.

    ``fake_periods`` is NOT sorted in the source metadata -- e.g.
    ``test/000070.mp4`` carries ``[[11.3, 12.174], [10.2, 10.864]]``. Sorting
    here keeps every downstream IoU / NMS / mask computation honest.
    """
    periods = record.get("fake_periods") or []
    if not periods:
        return np.zeros((0, 2), dtype=np.float32)
    seg = np.asarray(periods, dtype=np.float32).reshape(-1, 2)
    return seg[np.argsort(seg[:, 0], kind="stable")]


def frame_mask(record: dict, fps: float = 25.0, n_frames: int | None = None) -> np.ndarray:
    """Per-frame forged indicator of length ``n_frames`` (default: video_frames)."""
    total = int(n_frames if n_frames is not None else record["video_frames"])
    mask = np.zeros(total, dtype=np.float32)
    for start, end in sorted_segments(record):
        lo = int(np.clip(round(start * fps), 0, total))
        hi = int(np.clip(round(end * fps), 0, total))
        if hi <= lo:                      # segment shorter than one frame
            hi = min(lo + 1, total)
        mask[lo:hi] = 1.0
    return mask


def mask_to_segments(
    mask: np.ndarray, fps: float = 25.0, min_seconds: float = 0.0
) -> list[tuple[float, float]]:
    """Inverse of :func:`frame_mask` -- contiguous runs back to second intervals.

    Used to turn model predictions into segments for AP/AR scoring.
    """
    binary = np.asarray(mask).astype(bool).astype(np.int8)
    if binary.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], binary, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    out = []
    for s, e in zip(starts, ends, strict=True):
        span = (float(s) / fps, float(e) / fps)
        if span[1] - span[0] >= min_seconds:
            out.append(span)
    return out
