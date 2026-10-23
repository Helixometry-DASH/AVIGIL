"""Label construction.

Each metadata record describes one clip:

    file           str          path to the clip, relative to the dataset root
    split          str          which split the clip belongs to
    video_frames   int          number of video frames in the clip
    original       str | null   the real clip this one was derived from

and states what was manipulated, in either of two ways.

**Independent intervals.** When the audio and the video were manipulated over
different spans, give each modality its own list:

    audio_fake_periods    [[s, e]]   manipulated audio intervals, in seconds
    visual_fake_periods   [[s, e]]   manipulated video intervals, in seconds

Either may be empty or absent, and they need not overlap. A clip with audio
manipulated at 1-3 s and video at 5-7 s is expressed exactly, and each branch
is supervised only on its own list.

**Shared intervals.** When both modalities were manipulated over the same
spans, one list plus two flags is enough:

    fake_periods   [[s, e]]     manipulated intervals, in seconds
    modify_audio   bool         whether the audio was manipulated
    modify_video   bool         whether the video was manipulated

If the per-modality fields are present they take precedence, and the flags are
derived from them (a non-empty list means that modality was manipulated) so the
two can never contradict each other.

Three label views are derived from a record either way:

  frame_mask  (T,)  float32  1.0 where the frame falls inside a forged segment
  segments    (N,2) float32  forged intervals in *seconds*, sorted
  modality    int            0 real, 1 visual-only, 2 audio-only, 3 both

The joint view is the **union** of the two modality views, with overlapping
intervals merged so a region manipulated in both is counted once.

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


AUDIO_PERIODS = "audio_fake_periods"
VISUAL_PERIODS = "visual_fake_periods"


def _as_segments(periods) -> np.ndarray:
    """``[start, end]`` pairs as a sorted ``(N, 2)`` float32 array.

    Accepts a list, a tuple or an array, so it can be reused on values that
    have already been through here once.
    """
    if periods is None or len(periods) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    seg = np.asarray(periods, dtype=np.float32).reshape(-1, 2)
    return seg[np.argsort(seg[:, 0], kind="stable")]


def merge_segments(seg: np.ndarray) -> np.ndarray:
    """Union of possibly overlapping intervals, sorted and non-overlapping."""
    if len(seg) < 2:
        return _as_segments(seg)
    seg = _as_segments(seg)
    out = [list(seg[0])]
    for s, e in seg[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], float(e))
        else:
            out.append([float(s), float(e)])
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)


def has_per_modality(record: dict) -> bool:
    """Whether this record states audio and visual intervals separately."""
    return AUDIO_PERIODS in record or VISUAL_PERIODS in record


def modality_flags(record: dict) -> tuple[bool, bool]:
    """``(modify_audio, modify_video)``, from the per-modality lists if given."""
    if has_per_modality(record):
        return (bool(record.get(AUDIO_PERIODS)), bool(record.get(VISUAL_PERIODS)))
    return (bool(record.get("modify_audio")), bool(record.get("modify_video")))


def segments_for(record: dict, which: str = "joint") -> np.ndarray:
    """Forged intervals for ``which`` in ``{"audio", "visual", "joint"}``.

    With per-modality fields each view is that modality's own list, and the
    joint view is their merged union. Otherwise the shared ``fake_periods``
    list is gated by the flags.
    """
    if which not in ("audio", "visual", "joint"):
        raise ValueError(f"which must be audio|visual|joint, got {which!r}")

    if has_per_modality(record):
        aud = _as_segments(record.get(AUDIO_PERIODS))
        vis = _as_segments(record.get(VISUAL_PERIODS))
        if which == "audio":
            return aud
        if which == "visual":
            return vis
        return merge_segments(np.concatenate([aud, vis], axis=0))

    a, v = modality_flags(record)
    keep = {"audio": a, "visual": v, "joint": a or v}[which]
    return sorted_segments(record) if keep else np.zeros((0, 2), dtype=np.float32)


def modality_label(record: dict) -> int:
    a, v = modality_flags(record)
    if not v and not a:
        return MODALITY_REAL
    if v and not a:
        return MODALITY_VISUAL
    if a and not v:
        return MODALITY_AUDIO
    return MODALITY_BOTH


def sorted_segments(record: dict) -> np.ndarray:
    """The shared ``fake_periods`` list, shape (N, 2), sorted by start.

    Metadata is not required to list intervals in order, and every downstream
    IoU, NMS and mask computation assumes they are, so they are sorted here.
    Prefer :func:`segments_for`, which also handles per-modality intervals.
    """
    return _as_segments(record.get("fake_periods") or [])


def frame_mask(record: dict, fps: float = 25.0, n_frames: int | None = None,
               which: str = "joint") -> np.ndarray:
    """Per-frame forged indicator of length ``n_frames`` (default: video_frames).

    ``which`` selects the label view, so an audio-only or visual-only mask can
    be built from the same record.
    """
    total = int(n_frames if n_frames is not None else record["video_frames"])
    mask = np.zeros(total, dtype=np.float32)
    for start, end in segments_for(record, which):
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
