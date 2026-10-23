"""Turn dense frame/start/end predictions into timed intervals.

Two generators, both scored by the frame evidence they enclose:

``region``    threshold the frame probabilities, take contiguous runs, then
              snap each edge to the nearest start/end peak. Robust, and the
              only generator that works for A0 (29 frames, no boundary detail).
``boundary``  pair every start peak with every later end peak, keep pairs whose
              duration is plausible, and score by interior frame probability.
              This is what can beat the frame grid at high IoU.

Both are followed by temporal NMS, and ``max_proposals`` caps how many
survive per clip.
"""

from __future__ import annotations

import numpy as np

from ...metrics.localization import nms
from .grid import frame_to_time


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0 or not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _peaks(x: np.ndarray, threshold: float, min_distance: int = 2) -> list[int]:
    """Local maxima above ``threshold``, greedily thinned by distance."""
    cand = [i for i in range(len(x)) if x[i] >= threshold]
    cand.sort(key=lambda i: -x[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_distance for j in kept):
            kept.append(i)
    return sorted(kept)


def _soft_index(x: np.ndarray, centre: int, w: int, floor: float = 0.05) -> float:
    """Probability-weighted centroid of ``x`` in a +/-w window around ``centre``.

    Returns a *fractional* frame index, so a boundary that falls between two
    frames is expressed as a continuous time rather than snapped to one. Since
    the metadata specifies boundaries to 1 ms while the grids are 15.19 ms and
    40 ms, this also removes the half-frame bias that snapping introduces.

    ``floor`` drops near-zero tail mass that would otherwise drag the centroid
    toward the window centre.
    """
    lo = max(0, centre - w)
    hi = min(len(x), centre + w + 1)
    seg = x[lo:hi]
    if seg.size == 0:
        return float(centre)
    peak = seg.max()
    if peak <= 0:
        return float(centre)
    m = seg >= max(floor, floor * peak)
    if not m.any():
        return float(lo + int(np.argmax(seg)))
    idx = np.arange(lo, hi, dtype=np.float64)[m]
    wt = seg[m].astype(np.float64)
    return float((idx * wt).sum() / wt.sum())


def decode_proposals(
    frame_prob: np.ndarray,
    start_prob: np.ndarray | None = None,
    end_prob: np.ndarray | None = None,
    block: int = 4,
    to_seconds=None,
    mode: str = "region",
    refine: str = "soft",
    frame_threshold: float = 0.5,
    peak_threshold: float = 0.2,
    min_seconds: float = 0.15,
    max_seconds: float = 2.0,
    nms_iou: float = 0.3,
    max_proposals: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (segments (N,2) in seconds, scores (N,)).

    ``to_seconds`` maps a frame index to seconds. It defaults to the encoder
    sinc grid selected by ``block``; the video head passes a uniform-grid
    mapping instead, so one decoder serves both modalities.
    """
    if to_seconds is None:
        def to_seconds(f, _b=block):
            return frame_to_time(f, _b)

    segs: list[tuple[float, float]] = []
    scores: list[float] = []

    if mode == "region" or start_prob is None or end_prob is None:
        for lo, hi in _runs(frame_prob >= frame_threshold):
            run_lo, run_hi = lo, hi            # keep integers for scoring
            fine_lo, fine_hi = float(lo), float(hi)
            if start_prob is not None and end_prob is not None:
                w = 3
                # Both targets are Gaussians centred ON the boundary, and
                # ``_runs`` returns a half-open span, so the start peak sits at
                # ``lo`` and the end peak at ``hi``: both are refined in a
                # +/-w window around those indices with no extra offset.
                if refine == "soft":
                    fine_lo = _soft_index(start_prob, lo, w)
                    fine_hi = _soft_index(end_prob, hi, w)
                elif refine == "argmax":
                    a = max(0, lo - w), min(len(start_prob), lo + w + 1)
                    b = max(0, hi - w), min(len(end_prob), hi + w + 1)
                    if start_prob[a[0] : a[1]].size:
                        fine_lo = float(a[0] + int(np.argmax(start_prob[a[0] : a[1]])))
                    if end_prob[b[0] : b[1]].size:
                        fine_hi = float(b[0] + int(np.argmax(end_prob[b[0] : b[1]])))
                elif refine != "none":
                    raise ValueError(f"refine must be soft|argmax|none, got {refine!r}")
            if fine_hi <= fine_lo:
                fine_lo, fine_hi = float(run_lo), float(run_hi)
            t0, t1 = to_seconds(fine_lo), to_seconds(fine_hi)
            if t1 - t0 >= min_seconds:
                segs.append((t0, t1))
                scores.append(
                    float(frame_prob[run_lo:run_hi].mean()) if run_hi > run_lo else 0.0
                )

    elif mode == "boundary":
        starts = _peaks(start_prob, peak_threshold)
        ends = _peaks(end_prob, peak_threshold)
        for i in starts:
            for j in ends:
                if j <= i:
                    continue
                if refine == "soft":
                    t0, t1 = to_seconds(_soft_index(start_prob, i, 2)), to_seconds(
                        _soft_index(end_prob, j, 2)
                    )
                else:
                    t0, t1 = to_seconds(i), to_seconds(j)
                if not (min_seconds <= t1 - t0 <= max_seconds):
                    continue
                inner = frame_prob[i : j + 1]
                if inner.size == 0:
                    continue
                # geometric mean of the two boundary confidences x interior evidence
                score = float(np.sqrt(start_prob[i] * end_prob[j]) * inner.mean())
                segs.append((t0, t1))
                scores.append(score)
    else:
        raise ValueError(f"mode must be region|boundary, got {mode!r}")

    if not segs:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    seg_arr = np.asarray(segs, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    seg_arr, score_arr = nms(seg_arr, score_arr, iou_threshold=nms_iou)
    order = np.argsort(-score_arr)[:max_proposals]
    return seg_arr[order].astype(np.float32), score_arr[order].astype(np.float32)
