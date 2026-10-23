"""Clip-level metrics: EER and ROC AUC over per-clip forgery scores.

These describe detection ("is this clip manipulated at all"), separately from
the localization metrics in :mod:`avigil.metrics.localization`.
"""

from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via rank statistic -- no sklearn dependency, handles ties."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold that achieves it.

    ``labels``: 1 = fake (positive). Returns ``(eer, threshold)``.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    if labels.all() or not labels.any():
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    fnr = 1.0 - tp / n_pos          # misses as threshold descends
    fpr = fp / n_neg
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2), float(s[i])


def frame_scores_to_clip(frame_probs: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Clip-level score from per-frame probabilities: max over valid frames.

    Max, not mean: a forged segment covers a median 8.5% of a fake video, so
    averaging buries the signal under genuine frames.
    """
    p = np.asarray(frame_probs, dtype=np.float64).reshape(-1)
    if valid is not None:
        v = np.asarray(valid).reshape(-1).astype(bool)
        p = p[v] if v.any() else p
    return float(p.max()) if p.size else 0.0
