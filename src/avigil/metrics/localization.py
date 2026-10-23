"""Temporal localization metrics: AP@IoU and AR@N.

AP is reported at IoU 0.5 / 0.75 / 0.9 / 0.95 and AR at 10 / 5 / 2 / 1
proposals, both configurable under ``eval`` in the config. Second-to-frame
conversion goes through the exact grid arithmetic in ``models.audio.grid``,
which is what keeps the tight 0.95 operating point meaningful.
"""

from __future__ import annotations

import numpy as np


def temporal_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between segment sets ``a`` (N,2) and ``b`` (M,2) -> (N,M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, 0], b[None, :, 0])
    hi = np.minimum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(hi - lo, 0, None)
    union = (a[:, None, 1] - a[:, None, 0]) + (b[None, :, 1] - b[None, :, 0]) - inter
    return np.where(union > 0, inter / union, 0.0)


def average_precision(
    predictions: list[tuple[int, float, float, float]],
    ground_truth: dict[int, np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Interpolated AP over all videos at one IoU threshold.

    ``predictions``: ``(video_id, start, end, score)``, any order.
    ``ground_truth``: ``video_id -> (N, 2)`` segments.
    Each ground-truth segment may be matched once; later matches are false
    positives, which is what penalizes duplicate proposals.
    """
    n_gt = int(sum(len(v) for v in ground_truth.values()))
    if n_gt == 0:
        return float("nan")
    if not predictions:
        return 0.0

    preds = sorted(predictions, key=lambda p: -p[3])
    matched: dict[int, np.ndarray] = {
        vid: np.zeros(len(segs), dtype=bool) for vid, segs in ground_truth.items()
    }
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (vid, start, end, _score) in enumerate(preds):
        gt = ground_truth.get(vid)
        if gt is None or len(gt) == 0:
            fp[i] = 1
            continue
        ious = temporal_iou(np.array([[start, end]]), gt)[0]
        order = np.argsort(-ious)
        hit = False
        for j in order:
            if ious[j] < iou_threshold:
                break
            if not matched[vid][j]:
                matched[vid][j] = True
                tp[i] = 1
                hit = True
                break
        if not hit:
            fp[i] = 1

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)

    # monotone envelope, then integrate over recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def average_recall(
    predictions: list[tuple[int, float, float, float]],
    ground_truth: dict[int, np.ndarray],
    topk: int = 100,
    iou_thresholds: tuple[float, ...] = (0.5, 0.95),
) -> float:
    """Recall at ``topk`` proposals per video, averaged over IoU thresholds."""
    n_gt = int(sum(len(v) for v in ground_truth.values()))
    if n_gt == 0:
        return float("nan")

    per_video: dict[int, list] = {}
    for vid, start, end, score in predictions:
        per_video.setdefault(vid, []).append((start, end, score))

    recalls = []
    for thr in iou_thresholds:
        hits = 0
        for vid, gt in ground_truth.items():
            if len(gt) == 0:
                continue
            props = sorted(per_video.get(vid, []), key=lambda p: -p[2])[:topk]
            if not props:
                continue
            ious = temporal_iou(np.array([[s, e] for s, e, _ in props]), gt)
            hits += int((ious.max(axis=0) >= thr).sum())
        recalls.append(hits / n_gt)
    return float(np.mean(recalls))


def nms(
    segments: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy temporal NMS -- collapses the duplicate proposals that hurt AP."""
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(segments) == 0:
        return segments, scores
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = temporal_iou(segments[i : i + 1], segments[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    keep = np.array(keep, dtype=int)
    return segments[keep], scores[keep]


def evaluate_localization(
    predictions: list[tuple[int, float, float, float]],
    ground_truth: dict[int, np.ndarray],
    ap_iou: tuple[float, ...] = (0.5, 0.75, 0.95),
    ar_topk: tuple[int, ...] = (100, 50, 20, 10),
    ar_iou: tuple[float, ...] = (0.5, 0.95),
) -> dict[str, float]:
    out = {f"AP@{t}": average_precision(predictions, ground_truth, t) for t in ap_iou}
    out.update(
        {f"AR@{k}": average_recall(predictions, ground_truth, k, ar_iou) for k in ar_topk}
    )
    return out


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision form).

    The right frame-level summary here: forged frames are ~6% of all audio, so
    ROC-AUC flatters a model that mostly predicts "genuine" while AUPRC does
    not. Frame *accuracy* is meaningless at this prior (93.7% for all-genuine).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def frame_f1(
    probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Precision / recall / F1 / IoU over frames at a fixed threshold."""
    pred = np.asarray(probs).reshape(-1) >= threshold
    gt = np.asarray(targets).reshape(-1).astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "frame_precision": prec,
        "frame_recall": rec,
        "frame_f1": f1,
        "frame_iou": tp / max(tp + fp + fn, 1),
    }


def boundary_error(
    predictions: list[tuple[int, float, float, float]],
    ground_truth: dict[int, np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Mean absolute start/end error over best-matched proposals, in seconds.

    Only proposals that match a ground-truth segment above ``iou_threshold``
    count: boundary error on a proposal that found nothing is meaningless.
    """
    best: dict[tuple[int, int], tuple[float, float, float]] = {}
    for vid, s, e, score in predictions:
        gt = ground_truth.get(vid)
        if gt is None or len(gt) == 0:
            continue
        ious = temporal_iou(np.array([[s, e]]), gt)[0]
        j = int(np.argmax(ious))
        if ious[j] < iou_threshold:
            continue
        key = (vid, j)
        if key not in best or score > best[key][2]:
            best[key] = (s, e, score)

    if not best:
        return {"start_mae": float("nan"), "end_mae": float("nan"),
                "boundary_mae": float("nan"), "matched": 0}
    ds, de = [], []
    for (vid, j), (s, e, _) in best.items():
        gs, ge = ground_truth[vid][j]
        ds.append(abs(s - gs))
        de.append(abs(e - ge))
    return {
        "start_mae": float(np.mean(ds)),
        "end_mae": float(np.mean(de)),
        "boundary_mae": float(np.mean(ds + de)),
        "matched": len(best),
    }
