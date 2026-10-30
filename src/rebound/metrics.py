"""Versioned temporal AP: official GT-first matching, corrected recall origin."""
import numpy as np


def iou(proposals, ground):
    p = np.asarray(proposals, dtype=np.float64).reshape(-1, 2)
    g = np.asarray(ground, dtype=np.float64).reshape(-1, 2)
    intersection = np.maximum(0, np.minimum(p[:, None, 1], g[None, :, 1]) - np.maximum(p[:, None, 0], g[None, :, 0]))
    union = (p[:, 1] - p[:, 0])[:, None] + (g[:, 1] - g[:, 0])[None] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def temporal_metrics(proposals, ground):
    if len(proposals) != len(ground):
        raise ValueError('Prediction/ground-truth clip counts differ')
    total = sum(len(g) for g in ground)
    if not total:
        raise ValueError('Temporal metrics require positive intervals')
    thresholds = (.5, .75, .95)
    confidence, flags = [], {t: [] for t in thresholds}
    recalls = {k: [] for k in (10, 20, 50, 100)}
    coverage = {t: 0 for t in thresholds}
    genuine_max = []
    for p, g in zip(proposals, ground):
        p = np.asarray(p).reshape(-1, 3)
        p = p[p[:, 0] >= 0]
        p = p[np.argsort(-p[:, 0], kind='stable')]
        confidence.extend(p[:, 0])
        overlap = iou(p[:, 1:], g)
        if not len(g):
            genuine_max.append(float(p[:, 0].max()) if len(p) else 0.)
        for threshold in thresholds:
            used = np.zeros(len(p), dtype=bool)
            for j in range(len(g)):
                candidates = np.flatnonzero((overlap[:, j] > threshold) & ~used)
                if len(candidates):
                    used[candidates[0]] = True
            flags[threshold].extend(used)
            coverage[threshold] += int(((overlap.max(0) if len(p) else np.zeros(len(g))) > threshold).sum())
        for k in recalls:
            recalls[k].extend((overlap[:k].max(0) if len(p) else np.zeros(len(g))).tolist())
    order = np.argsort(-np.asarray(confidence), kind='stable')
    ap = {}
    for threshold in thresholds:
        tp = np.asarray(flags[threshold], dtype=np.float64)[order]
        precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap[str(threshold)] = float(envelope[tp > 0].sum() / total * 100)
    return {'metric_version': 'corrected_initial_recall_gt_first_strict_iou_v1',
            'ap_pct': ap, 'ar_pct': {str(k): float(np.mean([np.mean(np.asarray(v) > t) for t in (.5, .55, .6, .65, .7, .75, .8, .85, .9, .95)]) * 100) for k, v in recalls.items()},
            'emitted_candidate_coverage_pct': {str(t): 100 * n / total for t, n in coverage.items()},
            'genuine_max_confidence_mean': float(np.mean(genuine_max)) if genuine_max else None,
            'videos': len(ground), 'ground_truth_intervals': total}


def suppress(proposals, budget=100, threshold=.8):
    """Fixed hard-NMS; method-specific postprocessing, not official AuViRe NMS."""
    proposals = np.asarray(proposals, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(-proposals[:, 0], kind='stable')
    kept = []
    while len(order) and len(kept) < budget:
        selected = int(order[0])
        kept.append(selected)
        remaining = order[1:]
        if not len(remaining):
            break
        overlap = iou(proposals[remaining, 1:], proposals[selected:selected + 1, 1:]).ravel()
        order = remaining[overlap <= threshold]
    output = np.zeros((budget, 3), dtype=np.float32)
    output[:, 0] = -1
    if kept:
        output[:len(kept)] = proposals[kept]
    return output


def prediction_first_metrics(proposals, ground):
    """AVIGIL-style AP matching, separately labeled; AR is intentionally omitted."""
    total = sum(len(g) for g in ground)
    if not total or len(proposals) != len(ground):
        raise ValueError('Invalid evaluation population')
    result = {}
    for threshold in (.5, .75, .95):
        confidence, flags = [], []
        for p, g in zip(proposals, ground):
            p = np.asarray(p).reshape(-1, 3)
            p = p[p[:, 0] >= 0]
            p = p[np.argsort(-p[:, 0], kind='stable')]
            overlap = iou(p[:, 1:], g)
            used = np.zeros(len(g), dtype=bool)
            for j in range(len(p)):
                eligible = np.flatnonzero(~used)
                match = int(eligible[np.argmax(overlap[j, eligible])]) if len(eligible) else -1
                hit = match >= 0 and overlap[j, match] >= threshold
                if hit:
                    used[match] = True
                confidence.append(p[j, 0])
                flags.append(hit)
        order = np.argsort(-np.asarray(confidence), kind='stable')
        tp = np.asarray(flags, dtype=float)[order]
        precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        result[str(threshold)] = float(envelope[tp > 0].sum() / total * 100)
    return {'metric_version': 'prediction_first_best_unmatched_inclusive_iou_v1', 'ap_pct': result}
