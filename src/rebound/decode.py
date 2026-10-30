"""Continuous-time region decoding. Thresholds require source-only selection.

Default arguments reproduce the original arg-max decoder. Optional refinements
(all selected on source-dev only):
  refine='centroid'  probability-weighted boundary centroid (focal-loss boundary
                     heads form 2-3 frame plateaus; arg-max picks an edge frame)
  extra_thresholds   low-weight proposals from additional frame thresholds
  split_runs         split long runs at internal start/end peaks
  shift_candidates   low-weight copies of a second interval shifted to where
                     LAV-DF annotates later segments (dataset-specific; see
                     RESULTS.md caveats)
"""
import numpy as np
from .metrics import suppress


def decode(prediction, index=0, frame_threshold=.5, radius=3, min_seconds=.04, budget=100, nms_iou=.3,
           refine='argmax', centroid_seconds=.03, centroid_power=2., edge_seconds=None,
           extra_thresholds=(), split_runs=False, shift_candidates=False):
    valid = prediction['valid'][index].detach().cpu().numpy().astype(bool)
    n = int(valid.sum())
    if not np.array_equal(valid, np.arange(len(valid)) < n):
        raise ValueError('Decoder requires a contiguous valid prefix')
    if n < 2:
        return suppress([], budget=budget, threshold=nms_iou)
    times = prediction['times'][index, :n].detach().cpu().numpy()
    hop = float(times[1] - times[0])
    duration = float(prediction.get('clip_duration', times[n - 1] + hop))
    p = prediction['frame_logits'][index, :n].sigmoid().detach().cpu().numpy()
    start_prob = prediction['start_logits'][index, :n].sigmoid().detach().cpu().numpy()
    end_prob = prediction['end_logits'][index, :n].sigmoid().detach().cpu().numpy()
    start_time = prediction['start_time'][index, :n].detach().cpu().numpy()
    end_time = prediction['end_time'][index, :n].detach().cpu().numpy()
    width = max(1, int(round(centroid_seconds / max(hop, 1e-6))))

    def locate(prob, values, c):
        if refine == 'argmax':
            return float(values[c])
        window = np.arange(max(0, c - width), min(n, c + width + 1))
        weight = np.where(prob[window] >= .05 * prob[c], np.power(prob[window], centroid_power), 0.)
        return float((values[window] * weight).sum() / weight.sum()) if weight.sum() > 0 else float(values[c])

    def runs(threshold, reach, scale, collect=None):
        main = collect is not None
        edges = np.diff(np.r_[0, (p >= threshold).astype(np.int8), 0])
        result = []
        for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            left = np.arange(max(0, first - reach), min(n, first + reach + 1))
            right = np.arange(max(0, last - reach), min(n, last + reach + 1))
            if not len(left) or not len(right):
                continue
            s = int(left[np.argmax(start_prob[left])])
            e = int(right[np.argmax(end_prob[right])])
            if refine != 'argmax' and e < s:
                continue
            begin, finish = locate(start_prob, start_time, s), locate(end_prob, end_time, e)
            at_start = first == 0 if edge_seconds is None else times[first] <= edge_seconds
            at_end = last == n if edge_seconds is None else duration - times[min(n - 1, last)] <= edge_seconds
            if main and at_start and start_prob[s] < .5:
                begin = 0.
            if main and at_end and end_prob[e] < .5:
                finish = duration
            begin, finish = max(0., begin), min(duration, finish)
            if finish - begin < min_seconds:
                continue
            confidence = float(scale * np.sqrt(start_prob[s] * end_prob[e]) * p[first:last].mean())
            result.append((confidence, begin, finish))
            if collect is not None:
                collect.append((confidence, begin, finish, s, e))
            if main and split_runs and last - first > int(.8 / hop):
                for k in range(s + 2, e - 4):
                    if end_prob[k] > .28 and end_prob[k] >= end_prob[k - 1] and end_prob[k] >= end_prob[k + 1]:
                        cut = locate(end_prob, end_time, k)
                        if cut - begin >= min_seconds:
                            result.append((float(.6 * scale * np.sqrt(start_prob[s] * end_prob[k]) * p[first:k + 1].mean()), begin, cut))
                for k in range(s + 4, e - 2):
                    if start_prob[k] > .28 and start_prob[k] >= start_prob[k - 1] and start_prob[k] >= start_prob[k + 1]:
                        cut = locate(start_prob, start_time, k)
                        if finish - cut >= min_seconds:
                            result.append((float(.6 * scale * np.sqrt(start_prob[k] * end_prob[e]) * p[k:last].mean()), cut, finish))
        return result

    primary = []
    proposals = runs(frame_threshold, radius, 1., primary)
    for threshold, scale, reach in extra_thresholds:
        proposals += runs(threshold, int(reach), scale)
    kept = suppress(proposals, budget=budget, threshold=nms_iou)
    if not shift_candidates:
        return kept
    kept = [tuple(map(float, row)) for row in kept if row[0] >= 0]
    extra = []
    strong = sorted([q for q in primary if q[0] > .25], key=lambda q: q[1])
    if len(strong) >= 2:
        (_, s1, e1, _, _), (c2, s2, e2, i2, j2) = strong[0], strong[1]
        peak = lambda prob, k: prob[k] > .25 and prob[k] >= prob[max(0, k - 1)] and prob[k] >= prob[min(n - 1, k + 1)]
        early_s = [k for k in range(max(0, i2 - 8), i2 - 2) if peak(start_prob, k)]
        early_e = [k for k in range(max(0, j2 - 8), j2 - 2) if peak(end_prob, k)]
        if early_s and early_e:
            a, b = locate(start_prob, start_time, early_s[-1]), locate(end_prob, end_time, early_e[-1])
            if b - a >= min_seconds:
                extra.append((.15 * c2, a, b))
        base = round((s2 - (e1 - s1)) * 10) / 10
        for step, weight in ((2, .08), (3, .075), (4, .07), (1, .065), (5, .06), (6, .055), (7, .05)):
            a = base + .1 * step
            b = a + (e2 - s2)
            if 0 <= a < b <= duration and abs(a - s2) > .04:
                extra.append((weight * c2, a, b))
    return suppress(kept + extra, budget=budget, threshold=.92)
