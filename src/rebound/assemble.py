"""Full-clip prediction via the same overlapping windows used for training."""
import math
import torch


def window_starts(length, width, hop):
    if length <= width:
        return [0]
    starts = list(range(0, length - width + 1, hop))
    # Reference windows remain aligned to video tubelets (two 25Hz frames).
    last = math.ceil((length - width) / 2) * 2
    if last != starts[-1]:
        starts.append(last)
    return starts


def edge_weights(length, device):
    positions = torch.arange(length, device=device).float()
    return (1 - (positions - (length - 1) / 2).abs() / ((length + 1) / 2)).clamp_min(.05)


def blend_predictions(windows, timeline, valid):
    """Timestamp interpolation, not length-based interpolation.

    Each window contains local prediction times and an absolute crop offset.
    Padded positions are excluded. Offset TIMES, not normalized offsets, blend.
    """
    keys = ('frame_logits', 'start_logits', 'end_logits', 'start_time', 'end_time')
    result = {key: torch.zeros_like(timeline) for key in keys}
    denominator = torch.zeros_like(timeline)
    for first, prediction in windows:
        n = int(prediction['valid'][0].sum())
        if n < 2:
            continue
        times = prediction['times'][0, :n] + first
        step = times[1] - times[0]
        selected = (timeline >= times[0] - step / 2) & (timeline <= times[-1] + step) & valid
        query = timeline[selected]
        high = torch.searchsorted(times, query).clamp(1, n - 1)
        low = high - 1
        ratio = ((query - times[low]) / (times[high] - times[low])).clamp(0, 1)
        weights = edge_weights(n, timeline.device)
        w = weights[low] * (1 - ratio) + weights[high] * ratio
        denominator[selected] += w
        for key in keys:
            values = prediction[key][0, :n]
            if key.endswith('_time'):
                values = values + first
            result[key][selected] += w * (values[low] * (1 - ratio) + values[high] * ratio)
    mask = (denominator > 0) & valid
    for key in keys:
        result[key] = (result[key] / denominator.clamp_min(1e-8))[None]
    result.update(times=timeline[None], valid=mask[None])
    return result
