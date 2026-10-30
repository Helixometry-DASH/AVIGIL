"""Fractional timestamp targets; cropped-away edges are not new boundaries."""
import torch
from torch.nn import functional as F


def targets(times, valid, periods, sigma_frames=1.5, crop_ranges=None, offset_limit=1.):
    if times.ndim == 1:
        times = times[None].expand(len(valid), -1)
    result = {k: torch.zeros_like(times) for k in ('frame', 'start', 'end', 'start_offset', 'end_offset', 'start_offset_valid', 'end_offset_valid')}
    result['valid'] = valid
    result['start_valid'] = valid.clone()
    result['end_valid'] = valid.clone()
    for b, intervals in enumerate(periods):
        n = int(valid[b].sum())
        if n < 2:
            raise ValueError('Too few valid timestamp positions')
        timeline = times[b, :n]
        hop = (timeline[1:] - timeline[:-1]).median()
        sigma = sigma_frames * hop
        if crop_ranges is None:
            crop_start, crop_end = (timeline[0] - hop / 2).clamp_min(0), timeline[-1] + hop / 2
        else:
            crop_start, crop_end = crop_ranges[b]
        for start, end in intervals:
            if end <= start:
                raise ValueError('Reversed or empty interval')
            result['frame'][b, :n] = torch.maximum(result['frame'][b, :n], ((timeline >= start) & (timeline < end)).float())
            for edge, value, inside in [('start', start, start > float(crop_start) + 1e-7), ('end', end, end < float(crop_end) - 1e-7)]:
                inside = inside and float(crop_start) < value < float(crop_end)
                if inside:
                    gaussian = (-.5 * ((timeline - value) / sigma).square()).exp()
                    result[edge][b, :n] = torch.maximum(result[edge][b, :n], gaussian)
                    index = int((timeline - value).abs().argmin())
                    normalized = (value - timeline[index]) / hop
                    if abs(float(normalized)) > offset_limit + 1e-4:
                        raise ValueError('Boundary lies outside continuous offset support; check timestamp grid')
                    result[edge + '_offset'][b, index] = normalized
                    result[edge + '_offset_valid'][b, index] = 1.
                elif edge == 'start' and start <= float(crop_start):
                    result['start_valid'][b, :min(n, 3)] = False
                elif edge == 'end' and end >= float(crop_end):
                    result['end_valid'][b, max(0, n - 3):n] = False
    return result


def localization_loss(prediction, target):
    valid = target['valid']
    frame_loss = F.binary_cross_entropy_with_logits(prediction['frame_logits'][valid], target['frame'][valid], pos_weight=prediction['frame_logits'].new_tensor(2.))
    p = prediction['frame_logits'].sigmoid() * valid
    y = target['frame'] * valid
    dice = (1 - (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean()
    boundary, offset = [], []
    for edge in ('start', 'end'):
        logits, labels = prediction[edge + '_logits'], target[edge]
        probability = logits.sigmoid()
        pt = probability * labels + (1 - probability) * (1 - labels)
        alpha = .75 * labels + .25 * (1 - labels)
        mask = target[edge + '_valid'] & valid
        focal = alpha * (1 - pt).square() * F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        boundary.append((focal * mask).sum() / (labels * mask).sum().clamp_min(1.))
        mask = target[edge + '_offset_valid'].bool() & valid
        if mask.any():
            offset.append(F.smooth_l1_loss(prediction[edge + '_offset'][mask], target[edge + '_offset'][mask]))
        else:
            offset.append(prediction[edge + '_offset'].sum() * 0)
    total = frame_loss + dice + sum(boundary) / 2 + sum(offset)
    return total, {'frame_bce': float(frame_loss.detach()), 'dice': float(dice.detach()),
                   'boundary': float((sum(boundary) / 2).detach()), 'offset': float(sum(offset).detach())}
