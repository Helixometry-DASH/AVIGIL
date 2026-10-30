import torch
from torch.nn import functional as F
from .targets import targets, localization_loss


def expert_loss(prediction, batch):
    ranges = [(0., float(n) / 16000) for n in batch['audio_length']]
    terms, statistics = [], {}
    for name, output, intervals in [('joint', prediction, batch['periods']),
                                    ('audio', prediction['audio'], batch['audio_periods']),
                                    ('video', prediction['video'], batch['video_periods'])]:
        truth = targets(output['times'], output['valid'], intervals, crop_ranges=ranges, offset_limit=3.)
        value, parts = localization_loss(output, truth)
        label = torch.tensor([bool(x) for x in intervals], dtype=torch.float32, device=value.device)
        clip = F.binary_cross_entropy_with_logits(output['clip_logit'], label)
        terms.append(value + .5 * clip)
        statistics[name] = float(value.detach())
    attribution = F.cross_entropy(prediction['modality_logits'], batch['modality'])
    total = sum(terms) + .5 * attribution
    statistics.update(loss=float(total.detach()), attribution=float(attribution.detach()))
    return total, statistics
