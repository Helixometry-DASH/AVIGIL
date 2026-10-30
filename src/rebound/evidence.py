"""Evidence bank and directed multi-scale kernel changes on native timestamps."""
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceBank(nn.Module):
    """32 features from frozen genuine self/cross predictors.

    A projection can discard information; the private forensic experts are
    intentionally retained. This module does not claim to be sufficient.
    """
    def __init__(self, residual_dimension=512, seed=41):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(residual_dimension, 12, generator=generator)
        projection, _ = torch.linalg.qr(projection, mode='reduced')
        self.register_buffer('projection', projection)

    def forward(self, residuals, surprises, valid):
        # residuals [B,2 modalities,2 predictors,T,D]; surprises [B,2,2,T].
        a, b = residuals[:, :, 0].float(), residuals[:, :, 1].float()
        pa = F.pad(a[:, :, :-1], (0, 0, 1, 0))
        pb = F.pad(b[:, :, :-1], (0, 0, 1, 0))
        extras = torch.stack((
            a.square().mean(-1).log1p(), b.square().mean(-1).log1p(),
            F.cosine_similarity(a, b, dim=-1),
            (a - pa).square().mean(-1).log1p(), (b - pb).square().mean(-1).log1p(),
            F.cosine_similarity(a, pa, dim=-1), surprises[:, :, 0], surprises[:, :, 1]), -1)
        output = torch.cat((a @ self.projection, b @ self.projection, extras), -1)
        return output.masked_fill(~valid[:, None, :, None], 0)


def _kernel(x, y, bandwidth):
    distance = (x[..., :, None, :] - y[..., None, :, :]).square().sum(-1)
    return (-distance / (2 * bandwidth)).exp()


def _mean_kernel(kernel, left_mask, right_mask):
    weights = left_mask[..., :, None] & right_mask[..., None, :]
    return (kernel * weights).sum((-1, -2)) / weights.sum((-1, -2)).clamp_min(1)


def gaussian_reference_mmd(samples, valid, bandwidth):
    """Biased empirical MMD² to N(0,I), analytic reference expectations.

    Nonnegative up to numerical rounding; not a calibrated p-value. Repeated
    temporal anchors and serial dependence require empirical real calibration.
    """
    dimension = samples.shape[-1]
    empirical = _mean_kernel(_kernel(samples, samples, bandwidth), valid, valid)
    expectation = (bandwidth / (bandwidth + 1.)) ** (dimension / 2.)
    expectation = expectation * (-samples.square().sum(-1) / (2 * (bandwidth + 1.))).exp()
    cross = (expectation * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    reference = (bandwidth / (bandwidth + 2.)) ** (dimension / 2.)
    return (empirical - 2 * cross + reference).clamp_min(0)


class DirectedChangeProfile(nn.Module):
    def __init__(self, windows=(.08, .16, .32), anchors=8):
        super().__init__()
        self.windows, self.anchors = tuple(windows), anchors

    def _sample(self, z, times, valid, window, side):
        b, t, d = z.shape
        fraction = (torch.arange(self.anchors, device=z.device).float() + .5) / self.anchors
        query = times[..., None] + window * (fraction[None, None] - (1 if side == 'left' else 0))
        index = torch.searchsorted(times.contiguous(), query.reshape(b, -1).contiguous()).reshape(b, t, self.anchors)
        bounded = index.clamp(0, t - 1)
        batch = torch.arange(b, device=z.device)[:, None, None]
        values = z[batch, bounded]
        length = valid.sum(-1)
        last = times[torch.arange(b, device=z.device), (length - 1).clamp_min(0)]
        mask = (query >= times[:, :1, None]) & (query <= last[:, None, None]) & valid[batch, bounded]
        return values, mask

    def forward(self, z, times, valid):
        if times.ndim == 1:
            times = times[None].expand(len(z), -1)
        if times.shape != valid.shape or (times[:, 1:] <= times[:, :-1]).any():
            raise ValueError('Evidence timestamps must be strictly increasing')
        output = []
        for window in self.windows:
            left, lm = self._sample(z, times, valid, window, 'left')
            right, rm = self._sample(z, times, valid, window, 'right')
            statistics = []
            for factor in (.5, 1., 2.):
                bandwidth = z.shape[-1] * factor
                a = gaussian_reference_mmd(left, lm, bandwidth)
                b = gaussian_reference_mmd(right, rm, bandwidth)
                change = (_mean_kernel(_kernel(left, left, bandwidth), lm, lm)
                          + _mean_kernel(_kernel(right, right, bandwidth), rm, rm)
                          - 2 * _mean_kernel(_kernel(left, right, bandwidth), lm, rm)).clamp_min(0)
                statistics.append(torch.stack((a, b, change, b - a), -1))
            stat = torch.stack(statistics).mean(0)
            support = (lm.sum(-1) >= self.anchors // 2) & (rm.sum(-1) >= self.anchors // 2) & valid
            output.append(stat.masked_fill(~support[..., None], 0))
            output.append(support[..., None].float())
        return torch.cat(output, -1)
