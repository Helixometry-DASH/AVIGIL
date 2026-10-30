"""Modality-free synthetic reference-space changes; never real forgery results."""
import torch


def sample_reference_changes(batch, frames, dimension=32, step=.0151875, device='cpu', seed=None):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    z = torch.randn(batch, frames, dimension, device=device, generator=generator)
    valid = torch.ones(batch, frames, dtype=torch.bool, device=device)
    times = torch.arange(frames, device=device).float() * step
    periods = []
    for b in range(batch):
        if b % 4 == 0:
            periods.append([])
            continue
        start = int(torch.randint(frames // 5, frames // 2, (), device=device, generator=generator))
        end = int(torch.randint(start + frames // 8, frames * 4 // 5, (), device=device, generator=generator))
        kind = (b + (seed or 0)) % 6
        shift_s = (float(torch.rand((), device=device, generator=generator)) - .5) * step
        shift_e = (float(torch.rand((), device=device, generator=generator)) - .5) * step
        start_time, end_time = float(times[start]) + shift_s, float(times[end]) + shift_e
        weight = (((times - start_time) / step + .5).clamp(0, 1)
                  * ((end_time - times) / step + .5).clamp(0, 1))[:, None]
        original = z[b].clone()
        if kind == 0:
            direction = torch.randn(dimension, device=device, generator=generator)
            direction = direction / direction.norm() * dimension ** .5
            alternative = original + direction * 2.
        elif kind == 1:
            alternative = original * 2.5
        elif kind == 2:
            alternative = .35 * original + 1.5
        elif kind == 3:
            ramp = ((times - start_time) / max((end_time - start_time) * .3, step)).clamp(0, 1)
            alternative = original + 3 * ramp[:, None]
        elif kind == 4:
            alternative = original.clone()
            alternative[:, 0] *= 1.9 ** .5
            alternative[:, 1] *= .1 ** .5
        else:
            rho = .85
            values = [original[0]]
            for i in range(1, frames):
                values.append(rho * values[-1] + (1 - rho ** 2) ** .5 * original[i])
            alternative = torch.stack(values)
        z[b] = original * (1 - weight) + alternative * weight
        periods.append([(start_time, end_time)])
    return z, times, valid, periods
