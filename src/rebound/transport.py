"""Conditional triangular normalizing flows fitted exclusively on genuine data.

These are practical affine autoregressive transports. Exact Gaussianization is
a property of the ideal Rosenblatt transform, not a guarantee of finite fitting.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, inputs, outputs, mask):
        super().__init__(inputs, outputs)
        self.register_buffer('mask', mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveAffine(nn.Module):
    def __init__(self, dimension, hidden=256, context=0):
        super().__init__()
        self.dimension = dimension
        self.context = context
        input_degree = torch.arange(1, dimension + 1)
        hidden_degree = torch.arange(hidden) % dimension
        self.first = MaskedLinear(dimension, hidden, hidden_degree[:, None] >= input_degree[None])
        self.middle = MaskedLinear(hidden, hidden, hidden_degree[:, None] >= hidden_degree[None])
        self.last = MaskedLinear(hidden, 2 * dimension, input_degree.repeat(2)[:, None] > hidden_degree[None])
        self.condition = nn.Linear(context, hidden, bias=False) if context else None
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def parameters_at(self, x, context=None):
        h = self.first(x)
        if self.condition is not None:
            if context is None:
                raise ValueError('Conditional reference transport needs context')
            h = h + self.condition(context)
        h = F.silu(h)
        h = F.silu(self.middle(h))
        mean, log_scale = self.last(h).chunk(2, -1)
        return mean, 2. * torch.tanh(log_scale / 2.)

    def forward(self, x, context=None):
        mean, log_scale = self.parameters_at(x, context)
        z = (x - mean) * (-log_scale).exp()
        return z, -log_scale.sum(-1)

    def inverse(self, z, context=None):
        values = []
        for i in range(self.dimension):
            x = torch.stack(values, -1) if values else z[..., :0]
            x = F.pad(x, (0, self.dimension - i))
            mean, log_scale = self.parameters_at(x, context)
            values.append(z[..., i] * log_scale[..., i].exp() + mean[..., i])
        return torch.stack(values, -1)


class RealReferenceTransport(nn.Module):
    def __init__(self, dimension=32, hidden=256, layers=4, context=0):
        super().__init__()
        self.dimension = dimension
        self.layers = nn.ModuleList([AutoregressiveAffine(dimension, hidden, context) for _ in range(layers)])
        permutation = torch.arange(dimension - 1, -1, -1)
        self.register_buffer('permutation', permutation)
        self.register_buffer('location', torch.zeros(dimension))
        self.register_buffer('scale', torch.ones(dimension))
        self.register_buffer('fitted_genuine_only', torch.tensor(False))

    @torch.no_grad()
    def fit_standardization(self, values, valid=None, *, groups):
        if set(groups) != {'Real'}:
            raise ValueError('Reference transport fitting requires genuine examples only')
        x = values[valid] if valid is not None else values.reshape(-1, self.dimension)
        if len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError('Invalid genuine reference sample')
        self.location.copy_(x.mean(0))
        self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-3))
        self.fitted_genuine_only.fill_(True)

    def forward(self, values, context=None):
        with torch.autocast(device_type=values.device.type, enabled=False):
            z = (values.float() - self.location) / self.scale
            logdet = -self.scale.log().sum().expand(values.shape[:-1])
            for layer in self.layers:
                z = z[..., self.permutation]
                z, determinant = layer(z, context)
                logdet = logdet + determinant
            return z, logdet

    def inverse(self, z, context=None):
        with torch.autocast(device_type=z.device.type, enabled=False):
            x = z.float()
            for layer in reversed(self.layers):
                x = layer.inverse(x, context)[..., self.permutation]
            return x * self.scale + self.location

    def genuine_nll(self, values, valid, *, groups, context=None):
        if set(groups) != {'Real'}:
            raise ValueError('Forged examples cannot update the real reference distribution')
        z, logdet = self(values, context)
        nll = (.5 * (z.square() + math.log(2 * math.pi)).sum(-1) - logdet) / self.dimension
        return nll[valid].mean()

    def freeze(self):
        self.eval().requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        return self
