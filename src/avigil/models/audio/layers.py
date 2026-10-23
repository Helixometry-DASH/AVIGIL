"""Front-end and graph layers for the audio branch.

Everything the audio encoder is built from lives here, so the branch has no
external model dependency:

    SincFilterBank                a fixed mel-spaced band-pass filterbank
    ResidualBlock                 2-D residual block with time-axis pooling
    GraphAttention                dense graph attention over a node set
    HeterogeneousGraphAttention   two node sets plus a master node
    GraphPool                     top-k node selection by learned score

The filterbank has no learnable parameters: its kernels are derived from mel
spacing at construction and held as a non-persistent buffer, so it moves with
the module across devices without entering the checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _xavier_param(*size: int) -> nn.Parameter:
    p = nn.Parameter(torch.empty(*size))
    nn.init.xavier_normal_(p)
    return p


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


class SincFilterBank(nn.Module):
    """Fixed band-pass filterbank, mel-spaced across the Nyquist range.

    ``out_channels`` band-pass kernels are built as the difference of two sinc
    low-pass responses and windowed with a Hamming taper. The kernel length is
    forced odd so each filter is symmetric and its group delay is exactly
    ``(kernel_size - 1) // 2`` samples.
    """

    def __init__(self, out_channels: int, kernel_size: int,
                 sample_rate: int = 16000, in_channels: int = 1,
                 stride: int = 1, padding: int = 0, dilation: int = 1) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"only one input channel is supported, got {in_channels}")
        if kernel_size % 2 == 0:
            kernel_size += 1                      # keep the filters symmetric

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride, self.padding, self.dilation = stride, padding, dilation

        nfft = 512
        freqs = (sample_rate // 2) * np.linspace(0, 1, nfft // 2 + 1)
        mel = _hz_to_mel(freqs)
        edges = _mel_to_hz(np.linspace(mel.min(), mel.max(), out_channels + 1))

        # The kernels are evaluated at single precision -- the precision the
        # convolution itself runs in -- so the filterbank is exactly the same
        # whatever dtype the surrounding arithmetic happens to use: the sinc
        # argument and the Hamming taper are both rounded to float32 before use.
        support = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1,
                            dtype=np.float32)
        window = np.hamming(kernel_size).astype(np.float32)
        sr32 = np.float32(sample_rate)
        kernels = np.zeros((out_channels, kernel_size), dtype=np.float32)
        for i in range(out_channels):
            lo, hi = float(edges[i]), float(edges[i + 1])
            high = np.float64(2 * hi / sample_rate) * np.sinc(
                np.float32(2 * hi) * support / sr32)
            low = np.float64(2 * lo / sample_rate) * np.sinc(
                np.float32(2 * lo) * support / sr32)
            kernels[i] = window * (high - low).astype(np.float32)

        # not persistent: derived from mel spacing, so it is rebuilt on load
        # rather than stored, and the checkpoint stays free of fixed constants
        self.register_buffer("band_pass", torch.from_numpy(kernels), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filters = self.band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters.to(x.dtype), stride=self.stride,
                        padding=self.padding, dilation=self.dilation)


class ResidualBlock(nn.Module):
    """2-D residual block that pools the time axis by 3.

    ``nb_filts`` is ``[in_channels, out_channels]``. A 1x3 convolution matches
    the channel count on the skip path when they differ. ``first=True`` drops
    the pre-activation norm, which the first block in a stack does not need
    because its input has already been normalised.
    """

    def __init__(self, nb_filts: list[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        c_in, c_out = nb_filts[0], nb_filts[1]

        if not first:
            self.bn1 = nn.BatchNorm2d(num_features=c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=(2, 3), padding=(0, 1), stride=1)

        self.downsample = c_in != c_out
        if self.downsample:
            self.conv_downsample = nn.Conv2d(c_in, c_out, kernel_size=(1, 3),
                                             padding=(0, 1), stride=1)
        self.mp = nn.MaxPool2d((1, 3))            # time / 3, frequency untouched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(self.selu(self.bn2(out)))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)


class GraphAttention(nn.Module):
    """Dense attention over a node set, with a residual-free dual projection.

    Edge scores come from the elementwise product of every node pair, so the
    attention is content-based and symmetric in its inputs. Each node is then
    projected twice -- once through the attention-weighted neighbourhood and
    once directly -- and the two are summed, which keeps a path that does not
    depend on the attention map.

    ``temperature`` divides the logits before the softmax; larger values make
    the attention flatter.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.temp = float(temperature)
        self.att_weight = _xavier_param(out_dim, 1)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.SELU(inplace=True)

    def _pairwise(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        a = x.unsqueeze(2).expand(-1, -1, n, -1)
        return a * a.transpose(1, 2)

    def _att_map(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.att_proj(self._pairwise(x)))
        a = torch.matmul(a, self.att_weight) / self.temp
        return F.softmax(a, dim=-2)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        return self.bn(x.reshape(-1, size[-1])).view(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, N, in_dim) -> (B, N, out_dim); N is preserved."""
        x = self.input_drop(x)
        att = self._att_map(x)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        return self.act(self._norm(x))


class HeterogeneousGraphAttention(nn.Module):
    """Attention over two node sets of different kinds, plus a master node.

    The two sets are projected separately, concatenated into one graph, and
    attended jointly with four edge weights -- within set 1, within set 2, and
    the two cross-set directions -- so an edge is scored according to which
    kinds it connects. The master node aggregates the whole graph through its
    own attention and is returned for the next stage to carry forward.

    Both sets come back with their lengths and ordering unchanged.
    """

    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.temp = float(temperature)
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _xavier_param(out_dim, 1)
        self.att_weight22 = _xavier_param(out_dim, 1)
        self.att_weight12 = _xavier_param(out_dim, 1)
        self.att_weightM = _xavier_param(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.SELU(inplace=True)

    def _pairwise(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        a = x.unsqueeze(2).expand(-1, -1, n, -1)
        return a * a.transpose(1, 2)

    def _att_map(self, x: torch.Tensor, n1: int, n2: int) -> torch.Tensor:
        a = torch.tanh(self.att_proj(self._pairwise(x)))
        board = torch.zeros_like(a[:, :, :, 0]).unsqueeze(-1)
        board[:, :n1, :n1, :] = torch.matmul(a[:, :n1, :n1, :], self.att_weight11)
        board[:, n1:, n1:, :] = torch.matmul(a[:, n1:, n1:, :], self.att_weight22)
        board[:, :n1, n1:, :] = torch.matmul(a[:, :n1, n1:, :], self.att_weight12)
        board[:, n1:, :n1, :] = torch.matmul(a[:, n1:, :n1, :], self.att_weight12)
        return F.softmax(board / self.temp, dim=-2)

    def _master_att(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.att_projM(x * master))
        a = torch.matmul(a, self.att_weightM) / self.temp
        return F.softmax(a, dim=-2)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        return self.bn(x.reshape(-1, size[-1])).view(size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                master: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x1`` (B, N1, C), ``x2`` (B, N2, C) -> both back, plus the master."""
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = x.mean(dim=1, keepdim=True)

        x = self.input_drop(x)
        att = self._att_map(x, n1, n2)

        m_att = self._master_att(x, master)
        master = (self.proj_with_attM(torch.matmul(m_att.squeeze(-1).unsqueeze(1), x))
                  + self.proj_without_attM(master))

        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.act(self._norm(x))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    """Keep the top ``k`` fraction of nodes by learned score.

    Scores also gate the surviving nodes, so a node that barely made the cut
    contributes proportionally less. Node order follows the scores, not the
    input, so this is only applied to axes with no ordering to preserve.
    """

    def __init__(self, k: float, in_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.sigmoid(self.proj(self.drop(h)))
        n_keep = max(int(h.size(1) * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        idx = idx.expand(-1, -1, h.size(-1))
        return torch.gather(h * scores, 1, idx)
