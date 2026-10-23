"""Variable-length batching, for feeding whole clips with nothing cropped.

The architecture takes any input length, but a batch must be rectangular.
:class:`LengthBucketSampler` groups clips of similar duration so padding stays
a few percent of the batch, and evaluation runs at ``batch_size=1``, where
there is no padding at all.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


def collate_varlen(batch: list[dict]) -> dict:
    """Pad to the longest clip in the batch; report true lengths."""
    lengths = torch.tensor([b["audio"].numel() for b in batch], dtype=torch.long)
    width = int(lengths.max())
    audio = torch.zeros(len(batch), width, dtype=torch.float32)
    for i, b in enumerate(batch):
        audio[i, : b["audio"].numel()] = b["audio"]
    out = {"audio": audio, "length": lengths}
    for key in ("label", "modality", "index"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    out["pad_fraction"] = 1.0 - float(lengths.sum()) / (len(batch) * width)
    return out


class LengthBucketSampler(Sampler[list[int]]):
    """Yields batches of similar-length clips.

    Sorts by duration, cuts into batches, then shuffles batch *order* so the
    model still sees lengths in a random sequence across a run while each
    individual batch stays nearly uniform.
    """

    def __init__(
        self,
        durations: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 1337,
        jitter: int = 8,
    ) -> None:
        self.durations = np.asarray(durations, dtype=np.float64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.jitter = int(jitter)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        order = np.argsort(self.durations, kind="stable")
        if self.shuffle and self.jitter > 1:
            # local shuffle within a window, so identical-length runs vary
            for start in range(0, len(order), self.jitter):
                chunk = order[start : start + self.jitter]
                rng.shuffle(chunk)
                order[start : start + self.jitter] = chunk
        batches = [
            order[i : i + self.batch_size].tolist()
            for i in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        n = len(self.durations)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
