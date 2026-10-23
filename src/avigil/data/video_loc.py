"""Visual localization dataset over the cached VideoMAE V2 features.

Visual segments exist only where ``modify_video`` is true. ``fake_periods`` is
shared between the modalities, so it is not by itself a visual annotation: a
clip whose audio alone was manipulated carries intervals while its video is
genuine, and its visual target is empty.

Features come from the offline cache (``scripts/extract_features.py``): one
fp16 vector set per 80 ms tubelet, channel 0 the global mean over all 196
patch tokens and channels 1: a 4x4 region grid. Running the backbone once
offline and reading the cache during training costs ~40 s per epoch instead of
~15 min per epoch per GPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.video.vgrid import n_out, segments_to_targets
from .index import load_metadata, select_split
from .labels import modality_label, segments_for


def video_segments(record: dict) -> np.ndarray:
    """Forged *visual* intervals in seconds, or empty if the video is genuine."""
    return segments_for(record, "visual")


class VmaeCache:
    """Read-only view of one split's feature memmap.

    Opened lazily per process: a ``np.memmap`` created in the parent survives
    fork but its page cache state does not, and DataLoader workers each want
    their own handle.
    """

    def __init__(self, root: str | Path, split: str,
                 expect_regions: int | None = None) -> None:
        self.root, self.split = Path(root), split
        with open(self.root / f"{split}_index.json") as f:
            self.index = json.load(f)
        self.offsets = self.index["offsets"]
        self.nvec = int(self.index["vectors_per_position"])
        self.regions = int(self.index.get("regions", 0)) or int(round((self.nvec - 1) ** 0.5))
        if expect_regions is not None and self.regions != int(expect_regions):
            raise ValueError(
                f"feature cache at {self.root}/{split} was built with "
                f"--regions {self.regions} ({self.nvec} vectors per position), but "
                f"task.video_regions is {expect_regions}. The spatial pooling attends "
                f"over these tokens, so the two must agree. Either re-run "
                f"scripts/extract_features.py with --regions {expect_regions}, or set "
                f"task.video_regions={self.regions} in the config."
            )
        self.dim = int(self.index["embed_dim"])
        self.shape = (int(self.index["total_positions"]), self.nvec, self.dim)
        self._mm = None
        self._pid = None

    @property
    def mm(self) -> np.memmap:
        if self._mm is None or self._pid != os.getpid():
            self._mm = np.memmap(
                self.root / f"{self.split}.f16", dtype=np.float16, mode="r", shape=self.shape
            )
            self._pid = os.getpid()
        return self._mm

    def positions(self, file: str) -> int:
        return int(self.offsets[file][1])

    def read(self, file: str, lo: int, hi: int) -> np.ndarray:
        off, length = self.offsets[file]
        lo = max(0, min(lo, length))
        hi = max(lo, min(hi, length))
        return np.asarray(self.mm[off + lo : off + hi], dtype=np.float32)


class VideoLocDataset(Dataset):
    """Cached-feature crops with dense frame / start / end targets.

    Returns:
        feats        float32 (P, V, 768)  P feature positions, 80 ms apart
        frame        float32 (T,)         T = P * upsample, 1 inside a forged region
        start, end   float32 (T,)         Gaussian-softened boundary targets
        start_valid  float32 (T,)         0 where a crop edge truncated a segment
        end_valid    float32 (T,)
        clip         float32 ()           1 if any forged video in this crop
        segments     float32 (N,2)        crop-relative seconds, for AP/AR
        modality     int64   ()
        index        int64   ()
    """

    def __init__(
        self,
        root: str | Path,
        feature_root: str | Path,
        split: str,
        metadata: str = "metadata.min.json",
        crop_positions: int = 96,
        upsample: int = 2,
        crop: str = "random",
        boundary_sigma: float = 1.5,
        seed: int = 1337,
    ) -> None:
        self.root = Path(root)
        self.crop_positions = int(crop_positions)
        self.upsample = int(upsample)
        self.crop = crop
        self.boundary_sigma = float(boundary_sigma)
        self.seed = int(seed)
        self.epoch = 0

        self.cache = VmaeCache(feature_root, split)
        records = select_split(load_metadata(self.root, metadata), split)
        self.records = [r for r in records if r["file"] in self.cache.offsets]
        if not self.records:
            raise RuntimeError(f"no cached records for split={split!r} under {feature_root}")
        missing = len(records) - len(self.records)
        if missing:
            raise RuntimeError(
                f"{missing} of {len(records)} {split} clips are absent from the feature "
                f"cache at {feature_root}; extraction is incomplete"
            )

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the random crop for a new epoch.

        Without this the crop offset is a pure function of ``(seed, index)``,
        so every clip would be cropped at the same place in every epoch and
        the rest of it never seen. Re-rolling per epoch exposes the whole
        recording over the course of training.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def durations(self) -> np.ndarray:
        return np.array([self.cache.positions(r["file"]) for r in self.records], dtype=np.float64)

    def _window(self, n_pos: int, i: int) -> int:
        """Start position of this item's crop."""
        if self.crop == "full" or n_pos <= self.crop_positions:
            return 0
        span = n_pos - self.crop_positions
        if self.crop == "head":
            return 0
        if self.crop == "center":
            return span // 2
        if self.crop == "random":
            rng = np.random.default_rng((self.seed, self.epoch, i))
            return int(rng.integers(0, span + 1))
        raise ValueError(f"crop must be random|head|center|full, got {self.crop!r}")

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        n_pos = self.cache.positions(rec["file"])
        want = n_pos if self.crop == "full" else min(self.crop_positions, n_pos)
        lo = self._window(n_pos, i)
        feats = self.cache.read(rec["file"], lo, lo + want)

        hop = 2.0 / 25.0                      # one feature position, in seconds
        t0 = lo * hop
        span = feats.shape[0] * hop

        segs = []
        for s, e in video_segments(rec):
            s, e = float(s) - t0, float(e) - t0
            s, e = max(0.0, s), min(span, e)
            if e - s > 1e-6:
                segs.append((s, e))

        tgt = segments_to_targets(
            segs, feats.shape[0], upsample=self.upsample,
            boundary_sigma=self.boundary_sigma, clip_start=0.0,
        )
        return {
            "feats": torch.from_numpy(feats),
            "frame": torch.from_numpy(tgt["frame"]),
            "start": torch.from_numpy(tgt["start"]),
            "end": torch.from_numpy(tgt["end"]),
            "start_valid": torch.from_numpy(tgt["start_valid"]),
            "end_valid": torch.from_numpy(tgt["end_valid"]),
            "clip": torch.tensor(1.0 if segs else 0.0),
            "segments": torch.from_numpy(np.asarray(segs, dtype=np.float32).reshape(-1, 2)),
            "modality": torch.tensor(modality_label(rec), dtype=torch.long),
            "index": torch.tensor(i, dtype=torch.long),
        }

    def segment_stats(self) -> dict:
        n = sum(1 for r in self.records if r["modify_video"])
        segs = [e - s for r in self.records for s, e in video_segments(r)]
        return {
            "visual_fake_clips": n,
            "genuine_visual_clips": len(self.records) - n,
            "segments": len(segs),
            "mean_seconds": float(np.mean(segs)) if segs else 0.0,
        }


def collate_video_loc(batch: list[dict]) -> dict:
    """Pad to the longest clip in the batch and report a validity mask.

    Padding is explicit rather than cosmetic: every conv block and the clip
    readout are masked with it, so a short clip batched with a long one is not
    scored on positions that do not exist.
    """
    pos = [b["feats"].shape[0] for b in batch]
    pmax = max(pos)
    up = batch[0]["frame"].shape[0] // pos[0]
    tmax = n_out(pmax, up)
    nb, nvec, dim = len(batch), batch[0]["feats"].shape[1], batch[0]["feats"].shape[2]

    feats = torch.zeros(nb, pmax, nvec, dim, dtype=torch.float32)
    mask = torch.zeros(nb, pmax, dtype=torch.bool)
    dense = {k: torch.zeros(nb, tmax, dtype=torch.float32)
             for k in ("frame", "start", "end", "start_valid", "end_valid")}
    tmask = torch.zeros(nb, tmax, dtype=torch.bool)

    for i, b in enumerate(batch):
        p, t = b["feats"].shape[0], b["frame"].shape[0]
        feats[i, :p] = b["feats"]
        mask[i, :p] = True
        tmask[i, :t] = True
        for k in dense:
            dense[k][i, :t] = b[k]

    out = {"feats": feats, "mask": mask, "tmask": tmask, **dense}
    out["clip"] = torch.stack([b["clip"] for b in batch])
    out["modality"] = torch.stack([b["modality"] for b in batch])
    out["index"] = torch.stack([b["index"] for b in batch])
    out["segments"] = [b["segments"] for b in batch]
    out["length"] = torch.tensor(pos, dtype=torch.long)
    out["pad_fraction"] = 1.0 - float(sum(pos)) / (nb * pmax)
    return out
