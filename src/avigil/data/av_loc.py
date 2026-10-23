"""Audio-visual localization dataset.

Yields both modalities for the same wall-clock window plus three label views.
The audio and visual views are **independent**: a record may place them over
entirely different spans, and the joint view is their merged union. A clip
whose audio alone was manipulated carries an empty visual target, and a clip
with audio manipulated at 1-3 s and video at 5-7 s supervises each branch on
its own interval only. See :mod:`avigil.data.labels` for the record schema.

Both modalities are cropped to the same window, which preserves the temporal
correspondence the fusion head reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.audio.grid import n_frames, segments_to_targets
from ..models.video.vgrid import segments_to_targets as video_targets
from .index import load_metadata, select_split
from .labels import modality_label, segments_for
from .video_loc import VmaeCache

VIDEO_HOP = 0.08            # one cached video position, in seconds


def modality_segments(record: dict, which: str) -> np.ndarray:
    """Forged intervals for ``which`` in {"audio", "visual", "joint"}.

    The audio and visual views are independent: a record may place them over
    entirely different spans. See :mod:`avigil.data.labels`.
    """
    return segments_for(record, which)


class AVLocDataset(Dataset):
    """Returns:
        audio        float32 (S,)        waveform for the crop
        feats        float32 (P, V, 768) cached video features for the same span
        frame/start/end, *_valid  float32 (T,)  JOINT targets on the audio grid
        clip         float32 ()          1 if any forged region in the crop
        modality     int64   ()          0 real / 1 visual-only / 2 audio-only / 3 both
        segments     float32 (N,2)       joint segments, crop-relative seconds
        seg_audio, seg_visual  float32 (N,2)  per-modality, for branch scoring
        index        int64   ()
    """

    def __init__(
        self,
        root: str | Path,
        feature_root: str | Path,
        split: str,
        metadata: str = "metadata.min.json",
        sample_rate: int = 16000,
        crop_seconds: float = 4.0,
        crop: str = "random",
        block: int = 4,
        video_upsample: int = 2,
        boundary_sigma: float = 1.5,
        window_hop: float = 0.0,     # >0 enumerates every window of every clip
        seed: int = 1337,
    ) -> None:
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.crop_seconds = float(crop_seconds)
        self.crop = crop
        self.block = int(block)
        self.video_upsample = int(video_upsample)
        self.boundary_sigma = float(boundary_sigma)
        self.seed = int(seed)
        self.epoch = 0

        self.cache = VmaeCache(feature_root, split)
        records = select_split(load_metadata(self.root, metadata), split)
        self.records = [r for r in records if r["file"] in self.cache.offsets]
        missing = len(records) - len(self.records)
        if missing:
            raise RuntimeError(
                f"{missing} of {len(records)} {split} clips are absent from the video "
                f"feature cache at {feature_root}; extraction is incomplete"
            )
        self.windows = self._build_windows(float(window_hop)) if window_hop > 0 else None

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the random crop for a new epoch.

        Without this the crop offset is a pure function of ``(seed, index)``,
        so every clip would be cropped at the same place in every epoch and
        the rest of it never seen. Re-rolling per epoch exposes the whole
        recording over the course of training.
        """
        self.epoch = int(epoch)

    def _build_windows(self, hop_seconds: float) -> list[tuple[int, float]]:
        """Every (clip, window-start) pair, so one epoch covers all the data.

        Sampling a single random window per clip per epoch leaves data unused:
        with a 4 s window on clips averaging 8.4 s, an epoch touched 47.5% of
        the audio, and even across 12 epochs 1.9% of forged segments were never
        seen at all. A forgery the model never looks at cannot teach it
        anything.

        Enumerating all windows at a 2 s hop costs 3.70x a single crop per
        epoch and reaches **100% of forged segments in one epoch**. Twelve
        epochs of one crop and four epochs of all windows are the same compute
        (12.0 vs 12.8 clip-passes), but the second never skips anything.
        """
        out: list[tuple[int, float]] = []
        for i, r in enumerate(self.records):
            clip = self.cache.positions(r["file"]) * VIDEO_HOP
            if clip <= self.crop_seconds:
                out.append((i, 0.0))
                continue
            span = clip - self.crop_seconds
            n = int(np.ceil(span / hop_seconds))
            starts = [k * hop_seconds for k in range(n + 1)]
            # snap to the video grid so both modalities stay on one timeline
            starts = sorted({round(t / VIDEO_HOP) * VIDEO_HOP for t in starts
                             if t <= span + 1e-9})
            if starts[-1] < span - 1e-9:
                starts.append(round(span / VIDEO_HOP) * VIDEO_HOP)
            out += [(i, t) for t in starts]
        return out

    def __len__(self) -> int:
        return len(self.windows) if self.windows is not None else len(self.records)

    def durations(self) -> np.ndarray:
        return np.array([r["video_frames"] / 25.0 for r in self.records], dtype=np.float64)

    def _start_seconds(self, i: int, clip_seconds: float) -> float:
        span = clip_seconds - self.crop_seconds
        if self.crop == "full" or span <= 0:
            return 0.0
        if self.crop == "head":
            return 0.0
        if self.crop == "center":
            return span / 2.0
        if self.crop == "random":
            rng = np.random.default_rng((self.seed, self.epoch, i))
            # snap to a video position so the two modalities stay on one grid
            k = int(rng.integers(0, int(span / VIDEO_HOP) + 1))
            return k * VIDEO_HOP
        raise ValueError(f"crop must be random|head|center|full, got {self.crop!r}")

    def __getitem__(self, i: int) -> dict:
        from .decode import decode_audio

        if self.windows is not None:
            i, t0 = self.windows[i]
            rec = self.records[i]
            clip_seconds = self.cache.positions(rec["file"]) * VIDEO_HOP
        else:
            rec = self.records[i]
            clip_seconds = self.cache.positions(rec["file"]) * VIDEO_HOP
            t0 = self._start_seconds(i, clip_seconds)
        span = clip_seconds if self.crop == "full" else min(self.crop_seconds, clip_seconds)

        wav = decode_audio(str(self.root / rec["file"]), self.sample_rate)
        s0 = int(round(t0 * self.sample_rate))
        n_samp = int(round(span * self.sample_rate))
        wav = wav[s0 : s0 + n_samp]
        if len(wav) < n_samp:                       # tile a clip shorter than one window
            reps = int(n_samp / max(len(wav), 1)) + 1
            wav = np.tile(wav, reps)[:n_samp]

        p0 = int(round(t0 / VIDEO_HOP))
        n_pos = int(round(span / VIDEO_HOP))
        feats = self.cache.read(rec["file"], p0, p0 + n_pos)
        if feats.shape[0] < n_pos and feats.shape[0]:
            pad = np.repeat(feats[-1:], n_pos - feats.shape[0], axis=0)
            feats = np.concatenate([feats, pad], axis=0)

        def rebase(which):
            out = []
            for s, e in modality_segments(rec, which):
                s, e = max(0.0, float(s) - t0), min(span, float(e) - t0)
                if e - s > 1e-6:
                    out.append((s, e))
            return np.asarray(out, dtype=np.float32).reshape(-1, 2)

        joint, aud, vis = rebase("joint"), rebase("audio"), rebase("visual")

        # Three dense target sets, each on the grid its head predicts on and
        # each from its own label view. The joint view alone would teach the
        # audio branch to find video-only manipulations, which carry no
        # acoustic evidence, and the video branch the reverse. Only the fusion
        # head sees the union of the two.
        tgt = segments_to_targets(
            [tuple(x) for x in joint], n_samp, block=self.block,
            sample_rate=self.sample_rate, boundary_sigma=self.boundary_sigma,
        )
        tgt_a = segments_to_targets(
            [tuple(x) for x in aud], n_samp, block=self.block,
            sample_rate=self.sample_rate, boundary_sigma=self.boundary_sigma,
        )
        tgt_v = video_targets(
            [tuple(x) for x in vis], feats.shape[0], upsample=self.video_upsample,
            boundary_sigma=self.boundary_sigma,
        )
        per_mod = {}
        for suf, t in (("a", tgt_a), ("v", tgt_v)):
            for k in ("frame", "start", "end", "start_valid", "end_valid"):
                per_mod[f"{k}_{suf}"] = torch.from_numpy(t[k])
        per_mod["clip_a"] = torch.tensor(1.0 if len(aud) else 0.0)
        per_mod["clip_v"] = torch.tensor(1.0 if len(vis) else 0.0)

        return {
            **per_mod,
            "audio": torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
            "feats": torch.from_numpy(feats),
            "frame": torch.from_numpy(tgt["frame"]),
            "start": torch.from_numpy(tgt["start"]),
            "end": torch.from_numpy(tgt["end"]),
            "start_valid": torch.from_numpy(tgt["start_valid"]),
            "end_valid": torch.from_numpy(tgt["end_valid"]),
            "clip": torch.tensor(1.0 if len(joint) else 0.0),
            "modality": torch.tensor(modality_label(rec), dtype=torch.long),
            "segments": torch.from_numpy(joint),
            "seg_audio": torch.from_numpy(rebase("audio")),
            "seg_visual": torch.from_numpy(rebase("visual")),
            "index": torch.tensor(i, dtype=torch.long),
        }

    def target_frames(self, n_samples: int | None = None) -> int:
        n = n_samples or int(round(self.crop_seconds * self.sample_rate))
        return n_frames(n, self.block)

    def class_balance(self) -> dict:
        from collections import Counter

        c = Counter(modality_label(r) for r in self.records)
        return {"real": c[0], "visual_only": c[1], "audio_only": c[2], "both": c[3]}


def collate_av_loc(batch: list[dict]) -> dict:
    """Pad audio, video positions and dense targets; report both masks."""
    ns = max(b["audio"].numel() for b in batch)
    np_ = max(b["feats"].shape[0] for b in batch)
    nt = max(b["frame"].numel() for b in batch)
    nb = len(batch)
    nvec, dim = batch[0]["feats"].shape[1], batch[0]["feats"].shape[2]

    out = {
        "audio": torch.zeros(nb, ns, dtype=torch.float32),
        "feats": torch.zeros(nb, np_, nvec, dim, dtype=torch.float32),
        "video_mask": torch.zeros(nb, np_, dtype=torch.bool),
        "tmask": torch.zeros(nb, nt, dtype=torch.bool),
    }
    DENSE = ("frame", "start", "end", "start_valid", "end_valid")
    nv = max(b["frame_v"].numel() for b in batch)
    for k in DENSE:
        out[k] = torch.zeros(nb, nt, dtype=torch.float32)
        out[f"{k}_a"] = torch.zeros(nb, nt, dtype=torch.float32)
        out[f"{k}_v"] = torch.zeros(nb, nv, dtype=torch.float32)
    out["vmask_out"] = torch.zeros(nb, nv, dtype=torch.bool)
    for i, b in enumerate(batch):
        s, p, t = b["audio"].numel(), b["feats"].shape[0], b["frame"].numel()
        tv = b["frame_v"].numel()
        out["audio"][i, :s] = b["audio"]
        out["feats"][i, :p] = b["feats"]
        out["video_mask"][i, :p] = True
        out["tmask"][i, :t] = True
        out["vmask_out"][i, :tv] = True
        for k in DENSE:
            out[k][i, :t] = b[k]
            out[f"{k}_a"][i, :t] = b[f"{k}_a"]
            out[f"{k}_v"][i, :tv] = b[f"{k}_v"]
    for k in ("clip", "clip_a", "clip_v"):
        out[k] = torch.stack([b[k] for b in batch])
    out["modality"] = torch.stack([b["modality"] for b in batch])
    out["index"] = torch.stack([b["index"] for b in batch])
    for k in ("segments", "seg_audio", "seg_visual"):
        out[k] = [b[k] for b in batch]
    out["audio_length"] = torch.tensor([b["audio"].numel() for b in batch], dtype=torch.long)
    out["video_length"] = torch.tensor([b["feats"].shape[0] for b in batch], dtype=torch.long)
    return out
