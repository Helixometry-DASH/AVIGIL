"""Metadata loading and source-group bookkeeping.

See :mod:`avigil.data.labels` for the record schema.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_metadata(root: str | Path, filename: str = "metadata.min.json") -> list[dict]:
    path = Path(root) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"no metadata at {path}. Set data.root to the dataset directory "
            f"holding {filename} and the split folders; see the record schema "
            f"in avigil.data.labels."
        )
    with open(path) as f:
        return json.load(f)


def by_file(records: list[dict]) -> dict[str, dict]:
    return {r["file"]: r for r in records}


def source_key(record: dict) -> str:
    """The real video a record belongs to: itself if real, else its ``original``."""
    return record["original"] or record["file"]


def source_groups(records: list[dict]) -> dict[str, list[dict]]:
    """Map each real video to itself plus every fake derived from it.

    Grouping by source video is what keeps a resample leakage-free: a real
    video and every fake derived from it belong to the same group, so they
    cannot be split across train and test.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[source_key(r)].append(r)
    return dict(groups)


def select_split(
    records: list[dict], split: str, reals: str = "all"
) -> list[dict]:
    """Records for one split.

    ``reals="sources"`` keeps only real videos that at least one fake was
    derived from, giving a negative set paired with the positives. ``"all"``
    keeps every real video.
    """
    rows = [r for r in records if r["split"] == split]
    if reals == "all":
        return rows
    if reals != "sources":
        raise ValueError(f"reals must be 'all' or 'sources', got {reals!r}")
    used = {r["original"] for r in records if r["original"]}
    return [r for r in rows if r["original"] or r["file"] in used]


def class_counts(records: list[dict]) -> dict[str, int]:
    from .labels import MODALITY_NAMES, modality_label

    counts = dict.fromkeys(MODALITY_NAMES, 0)
    for r in records:
        counts[MODALITY_NAMES[modality_label(r)]] += 1
    return counts


def positive_frame_fraction(records: list[dict], fps: float = 25.0) -> float:
    """Share of frames that are forged -- the pos_weight for frame-level BCE.

    0.0633 over the full dataset, which is why the frame loss is weighted and
    the reported metrics are AP and AR rather than frame accuracy.
    """
    pos = total = 0
    for r in records:
        n = int(r["video_frames"])
        total += n
        for start, end in r.get("fake_periods") or []:
            lo, hi = round(start * fps), round(end * fps)
            pos += max(0, min(hi, n) - max(0, lo))
    return pos / max(total, 1)
