"""Label views: independent audio and visual intervals, and the joint union.

The audio and the video can be manipulated over completely different spans, so
each branch must be supervised on its own intervals only, and the joint view
must be the union of the two.
"""

from __future__ import annotations

import numpy as np
import pytest

from avigil.data.labels import (
    MODALITY_AUDIO,
    MODALITY_BOTH,
    MODALITY_REAL,
    MODALITY_VISUAL,
    frame_mask,
    merge_segments,
    modality_flags,
    modality_label,
    segments_for,
)


def seglist(seg) -> list[list[float]]:
    """Segments as plain floats, rounded past float32 representation error."""
    return [[round(float(a), 6), round(float(b), 6)] for a, b in seg]


# audio manipulated at 1-3 s, video at 5-7 s -- nothing in common
DISJOINT = {
    "file": "a.mp4", "split": "train", "video_frames": 250,
    "audio_fake_periods": [[1.0, 3.0]],
    "visual_fake_periods": [[5.0, 7.0]],
}

# overlapping but not equal
OVERLAP = {
    "file": "b.mp4", "split": "train", "video_frames": 250,
    "audio_fake_periods": [[1.0, 4.0]],
    "visual_fake_periods": [[3.0, 6.0]],
}

SHARED = {
    "file": "c.mp4", "split": "train", "video_frames": 250,
    "fake_periods": [[2.5, 3.2]], "modify_audio": True, "modify_video": True,
}


def test_disjoint_intervals_stay_separate() -> None:
    assert seglist(segments_for(DISJOINT, "audio")) == [[1.0, 3.0]]
    assert seglist(segments_for(DISJOINT, "visual")) == [[5.0, 7.0]]


def test_joint_is_the_union_of_both() -> None:
    assert seglist(segments_for(DISJOINT, "joint")) == [[1.0, 3.0], [5.0, 7.0]]


def test_overlapping_intervals_merge_in_the_joint_view() -> None:
    assert seglist(segments_for(OVERLAP, "audio")) == [[1.0, 4.0]]
    assert seglist(segments_for(OVERLAP, "visual")) == [[3.0, 6.0]]
    # counted once, not twice
    assert seglist(segments_for(OVERLAP, "joint")) == [[1.0, 6.0]]


def test_audio_only_record_has_no_visual_target() -> None:
    rec = {**DISJOINT, "visual_fake_periods": []}
    assert seglist(segments_for(rec, "audio")) == [[1.0, 3.0]]
    assert len(segments_for(rec, "visual")) == 0
    assert modality_label(rec) == MODALITY_AUDIO


def test_visual_only_record_has_no_audio_target() -> None:
    rec = {**DISJOINT, "audio_fake_periods": []}
    assert len(segments_for(rec, "audio")) == 0
    assert seglist(segments_for(rec, "visual")) == [[5.0, 7.0]]
    assert modality_label(rec) == MODALITY_VISUAL


def test_modality_is_inferred_from_the_interval_lists() -> None:
    assert modality_flags(DISJOINT) == (True, True)
    assert modality_label(DISJOINT) == MODALITY_BOTH
    empty = {**DISJOINT, "audio_fake_periods": [], "visual_fake_periods": []}
    assert modality_label(empty) == MODALITY_REAL


def test_unsorted_intervals_are_sorted() -> None:
    rec = {**DISJOINT, "audio_fake_periods": [[8.0, 9.0], [1.0, 2.0]]}
    assert seglist(segments_for(rec, "audio")) == [[1.0, 2.0], [8.0, 9.0]]


def test_per_modality_frame_masks_differ() -> None:
    a = frame_mask(DISJOINT, fps=25.0, which="audio")
    v = frame_mask(DISJOINT, fps=25.0, which="visual")
    j = frame_mask(DISJOINT, fps=25.0, which="joint")
    assert a[25:75].all() and not a[125:175].any()      # 1-3 s only
    assert v[125:175].all() and not v[25:75].any()      # 5-7 s only
    assert j.sum() == a.sum() + v.sum()                 # disjoint, so additive


# ---- the shared-interval schema must behave exactly as before --------------

def test_shared_schema_is_unchanged() -> None:
    assert seglist(segments_for(SHARED, "audio")) == [[2.5, 3.2]]
    assert seglist(segments_for(SHARED, "visual")) == [[2.5, 3.2]]
    assert seglist(segments_for(SHARED, "joint")) == [[2.5, 3.2]]
    assert modality_label(SHARED) == MODALITY_BOTH


@pytest.mark.parametrize(
    ("ma", "mv", "expect"),
    [(True, True, MODALITY_BOTH), (True, False, MODALITY_AUDIO),
     (False, True, MODALITY_VISUAL), (False, False, MODALITY_REAL)],
)
def test_shared_schema_flags_gate_each_view(ma: bool, mv: bool, expect: int) -> None:
    rec = {**SHARED, "modify_audio": ma, "modify_video": mv}
    assert (len(segments_for(rec, "audio")) > 0) is ma
    assert (len(segments_for(rec, "visual")) > 0) is mv
    assert (len(segments_for(rec, "joint")) > 0) is (ma or mv)
    assert modality_label(rec) == expect


def test_merge_segments_handles_edges() -> None:
    assert len(merge_segments(np.zeros((0, 2), dtype=np.float32))) == 0
    assert seglist(merge_segments(np.array([[1.0, 2.0]], dtype=np.float32))) == [[1.0, 2.0]]
    # touching intervals become one
    assert seglist(merge_segments(
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    )) == [[1.0, 3.0]]
    # fully contained
    assert seglist(merge_segments(
        np.array([[1.0, 9.0], [3.0, 4.0]], dtype=np.float32)
    )) == [[1.0, 9.0]]
