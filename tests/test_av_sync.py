"""Guard: decoded audio must start at t=0, aligned with video frame 0.

AAC encoder priming makes the first audio *packet* carry a negative pts. The
decoder is expected to trim it; if it stopped, every audio feature would shift
earlier than its label with no error raised anywhere -- a silent
desynchronisation in a multimodal model.

Set AVIGIL_DATASET_ROOT to a dataset root to run these; they skip otherwise.
"""

from __future__ import annotations

import glob
import os

import av
import pytest

ROOTS = [r for r in os.environ.get("AVIGIL_DATASET_ROOT", "").split(os.pathsep) if r]
FILES: list[str] = []
for root in ROOTS:
    FILES += sorted(glob.glob(f"{root}/*/*.mp4"))[:5]
    if FILES:
        break

pytestmark = pytest.mark.skipif(
    not FILES, reason="set AVIGIL_DATASET_ROOT to a dataset root to run these"
)


@pytest.mark.parametrize("path", FILES or [""])
def test_decoded_audio_starts_at_zero(path: str) -> None:
    with av.open(path) as c:
        first = next(c.decode(audio=0))
    assert first.pts == 0, (
        f"{path}: first decoded audio frame is at pts={first.pts}, not 0 -- "
        "encoder priming is no longer being trimmed and audio is "
        "desynchronised from video by that many samples"
    )


@pytest.mark.parametrize("path", FILES or [""])
def test_video_starts_at_zero(path: str) -> None:
    with av.open(path) as c:
        first = next(c.decode(video=0))
    assert first.pts == 0, f"{path}: first video frame is at pts={first.pts}, not 0"
