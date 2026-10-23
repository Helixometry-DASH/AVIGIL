"""Grid arithmetic: the mapping between time, samples and frame indices.

Targets and predictions share one grid, so an error here shifts every label
without raising anywhere. These checks need no dataset.
"""

from __future__ import annotations

import pytest

from avigil.models.audio.grid import (
    SINC_DELAY,
    SINC_KERNEL,
    frame_to_time,
    hop_samples,
    n_frames,
    time_to_frame,
)
from avigil.models.video import vgrid


def test_sinc_delay_is_half_the_kernel() -> None:
    assert SINC_DELAY == (SINC_KERNEL - 1) // 2


@pytest.mark.parametrize("block", [1, 2, 3, 4, 5, 6])
def test_hop_is_a_power_of_three(block: int) -> None:
    assert hop_samples(block) == 3 ** (block + 1)


def test_frame_counts_match_the_encoder() -> None:
    # a 4.0375 s window at 16 kHz
    assert n_frames(64600, 4) == 265
    assert n_frames(64600, 6) == 29


@pytest.mark.parametrize("block", [4, 6])
@pytest.mark.parametrize("frame", [0, 1, 7, 64, 264])
def test_audio_time_and_frame_are_inverses(block: int, frame: int) -> None:
    t = frame_to_time(frame, block)
    assert time_to_frame(t, block) == pytest.approx(frame, abs=1e-9)


def test_audio_frame_zero_is_the_group_delay() -> None:
    assert frame_to_time(0, 4) == pytest.approx(SINC_DELAY / 16000)


@pytest.mark.parametrize("upsample", [1, 2])
@pytest.mark.parametrize("frame", [0, 1, 50, 99])
def test_video_time_and_frame_are_inverses(upsample: int, frame: int) -> None:
    t = vgrid.frame_to_time(frame, upsample)
    assert vgrid.time_to_frame(t, upsample) == pytest.approx(frame, abs=1e-9)


def test_video_hop_is_forty_milliseconds_at_upsample_two() -> None:
    assert vgrid.hop_seconds(2) == pytest.approx(0.04)
    assert vgrid.n_out(50, 2) == 100
