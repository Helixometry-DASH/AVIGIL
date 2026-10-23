"""Model construction, output shapes, and length invariance.

These checks need no dataset and no pretrained weights.
"""

from __future__ import annotations

import pytest
import torch

from avigil.models import available, build

N_SAMPLES = 64600          # 4.0375 s at 16 kHz
N_POSITIONS = 50           # 50 tubelets = 4.0 s at 25 fps
EMBED = 768


def make(variant: str = "m5"):
    return build(
        f"avigil_{variant}",
        audio_branch=build("aloc_a3", variant="a3"),
        video_branch=build("vloc_v3", variant="v3"),
        variant=variant, dim=256, audio_dim=128, video_dim=256,
        n_blocks=4, heads=4, fuse_grid="audio", freeze_branches=False,
    ).eval()


def test_every_variant_is_registered() -> None:
    names = set(available())
    for v in ("m3", "m4", "m5"):
        assert f"avigil_{v}" in names
    for v in ("a0", "a1", "a2", "a3"):
        assert f"aloc_{v}" in names
    for v in ("v0", "v1", "v2", "v3"):
        assert f"vloc_{v}" in names


@pytest.mark.parametrize("variant", ["m3", "m4", "m5"])
def test_forward_shapes(variant: str) -> None:
    model = make(variant)
    audio = torch.randn(2, N_SAMPLES)
    feats = torch.randn(2, N_POSITIONS, 17, EMBED)
    mask = torch.ones(2, N_POSITIONS, dtype=torch.bool)
    with torch.no_grad():
        out = model(audio, feats, mask)

    # the joint head predicts on the audio grid
    assert out["frame_logits"].shape == (2, 265)
    assert out["start_logits"].shape == (2, 265)
    assert out["end_logits"].shape == (2, 265)
    assert out["clip_logit"].shape == (2,)
    assert out["modality_logits"].shape == (2, 4)

    # each branch keeps its own grid: audio 15.19 ms, video 40 ms
    assert out["audio"]["frame_logits"].shape == (2, 265)
    assert out["video"]["frame_logits"].shape == (2, 100)
    assert out["audio"]["features"].shape == (2, 128, 265)
    assert out["video"]["features"].shape == (2, 256, 100)


def test_outputs_are_finite() -> None:
    model = make()
    with torch.no_grad():
        out = model(torch.randn(1, N_SAMPLES),
                    torch.randn(1, N_POSITIONS, 17, EMBED),
                    torch.ones(1, N_POSITIONS, dtype=torch.bool))
    for k, v in out.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), k


def test_video_branch_is_padding_invariant() -> None:
    """A valid position's output must not depend on how much padding its batch
    carries. The branch re-masks after every biased op to guarantee this."""
    branch = build("vloc_v3", variant="v3").eval()
    torch.manual_seed(0)
    feats = torch.randn(1, 30, 17, EMBED)

    short_mask = torch.ones(1, 30, dtype=torch.bool)
    padded = torch.cat([feats, torch.randn(1, 20, 17, EMBED)], dim=1)
    padded_mask = torch.cat(
        [torch.ones(1, 30, dtype=torch.bool), torch.zeros(1, 20, dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        a = branch(feats, short_mask)["frame_logits"][0, :60]
        b = branch(padded, padded_mask)["frame_logits"][0, :60]
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_modality_logits_cover_four_classes() -> None:
    model = make()
    with torch.no_grad():
        out = model(torch.randn(1, N_SAMPLES),
                    torch.randn(1, N_POSITIONS, 17, EMBED),
                    torch.ones(1, N_POSITIONS, dtype=torch.bool))
    probs = out["modality_logits"].softmax(-1)
    assert probs.shape == (1, 4)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
