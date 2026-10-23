"""Sliding-window inference: crop for computation, predict on the full timeline.

Both encoders are fixed-window models -- the audio encoder takes 64,600-sample
(4.0375 s) crops and VideoMAE V2 is built around 16-frame clips -- while
recordings are longer. So the encoder runs on overlapping fixed-size windows,
each window's per-frame logits are placed back on the recording's own frame
grid by its start time, and the overlaps are blended. Nothing is truncated.

**Window frames land exactly on the global grid.** A window starting at sample
``s`` has its frame ``j`` at global frame ``s / hop + j``. Choosing the hop as a
whole number of frames (``hop_frames * 243`` samples) makes that an integer, so
no interpolation is needed and no resampling error is introduced.

Overlaps are blended with a triangular weight that is lowest at the window
edges, since a frame near an edge saw only part of its convolutional context --
the same weighting the feature cache uses.
"""

from __future__ import annotations

import numpy as np
import torch

from ..models.audio.grid import hop_samples, n_frames

HEADS = ("frame_logits", "start_logits", "end_logits")


def window_frame_starts(total_frames: int, t_win: int, hop_frames: int) -> list[int]:
    """Window starts in *frame* units, guaranteeing every frame is covered.

    Enumerated in frame space rather than sample space on purpose. Aligning the
    tail window to ``n_samples - window`` leaves a start that is not a multiple
    of the 243-sample hop, so the integer division to a global frame index
    truncates and the final frame is never covered -- an off-by-one that left a
    one-frame gap on an 8.5 s clip. Working in frames makes coverage exact by
    construction; the price is up to 242 samples of zero padding on the very
    last window, 0.4% of it.
    """
    if total_frames <= t_win:
        return [0]
    starts = list(range(0, total_frames - t_win + 1, hop_frames))
    tail = total_frames - t_win
    if starts[-1] < tail:
        starts.append(tail)
    return starts


def edge_weights(n: int, device=None) -> torch.Tensor:
    """Triangular weight over a window's frames, lowest at both edges."""
    j = torch.arange(n, dtype=torch.float32, device=device)
    c = (n - 1) / 2.0
    return 1.0 - 0.8 * (j - c).abs() / max(c, 1.0)


@torch.no_grad()
def predict_full(
    model,
    audio: np.ndarray,
    block: int = 4,
    window: int = 64600,
    hop_frames: int = 132,
    batch_windows: int = 16,
    device: str = "cuda",
    amp_dtype=torch.bfloat16,
) -> dict[str, np.ndarray]:
    """Per-frame probabilities for a whole recording, on its own frame grid.

    Returns ``frame``/``start``/``end`` arrays of length
    ``n_frames(len(audio), block)``.
    """
    n = len(audio)
    total = max(n_frames(n, block), 1)
    t_win = n_frames(window, block)
    starts = [p * hop_samples(block)
              for p in window_frame_starts(total, t_win, hop_frames)]

    acc = {k: torch.zeros(total, device=device) for k in HEADS}
    wsum = torch.zeros(total, device=device)
    wts = edge_weights(t_win, device)
    clip = []

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    for i in range(0, len(starts), batch_windows):
        chunk = starts[i : i + batch_windows]
        pieces = []
        for s in chunk:
            seg = wav[s : s + window]
            if seg.numel() < window:                     # tail window: pad the remainder
                seg = torch.nn.functional.pad(seg, (0, window - seg.numel()))
            pieces.append(seg)
        batch = torch.stack(pieces).to(device)
        with torch.autocast(device, dtype=amp_dtype):
            out = model(batch)
        clip.append(torch.sigmoid(out["clip_logit"].float()))
        probs = {k: torch.sigmoid(out[k].float()) for k in HEADS}
        for b, s in enumerate(chunk):
            # exact alignment: a whole number of frames, no interpolation
            g0 = s // hop_samples(block)
            g1 = min(g0 + t_win, total)
            m = g1 - g0
            if m <= 0:
                continue
            for k in HEADS:
                acc[k][g0:g1] += probs[k][b, :m] * wts[:m]
            wsum[g0:g1] += wts[:m]

    denom = wsum.clamp_min(1e-6)
    res = {k: (acc[k] / denom).cpu().numpy() for k in HEADS}
    res["coverage"] = (wsum > 0).float().cpu().numpy()
    res["clip"] = float(torch.cat(clip).max())           # any window suspicious -> clip fake
    res["n_windows"] = len(starts)
    return res
