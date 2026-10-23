"""VideoMAE V2 token extraction.

The upstream checkpoint's ``forward_features`` ends in ``fc_norm(x.mean(1))``,
a single pooled 768-vector per 16-frame window. That throws away exactly the
two axes a localization model needs: *where* in the frame and *when* in the
window. :func:`window_tokens` reruns the same stack and returns the tokens
before pooling.

``fc_norm`` is applied per token rather than after pooling. LayerNorm is not
linear, so ``mean(fc_norm(x)) != fc_norm(mean(x))``; applying it per token is
what the pooled vector would have been normalized from, and keeps every token
on the scale the pretrained head expects.

Token layout is set by ``PatchEmbed``: ``Conv3d`` output ``(B, C, T', H', W')``
is flattened from dim 2, so the sequence runs tubelet-major, then row, then
column -- ``view(B, T', 14, 14, C)`` is the correct unflattening and anything
else silently transposes space into time.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# Hugging Face id, or a local snapshot directory via AVIGIL_BACKBONE.
HF_REPO = os.environ.get("AVIGIL_BACKBONE") or "OpenGVLab/VideoMAEv2-Base"
TUBELET = 2            # frames per temporal token
WINDOW = 16            # frames the checkpoint's pos_embed was built for
GRID = 14              # 224 / 16 patches per side
EMBED = 768

# from the checkpoint's preprocessor_config.json
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def load_backbone(repo: str = HF_REPO, device: str = "cuda"):
    """The pretrained ViT-B/16 trunk, in eval mode with grads off."""
    from transformers import AutoModel

    vit = AutoModel.from_pretrained(repo, trust_remote_code=True).model
    vit = vit.to(device).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


def normalize(frames: torch.Tensor) -> torch.Tensor:
    """uint8 ``(T, H, W, 3)`` -> float ``(3, T, H, W)`` in the checkpoint's space.

    Frames are expected at 224x224 already, so there is no resize and no crop
    here: a resize-then-centre-crop would rescale every frame and discard a
    border, which can cut the subject off a source that is already framed on
    it. ``scripts/predict.py`` resizes arbitrary input before this point.
    """
    x = frames.permute(3, 0, 1, 2).float().div_(255.0)
    mean = torch.tensor(PIXEL_MEAN, device=x.device).view(3, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=x.device).view(3, 1, 1, 1)
    return x.sub_(mean).div_(std)


@torch.no_grad()
def window_tokens(vit, x: torch.Tensor) -> torch.Tensor:
    """``(B, 3, 16, 224, 224)`` -> ``(B, 8, 196, 768)`` patch tokens.

    Mirrors ``VisionTransformer.forward_features`` up to the final pooling.
    """
    b = x.size(0)
    t = vit.patch_embed(x)
    if vit.pos_embed is not None:
        t = t + vit.pos_embed.expand(b, -1, -1).type_as(t).to(t.device).clone().detach()
    t = vit.pos_drop(t)
    for blk in vit.blocks:
        t = blk(t)
    t = vit.fc_norm(t) if vit.fc_norm is not None else vit.norm(t)
    return t.view(b, WINDOW // TUBELET, GRID * GRID, EMBED)


def pool_regions(tokens: torch.Tensor, regions: int = 4) -> torch.Tensor:
    """``(B, T', 196, C)`` -> ``(B, T', regions**2 + 1, C)``.

    Channel ``0`` is the mean over all 196 patches -- taken before the region
    pooling so it is the true spatial mean, not a mean of means. Channels
    ``1:`` are a ``regions x regions`` average-pooled grid, row-major.

    Caching all 196 patch tokens would be two orders of magnitude larger on
    disk; a 4x4 grid plus the global vector keeps the cache tractable and
    still resolves a mouth-region manipulation to one sixteenth of the frame.
    """
    b, t, n, c = tokens.shape
    glob = tokens.mean(dim=2, keepdim=True)
    grid = tokens.view(b * t, GRID, GRID, c).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, regions)
    pooled = pooled.flatten(2).transpose(1, 2).reshape(b, t, regions * regions, c)
    return torch.cat([glob, pooled], dim=2)


def window_starts(n_frames: int, stride: int) -> list[int]:
    """Window start frames covering ``n_frames``, the last one edge-padded."""
    if n_frames <= WINDOW:
        return [0]
    n = (n_frames - WINDOW + stride - 1) // stride + 1
    return [i * stride for i in range(n)]


def center_weights(device=None) -> torch.Tensor:
    """Triangular weight over a window's 8 tubelets, peaked at the centre.

    With a 50%-overlapping stride every temporal position is seen by two
    windows. A tubelet near a window edge saw only half its temporal context,
    so it is down-weighted 4.5:1 against one at the centre instead of being
    averaged in as an equal.
    """
    j = torch.arange(WINDOW // TUBELET, dtype=torch.float32, device=device)
    return 4.5 - (j - (WINDOW // TUBELET - 1) / 2).abs()
