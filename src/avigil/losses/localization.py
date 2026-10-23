"""Losses for temporal localization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Soft Dice on frame probabilities.

    Paired with BCE because BCE alone is dominated by the genuine majority:
    forged frames are ~19% of a spoofed crop and ~6% of all audio. Dice is
    scale-free in the positive count, so short segments still matter.

    ``mask`` zeroes padded frames. Dice is a *ratio*, so padding is not merely
    diluting here: predicted probability mass on a padded tail inflates the
    denominator and makes the loss look worse than the model is.
    """
    prob = torch.sigmoid(logits)
    if mask is not None:
        prob, target = prob * mask, target * mask
    num = 2.0 * (prob * target).sum(dim=-1) + eps
    den = prob.sum(dim=-1) + target.sum(dim=-1) + eps
    return (1.0 - num / den).mean()


def masked_focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE over soft boundary targets, ignoring masked positions.

    ``valid`` is 0 where a crop edge truncated a segment: the region is still
    fake but the edge is not a real transition, so supervising a boundary there
    would teach the model that crop edges are manipulation boundaries.

    Normalised by **positive mass**, not frame count. A Gaussian boundary
    target with sigma=1.5 carries ~3.8 of mass spread over 265 frames, so
    dividing by the number of frames would scale the term down to a fraction
    of the region losses. Dividing by positive mass is the RetinaNet
    convention and keeps the boundary and region terms comparable.

    ``alpha`` defaults to 0.75 to up-weight the rare positive frames.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    loss = a_t * (1 - p_t).pow(gamma) * ce
    if valid is not None:
        loss = loss * valid
    norm = target.sum() if valid is None else (target * valid).sum()
    return loss.sum() / norm.clamp(min=1.0)


class LocalizationLoss(nn.Module):
    """L = w_frame * (BCE + Dice) + w_boundary * focal + w_clip * BCE."""

    def __init__(
        self,
        w_frame: float = 1.0,
        w_dice: float = 1.0,
        w_boundary: float = 1.0,
        w_clip: float = 0.5,
        pos_weight: float | None = None,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.w_frame, self.w_dice = w_frame, w_dice
        self.w_boundary, self.w_clip = w_boundary, w_clip
        self.pos_weight = pos_weight
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma

    def forward(self, pred: dict, target: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """``target["frame_mask"]``, if given, is 1 on real frames and 0 on padding.

        Batching full-length clips means every batch has a padded tail. An
        unmasked mean over it dilutes the loss by the pad fraction (0.147 on a
        random batch of 8) and, worse, teaches the model that padded silence is
        genuine -- a free negative it will never see at inference.
        """
        pw = (
            torch.tensor(self.pos_weight, device=pred["frame_logits"].device)
            if self.pos_weight
            else None
        )
        fm = target.get("frame_mask")
        if fm is None:
            l_frame = F.binary_cross_entropy_with_logits(
                pred["frame_logits"], target["frame"], pos_weight=pw
            )
        else:
            per = F.binary_cross_entropy_with_logits(
                pred["frame_logits"], target["frame"], pos_weight=pw, reduction="none"
            )
            l_frame = (per * fm).sum() / fm.sum().clamp_min(1.0)
        l_dice = dice_loss(pred["frame_logits"], target["frame"], mask=fm)
        l_start = masked_focal_bce(
            pred["start_logits"], target["start"], target.get("start_valid"),
            self.focal_alpha, self.focal_gamma,
        )
        l_end = masked_focal_bce(
            pred["end_logits"], target["end"], target.get("end_valid"),
            self.focal_alpha, self.focal_gamma,
        )
        l_clip = F.binary_cross_entropy_with_logits(pred["clip_logit"], target["clip"])

        total = (
            self.w_frame * l_frame
            + self.w_dice * l_dice
            + self.w_boundary * (l_start + l_end) * 0.5
            + self.w_clip * l_clip
        )
        return total, {
            "loss": float(total),
            "l_frame": float(l_frame),
            "l_dice": float(l_dice),
            "l_start": float(l_start),
            "l_end": float(l_end),
            "l_clip": float(l_clip),
        }
