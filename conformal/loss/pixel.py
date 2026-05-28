"""Per-image pixel-wise recall loss (methodology Eq. 2) and its set-level Risk.

`image_pixel_loss` implements `LossFunction` — the per-image failure metric.
`pixel_risk = Risk(image_pixel_loss)` is the canonical set-level wrapper
plugged into the Calibrator.

All box tensors are expected in pixel-space `xyxy` form, shape [N, 4].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import LossFunction, Risk


# ── Pixel-area helpers ───────────────────────────────────────────────────────

def _union_pixel_area_inside(
    gt_xyxy: torch.Tensor, preds_xyxy: torch.Tensor
) -> int:
    """Pixel count of `(∪_j preds_j) ∩ gt` via local-mask rasterization.

    Each predicted box is clipped to the integer-aligned gt bounding box
    and OR'd into a local mask. Avoids the O(2^N) inclusion-exclusion
    blowup for overlapping boxes while staying exact at pixel resolution.
    """
    gx1 = int(torch.floor(gt_xyxy[0]).item())
    gy1 = int(torch.floor(gt_xyxy[1]).item())
    gx2 = int(torch.ceil(gt_xyxy[2]).item())
    gy2 = int(torch.ceil(gt_xyxy[3]).item())
    w, h = gx2 - gx1, gy2 - gy1
    if w <= 0 or h <= 0 or preds_xyxy.numel() == 0:
        return 0

    mask = torch.zeros((h, w), dtype=torch.bool)
    for p in preds_xyxy:
        px1 = max(int(torch.floor(p[0]).item()) - gx1, 0)
        py1 = max(int(torch.floor(p[1]).item()) - gy1, 0)
        px2 = min(int(torch.ceil(p[2]).item()) - gx1, w)
        py2 = min(int(torch.ceil(p[3]).item()) - gy1, h)
        if px1 < px2 and py1 < py2:
            mask[py1:py2, px1:px2] = True
    return int(mask.sum().item())


def _gt_pixel_area(gt_xyxy: torch.Tensor) -> int:
    """Integer-rounded pixel area of one gt box."""
    w = int(torch.ceil(gt_xyxy[2]).item()) - int(torch.floor(gt_xyxy[0]).item())
    h = int(torch.ceil(gt_xyxy[3]).item()) - int(torch.floor(gt_xyxy[1]).item())
    return max(w, 0) * max(h, 0)


# ── Per-image loss ───────────────────────────────────────────────────────────

def image_pixel_loss(
    prediction_set: torch.Tensor, ground_truth: torch.Tensor
) -> float:
    """Implements `LossFunction` — `L_i^{OD-pixel}` (methodology Eq. 2).

    Args:
        prediction_set: predicted boxes with expansion already applied,
            pixel xyxy. Shape `[P, 4]` or `[P, ≥5]` — only columns 0–3
            are read; any extra (e.g. the confidence score at col 4)
            is ignored.
        ground_truth: GT boxes [G, 4], pixel xyxy.

    Returns:
        `0.0` if `G == 0`. Otherwise `1 - mean_k coverage(GT_k)`, where
        coverage is the fraction of GT-k's pixel area inside the union of
        the predicted boxes.
    """
    if ground_truth.numel() == 0:
        return 0.0

    total_recall = 0.0
    n_gt = ground_truth.shape[0]
    for k in range(n_gt):
        gt = ground_truth[k]
        gt_area = _gt_pixel_area(gt)
        if gt_area == 0:
            n_gt -= 1
            continue
        covered = _union_pixel_area_inside(gt, prediction_set)
        total_recall += covered / gt_area

    if n_gt == 0:
        return 0.0
    return 1.0 - total_recall / n_gt


# ── Set-level risk ───────────────────────────────────────────────────────────

pixel_risk = Risk(image_pixel_loss)
"""Set-level pixel risk = mean of `image_pixel_loss` over the calibration set."""


# ── Protocol-conformance assertions ──────────────────────────────────────────

if TYPE_CHECKING:
    _check_loss: LossFunction = image_pixel_loss
    _check_risk: Risk         = pixel_risk
