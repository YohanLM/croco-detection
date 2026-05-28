"""Binary 75 %-coverage indicator loss + its set-level Risk wrapper.

A coarser cousin of `image_pixel_loss`: instead of summing fractional
coverage per GT, each GT contributes either 0 (≥ 75 % of its pixels
covered by the predicted set) or 1 (otherwise). The image-level loss
is the mean over GTs:

    L_i^{indicator} = (1 / n_i) · Σ_k 1{coverage(GT_k) < 0.75}

Useful when partial coverage isn't acceptable — a GT is either "found"
or "missed" at the 75 % threshold. Monotone non-increasing in λ: a
larger expansion can only push borderline GTs over the threshold,
never below it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import LossFunction, Risk
from conformal.loss.pixel import _gt_pixel_area, _union_pixel_area_inside


COVERAGE_FRACTION = 0.75
"""GT is considered "found" if at least this fraction of its pixels is covered."""


def image_coverage_indicator_loss(
    prediction_set: torch.Tensor, ground_truth: torch.Tensor
) -> float:
    """Implements `LossFunction` — binary 75 %-coverage indicator.

    Args:
        prediction_set: predicted boxes with expansion already applied,
            pixel xyxy. Shape `[P, 4]` or `[P, ≥5]` — only columns 0–3
            are read; extras (e.g. confidence at col 4) are ignored.
        ground_truth: GT boxes [G, 4], pixel xyxy.

    Returns:
        `0.0` if `G == 0`. Otherwise the fraction of GTs whose coverage
        is below `COVERAGE_FRACTION` (0.75).
    """
    if ground_truth.numel() == 0:
        return 0.0

    n_uncovered = 0
    n_valid = 0
    for k in range(ground_truth.shape[0]):
        gt = ground_truth[k]
        gt_area = _gt_pixel_area(gt)
        if gt_area == 0:
            continue
        n_valid += 1
        covered = _union_pixel_area_inside(gt, prediction_set)
        if covered / gt_area < COVERAGE_FRACTION:
            n_uncovered += 1

    if n_valid == 0:
        return 0.0
    return n_uncovered / n_valid


# ── Set-level risk ───────────────────────────────────────────────────────────

coverage_risk = Risk(image_coverage_indicator_loss)
"""Set-level coverage-indicator risk = mean of the per-image indicator loss."""


# ── Protocol-conformance assertions ──────────────────────────────────────────

if TYPE_CHECKING:
    _check_loss: LossFunction = image_coverage_indicator_loss
    _check_risk: Risk         = coverage_risk
