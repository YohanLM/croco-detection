"""Total predicted box area — a pixel-space `EfficiencyMetric`.

`total_box_area` implements `EfficiencyMetric`: it measures the size of a
conformal prediction set as the summed pixel area of its boxes. Pairs with
the multiplicative expansion + pixel loss — larger λ inflates every box, so
area rises monotonically while risk falls. Reporting area at λ̂ relative to
the raw (λ=0) set gives the geometric *cost* of calibration
(`EvaluationResult.inflation_ratio`).

All box tensors are expected in pixel-space `xyxy` form, shape `[N, ≥4]`;
only columns 0–3 are read (any confidence column is ignored).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import EfficiencyMetric


def total_box_area(prediction_set: torch.Tensor) -> float:
    """Implements `EfficiencyMetric` — `Σ_j w_j · h_j` over the set's boxes.

    Sums the per-box pixel area without de-duplicating overlap: it is a
    cost proxy, not a footprint measurement, so two overlapping boxes count
    twice. Widths/heights are clamped at 0 to ignore any degenerate boxes.
    Empty sets have zero area.
    """
    if prediction_set.numel() == 0:
        return 0.0
    w = (prediction_set[:, 2] - prediction_set[:, 0]).clamp(min=0)
    h = (prediction_set[:, 3] - prediction_set[:, 1]).clamp(min=0)
    return float((w * h).sum().item())


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_efficiency: EfficiencyMetric = total_box_area
