"""Number of boxes in a prediction set - an `EfficiencyMetric`.

`box_count` implements `EfficiencyMetric`: it measures set size as the plain
count of boxes. This is the natural cost notion for the confidence-filter
expansion, where larger lambda *admits* lower-confidence boxes (the count
grows) rather than inflating box geometry. Pairs with that expansion the way
`total_box_area` pairs with the multiplicative one.

All box tensors are expected in pixel-space `xyxy` form, shape `[N, >=4]`.
Empty sets have count zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import EfficiencyMetric


def box_count(prediction_set: torch.Tensor) -> float:
    """Implements `EfficiencyMetric` - the number of boxes in the set."""
    if prediction_set.numel() == 0:
        return 0.0
    return float(prediction_set.shape[0])


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_efficiency: EfficiencyMetric = box_count
