"""Phase-1 detection-miss loss for SeqCRC + its set-level Risk wrapper.

Where `coverage.py` asks *"is >= 75 % of the GT covered?"*, this asks the
coarser, upstream question that SeqCRC Phase 1 controls: *"did the detector
fire on the target at all — is there at least one surviving box near it?"*.
It is the "Target in Y_i is missed by predictions" indicator of the
methodology (Eq. 2).

Per image the loss is the fraction of valid GT boxes that NO surviving
prediction *hits*:

    L_i^{det} = (1 / n_i) * sum_k 1{GT_k is hit by no prediction}

For the typical single-target 60x5 frame this is just 0 (target found) or
1 (target missed). It is the natural Phase-1 loss because additive expansion
can only grow a box that already exists near the GT — a GT with no nearby
box is *unrecoverable* by Phase 2, so it must be charged to Phase 1.

Two hit criteria are provided:

  - **nonzero overlap** (default): a GT is hit if any prediction has a
    strictly positive pixel intersection with it. The most lenient notion —
    "a box touches the target, so expansion has something to grow".
  - **IoU >= threshold**: stricter — a GT is hit only if some prediction
    overlaps it well enough. Cheaper Phase 2 (boxes already sit tightly) at
    the cost of a harder Phase 1.

Monotone non-increasing in the Phase-1 confidence knob: lowering the
threshold only admits more boxes, which can turn a miss into a hit but never
the reverse — required for the CRC root search.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from conformal.calibrator import LossFunction, Risk


DEFAULT_IOU_MATCH = 0.10
"""IoU floor for the strict (IoU-based) hit criterion, matching the
detection-counter diagnostics' default."""


# ── Hit criteria: does any prediction "find" a single GT box? ─────────────────

HitCriterion = Callable[[torch.Tensor, torch.Tensor], bool]
"""`(gt_xyxy[4], preds_xyxy[P, >=4]) -> bool` — was this GT hit?"""


def _pairwise_intersection(gt: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Per-prediction intersection area with one GT box. Shape `[P]`."""
    x1 = torch.maximum(preds[:, 0], gt[0])
    y1 = torch.maximum(preds[:, 1], gt[1])
    x2 = torch.minimum(preds[:, 2], gt[2])
    y2 = torch.minimum(preds[:, 3], gt[3])
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def nonzero_overlap_hit(gt: torch.Tensor, preds: torch.Tensor) -> bool:
    """Hit iff some prediction has strictly positive overlap with the GT."""
    if preds.numel() == 0:
        return False
    return bool((_pairwise_intersection(gt, preds) > 0).any().item())


def make_iou_hit(iou_threshold: float = DEFAULT_IOU_MATCH) -> HitCriterion:
    """Build a hit criterion: GT is hit iff some prediction's IoU >= threshold."""

    def hit(gt: torch.Tensor, preds: torch.Tensor) -> bool:
        if preds.numel() == 0:
            return False
        inter = _pairwise_intersection(gt, preds)
        gt_area = (gt[2] - gt[0]).clamp(min=0) * (gt[3] - gt[1]).clamp(min=0)
        p_area = ((preds[:, 2] - preds[:, 0]).clamp(min=0)
                  * (preds[:, 3] - preds[:, 1]).clamp(min=0))
        union = p_area + gt_area - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        return bool((iou >= iou_threshold).any().item())

    return hit


# ── Per-image loss + set-level Risk, parameterized by the hit criterion ───────

def make_detection_miss_loss(hit_fn: HitCriterion) -> LossFunction:
    """Build a `LossFunction` — fraction of valid GTs hit by no prediction.

    Degenerate (zero-area) GT boxes are skipped; a frame with no valid GT
    contributes loss `0.0` (nothing to detect).
    """

    def loss(prediction_set: torch.Tensor, ground_truth: torch.Tensor) -> float:
        if ground_truth.numel() == 0:
            return 0.0
        n_valid = 0
        n_missed = 0
        for k in range(ground_truth.shape[0]):
            gt = ground_truth[k]
            if (gt[2] - gt[0]) <= 0 or (gt[3] - gt[1]) <= 0:
                continue
            n_valid += 1
            if not hit_fn(gt, prediction_set):
                n_missed += 1
        if n_valid == 0:
            return 0.0
        return n_missed / n_valid

    return loss


def make_detection_risk(hit_fn: HitCriterion) -> Risk:
    """Set-level detection-miss risk = mean per-image miss over the sample."""
    return Risk(make_detection_miss_loss(hit_fn))


# ── Ready-made instances ──────────────────────────────────────────────────────

#: Default Phase-1 loss / risk — nonzero-overlap hit criterion.
detection_miss_loss = make_detection_miss_loss(nonzero_overlap_hit)
detection_risk = make_detection_risk(nonzero_overlap_hit)


def make_iou_detection_risk(iou_threshold: float = DEFAULT_IOU_MATCH) -> Risk:
    """Convenience: Phase-1 risk under the stricter IoU-based hit criterion."""
    return make_detection_risk(make_iou_hit(iou_threshold))


# ── Protocol-conformance assertions ──────────────────────────────────────────

if TYPE_CHECKING:
    _check_loss: LossFunction = detection_miss_loss
    _check_risk: Risk         = detection_risk
