"""Top-k selection wrapper around any `PredictionFunction`.

For a single-object detector, lowering the confidence threshold to recover
recall also surfaces many spurious / duplicate boxes for the one object.
`TopKPredictor` caps the prediction set at the `k` highest-confidence boxes
(k = 1 by default), so each frame yields at most one region — the duplicates
and low-confidence false positives never enter the pipeline.

This wraps a base predictor and is a drop-in `PredictionFunction`: it runs the
base detector at the given threshold, then keeps only the top-k boxes by the
confidence column (col 4). Composing it with `confidence_filter_expansion`
stays monotone — at k = 1 the single top box survives iff its score clears the
threshold, so lowering the threshold can only keep it more often (fewer
misses), exactly as the CRC root search requires.

Note the consequence of k = 1: lowering the threshold can no longer promote a
*different* box, only decide whether the top box survives. A frame whose top-1
box is a confident false positive is therefore unrecoverable by the threshold
knob — its detection-miss loss is locked. The feasible Phase-1 budget must
exceed that locked rate (`1 - "top-1 hits" rate`); see `diagnose_top1.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import PredictionFunction


class TopKPredictor:
    """Implements `PredictionFunction` — keep only the k highest-confidence boxes.

    Args:
        base: the wrapped `PredictionFunction` (e.g. a `YoloPredictor`).
        k: maximum number of boxes to keep per frame, by confidence. `1`
            (default) keeps only the single most-confident box — the natural
            choice for a single-object detector.
    """

    def __init__(self, base: PredictionFunction, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.base = base
        self.k = k

    def __call__(
        self, image_path: str, confidence_threshold: float
    ) -> torch.Tensor:
        preds = self.base(image_path, confidence_threshold)
        if preds.numel() == 0 or preds.shape[0] <= self.k:
            return preds
        top_idx = torch.topk(preds[:, 4], self.k).indices
        return preds[top_idx]


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_predictor: PredictionFunction = TopKPredictor.__new__(TopKPredictor)
