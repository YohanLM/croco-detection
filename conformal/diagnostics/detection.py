"""Per-image false-positive / true-positive box counters.

These have the same `(prediction_set, ground_truth) -> float` shape as a
LossFunction, but they are NOT losses and are NOT used in calibration. They
answer a precision question that the coverage/pixel loss cannot: when the
confidence-filter expansion admits more boxes (lower threshold), what do we
PAY for the coverage gain -- how many of the admitted boxes are false alarms
or sit on pure background?

Plug them into `Calibrator.evaluate(extra_metrics=...)`; the means/totals are
carried on `EvaluationResult.extra` and printed in the summary / comparison.

Matching is by IoU between a predicted box and the GT boxes:

  - true positive  : max IoU over GT >= iou_threshold (the box found a clip)
  - false positive : max IoU over GT <  iou_threshold (the box matched nothing
                     well enough -- a false alarm)
  - empty highlight: ZERO pixel intersection with every GT box -- a strict
                     subset of false positives: boxes drawn on background that
                     "highlight nothing"

On an image with no GT, every predicted box is both a false positive and an
empty highlight. Only columns 0-3 (xyxy, pixels) are read; a confidence
column, if present, is ignored.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _iou_and_intersection(
    preds: torch.Tensor, gts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU and intersection-area between pred and GT boxes.

    Returns `(iou, inter)`, each shape `[P, G]`. Both empty if either side is.
    """
    p = preds[:, :4]
    g = gts[:, :4]
    x1 = torch.maximum(p[:, None, 0], g[None, :, 0])
    y1 = torch.maximum(p[:, None, 1], g[None, :, 1])
    x2 = torch.minimum(p[:, None, 2], g[None, :, 2])
    y2 = torch.minimum(p[:, None, 3], g[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    pa = ((p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0))
    ga = ((g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0))
    union = pa[:, None] + ga[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return iou, inter


def make_detection_counters(
    iou_threshold: float = 0.10,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
    """Build the per-image counter dict for `evaluate(extra_metrics=...)`.

    `iou_threshold` is the IoU above which a predicted box counts as having
    found a GT clip. Lower it to be lenient about localization, raise it to
    demand tight boxes. Returns counters keyed by readable label.
    """

    def n_predictions(pred: torch.Tensor, gt: torch.Tensor) -> float:
        return float(pred.shape[0]) if pred.numel() else 0.0

    def n_true_positives(pred: torch.Tensor, gt: torch.Tensor) -> float:
        if pred.numel() == 0 or gt.numel() == 0:
            return 0.0
        iou, _ = _iou_and_intersection(pred, gt)
        return float((iou.max(dim=1).values >= iou_threshold).sum().item())

    def n_false_positives(pred: torch.Tensor, gt: torch.Tensor) -> float:
        if pred.numel() == 0:
            return 0.0
        if gt.numel() == 0:
            return float(pred.shape[0])  # nothing to match -> all are false
        iou, _ = _iou_and_intersection(pred, gt)
        return float((iou.max(dim=1).values < iou_threshold).sum().item())

    def n_empty_highlights(pred: torch.Tensor, gt: torch.Tensor) -> float:
        if pred.numel() == 0:
            return 0.0
        if gt.numel() == 0:
            return float(pred.shape[0])
        _, inter = _iou_and_intersection(pred, gt)
        return float((inter.sum(dim=1) == 0).sum().item())

    return {
        "predicted boxes": n_predictions,
        "true positives": n_true_positives,
        "false positives": n_false_positives,
        "empty highlights": n_empty_highlights,
    }
