"""SeqCRC per-image losses (spec Section 4), single-class specialization.

Both losses are `[0, 1]`-valued (so the loss bound is `B = 1`) and take the
*full prefiltered* `[P, 5]` prediction set plus the parameters they depend on —
they construct `Gamma_cnf` internally, because the confidence step sweeps
`lambda_cnf` and needs the loss as a function of it. This is why they do not
fit the project's plain `LossFunction(expanded_pred, gt)` protocol and live
here rather than in `conformal/loss`.

Edge cases, handled in both (spec Section 4):
  - `|y| == 0` (no ground truth)      -> loss 0.
  - `Gamma_cnf` empty (nothing kept)  -> loss B = 1.

`l_cnf_image`  : confidence loss = false-negative rate on object **count**.
`l_loc_image`  : localization loss = 1 - recall over true boxes whose matched,
                 margin-expanded prediction superposes at least `tau_pix` of
                 their area.
"""

from __future__ import annotations

import torch

from conformal.seqcrc.config import MarginMode
from conformal.seqcrc.geometry import area, expand_boxes, intersection_area
from conformal.seqcrc.matching import match
from conformal.seqcrc.sets import confidence_set


def _n_valid_gt(ground_truth: torch.Tensor) -> int:
    """Count ground-truth boxes with strictly positive area."""
    if ground_truth.numel() == 0:
        return 0
    w = (ground_truth[:, 2] - ground_truth[:, 0]).clamp(min=0)
    h = (ground_truth[:, 3] - ground_truth[:, 1]).clamp(min=0)
    return int(((w * h) > 0).sum().item())


# ── Confidence loss: false-negative rate by count (spec Section 4.1) ──────────

def l_cnf_image(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    lambda_cnf: float,
    prefilter: float,
) -> float:
    """`L_cnf_i(lambda_cnf)` — count-based FNR = `max(0, |y| - |G|) / |y|`.

    `1 - recall` measured by box count: penalizes keeping fewer boxes than
    there are ground truths. Non-increasing in `lambda_cnf` (lower threshold =>
    more boxes => lower FNR), so it satisfies the CRC monotonicity assumption
    directly. An empty `Gamma_cnf` with `|y| > 0` yields `1.0 = B`.
    """
    n_gt = _n_valid_gt(ground_truth)
    if n_gt == 0:
        return 0.0
    n_keep = confidence_set(predictions, lambda_cnf, prefilter).shape[0]
    return max(0, n_gt - n_keep) / n_gt


# ── Localization loss: pixel-superposition threshold (spec Section 4.2) ───────

def l_loc_image(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    lambda_cnf: float,
    lambda_loc: float,
    prefilter: float,
    tau_pix: float,
    mode: MarginMode,
) -> float:
    """`L_loc_i(lambda_cnf, lambda_loc)` — pixel-superposition-threshold recall.

    A true box is "covered" when the fraction of its area overlapped by its
    matched, margin-expanded prediction reaches `tau_pix`; the loss is
    `1 - covered/|y|`. Non-increasing in `lambda_loc` (bigger margin => more
    coverage), but *not* necessarily in `lambda_cnf` (adding boxes re-matches),
    which is why calibration monotonizes it on the fly (spec Section 6).
    """
    n_gt = _n_valid_gt(ground_truth)
    if n_gt == 0:
        return 0.0
    kept = confidence_set(predictions, lambda_cnf, prefilter)
    if kept.shape[0] == 0:
        return 1.0

    expanded = expand_boxes(kept, lambda_loc, mode)
    matched = match(ground_truth, kept)        # match on Gamma_cnf (unexpanded)

    covered = 0
    for j in range(ground_truth.shape[0]):
        gt_area = area(ground_truth[j])
        if gt_area <= 0.0:
            continue                            # degenerate GT excluded from |y|
        b_hat = expanded[matched[j]]
        if intersection_area(ground_truth[j], b_hat) / gt_area >= tau_pix:
            covered += 1
    return 1.0 - covered / n_gt
