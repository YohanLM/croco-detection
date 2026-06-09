"""Hausdorff matching (spec Section 3), specialized to pure localization.

With a single object class the classification distance vanishes, so the mixed
matching distance collapses to the asymmetric signed Hausdorff distance alone:

    d_haus(b, b_hat) = max( b_hat_left - b_left,
                            b_hat_top  - b_top,
                            b_right     - b_hat_right,
                            b_bottom    - b_hat_bottom )

— the smallest per-side margin that, added to `b_hat`, makes it fully cover the
true box `b`. Each true box is matched to its nearest prediction in
`Gamma_cnf` under this distance. The matching need not be injective (several
true boxes may share one prediction) and is recomputed whenever `Gamma_cnf`
changes, i.e. whenever `lambda_cnf` changes.
"""

from __future__ import annotations

import torch


def d_haus(true_box: torch.Tensor, pred_box: torch.Tensor) -> float:
    """Asymmetric signed Hausdorff distance from prediction to truth (scalar)."""
    return float(
        max(
            (pred_box[0] - true_box[0]).item(),
            (pred_box[1] - true_box[1]).item(),
            (true_box[2] - pred_box[2]).item(),
            (true_box[3] - pred_box[3]).item(),
        )
    )


def match(ground_truth: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Match each true box to its nearest prediction under `d_haus`.

    Args:
        ground_truth: true boxes `[G, >=4]`, pixel `xyxy`.
        candidates: the confidence set `Gamma_cnf`, `[P, >=4]`, pixel `xyxy`.

    Returns:
        `LongTensor[G]` — for each true box, the index of its nearest
        prediction in `candidates`. Empty if there are no candidates (callers
        must handle the empty-`Gamma_cnf` case before matching).
    """
    g = ground_truth.shape[0]
    p = candidates.shape[0]
    if g == 0 or p == 0:
        return torch.empty(0, dtype=torch.long)

    gt = ground_truth[:, None, :4]    # [G, 1, 4]
    cand = candidates[None, :, :4]    # [1, P, 4]
    dist = torch.maximum(
        torch.maximum(cand[..., 0] - gt[..., 0], cand[..., 1] - gt[..., 1]),
        torch.maximum(gt[..., 2] - cand[..., 2], gt[..., 3] - cand[..., 3]),
    )                                 # [G, P]
    return dist.argmin(dim=1)
