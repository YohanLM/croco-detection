"""Box geometry for SeqCRC — analytic area, intersection, and margin expansion.

`area` and `intersection_area` operate on single pixel-`xyxy` boxes (shape
`[>=4]`) and back the pixel-superposition loss (`losses.l_loc_image`): the
"superposition fraction" of a true box `b_j` against its matched prediction is
`intersection_area(b_j, b_hat) / area(b_j)` (spec Section 4.2).

`expand_boxes` is the localization-set operator `expand(., lambda_loc)` of the
spec (Section 2.2). It does *not* re-implement margin math — it dispatches to
the project's existing, CRC-monotone expansion functions, so calibration and
inference share exactly one definition of "grow a box":

  - "additive"       -> additive_expansion        b + lam * (-1, -1, +1, +1)
  - "multiplicative" -> multiplicative_expansion   b + lam * (-w, -h, +w, +h)

Both expansions touch only columns 0-3 and carry the confidence column through
untouched, so a `[P, 5]` prediction set stays a `[P, 5]` set after expansion.
"""

from __future__ import annotations

import torch

from conformal.expansion.additive import additive_expansion
from conformal.expansion.multiplicative import multiplicative_expansion
from conformal.seqcrc.config import MarginMode


# ── Areas ─────────────────────────────────────────────────────────────────────

def area(box: torch.Tensor) -> float:
    """Pixel area of one `xyxy` box; `0.0` for a degenerate (non-positive) box."""
    w = (box[2] - box[0]).clamp(min=0)
    h = (box[3] - box[1]).clamp(min=0)
    return float((w * h).item())


def intersection_area(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pixel area of the overlap of two `xyxy` boxes; `0.0` if disjoint."""
    x1 = torch.maximum(a[0], b[0])
    y1 = torch.maximum(a[1], b[1])
    x2 = torch.minimum(a[2], b[2])
    y2 = torch.minimum(a[3], b[3])
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return float((w * h).item())


# ── Localization expansion (spec Section 2.2) ─────────────────────────────────

def expand_boxes(
    boxes: torch.Tensor, lam: float, mode: MarginMode
) -> torch.Tensor:
    """Apply the `expand(., lambda_loc)` operator under the chosen margin mode.

    Reuses the project's `ExpansionFunction`s; the `confidence_threshold`
    argument they accept for protocol conformance is irrelevant for a pure
    geometric expansion and passed as `0.0`.
    """
    if mode == "additive":
        return additive_expansion(boxes, lam, 0.0)
    if mode == "multiplicative":
        return multiplicative_expansion(boxes, lam, 0.0)
    raise ValueError(
        f"margin_mode must be 'additive' or 'multiplicative', got {mode!r}"
    )
