"""Additive expansion — adds a fixed pixel margin λ to every side of every box.

Implements `ExpansionFunction`. Unlike the multiplicative expansion, the margin
is the same absolute number of pixels regardless of box size, so small and large
clips receive the same absolute padding. This can be preferable when clips vary
little in size or when a uniform safety margin in pixels is operationally
meaningful.

Monotone non-decreasing in λ: larger λ → larger boxes → loss can only decrease.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import ExpansionFunction


def additive_expansion(
    prediction_set: torch.Tensor,
    lam: float,
    confidence_threshold: float,
) -> torch.Tensor:
    """Implements `ExpansionFunction`.

    Each box grows by `λ` pixels on every side (x1 -= λ, y1 -= λ,
    x2 += λ, y2 += λ). Operates on columns 0–3 only — any extra columns
    (e.g. the confidence score at column 4) are preserved untouched.

    `confidence_threshold` is accepted to satisfy the protocol but is
    unused here.
    """
    del confidence_threshold  # accepted for protocol conformance, unused
    if prediction_set.numel() == 0:
        return prediction_set
    out = prediction_set.clone()  # carries column 4 (score) through unchanged
    out[:, 0] -= lam
    out[:, 1] -= lam
    out[:, 2] += lam
    out[:, 3] += lam
    return out


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_expansion: ExpansionFunction = additive_expansion
