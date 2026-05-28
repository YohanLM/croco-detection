"""Multiplicative expansion C_λ (methodology Eq. 3).

Implements `ExpansionFunction`. Each side of every box grows by `w·λ` /
`h·λ`, preserving aspect ratio — critical for thin elongated targets (§3).
Monotone non-decreasing in λ.

The `confidence_threshold` argument is part of the protocol signature so
that confidence-aware expansions (e.g. low-conf boxes inflated more) can
slot in without changing the Calibrator. This particular expansion does
not use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import ExpansionFunction


def multiplicative_expansion(
    prediction_set: torch.Tensor,
    lam: float,
    confidence_threshold: float,
) -> torch.Tensor:
    """Implements `ExpansionFunction`.

    Each box grows by `w·λ` horizontally and `h·λ` vertically on every
    side. Operates on columns 0–3 only — any extra columns (e.g. the
    confidence score at column 4) are preserved untouched, so
    downstream confidence-aware components still see the scores.

    `confidence_threshold` is accepted to satisfy the protocol but is
    unused here.
    """
    del confidence_threshold  # accepted for protocol conformance, unused
    if prediction_set.numel() == 0:
        return prediction_set
    w = prediction_set[:, 2] - prediction_set[:, 0]
    h = prediction_set[:, 3] - prediction_set[:, 1]
    out = prediction_set.clone()  # carries column 4 (score) through unchanged
    out[:, 0] -= w * lam
    out[:, 1] -= h * lam
    out[:, 2] += w * lam
    out[:, 3] += h * lam
    return out


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_expansion: ExpansionFunction = multiplicative_expansion
