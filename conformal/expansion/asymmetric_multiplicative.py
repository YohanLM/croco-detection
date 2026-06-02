"""Asymmetric multiplicative expansion — reduced horizontal growth (methodology variant).

Implements `ExpansionFunction`. Identical to `multiplicative_expansion` except
the horizontal margin is scaled by 1/3: each box grows by `w·λ/3` on the left
and right sides, and by `h·λ` on the top and bottom. Useful when horizontal
over-coverage is costly (e.g. narrow elongated targets where lateral spill
creates false associations) while vertical coverage must remain generous.
Monotone non-decreasing in λ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import ExpansionFunction

_HORIZONTAL_FACTOR = 1/3 


def asymmetric_multiplicative_expansion(
    prediction_set: torch.Tensor,
    lam: float,
    confidence_threshold: float,
) -> torch.Tensor:
    """Implements `ExpansionFunction`.

    Horizontal sides grow by `w·λ/3` per side; vertical sides grow by `h·λ`
    per side. Columns 4+ (e.g. confidence score) are preserved untouched.

    `confidence_threshold` is accepted to satisfy the protocol but is unused.
    """
    del confidence_threshold  # accepted for protocol conformance, unused
    if prediction_set.numel() == 0:
        return prediction_set
    w = prediction_set[:, 2] - prediction_set[:, 0]
    h = prediction_set[:, 3] - prediction_set[:, 1]
    out = prediction_set.clone()
    out[:, 0] -= w * lam * _HORIZONTAL_FACTOR
    out[:, 1] -= h * lam
    out[:, 2] += w * lam * _HORIZONTAL_FACTOR
    out[:, 3] += h * lam
    return out


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_expansion: ExpansionFunction = asymmetric_multiplicative_expansion
