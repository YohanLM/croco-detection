"""Confidence-threshold expansion — keeps boxes whose score clears a
λ-dependent cutoff.

Implements `ExpansionFunction`. As λ grows, the cutoff falls, so more
boxes survive — monotone non-decreasing in λ, as CRC requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conformal.calibrator import ExpansionFunction


def confidence_filter_expansion(
    prediction_set: torch.Tensor,
    lam: float,
    confidence_threshold: float,
) -> torch.Tensor:
    """Filter the prediction set by a λ-dependent confidence cutoff.

    Effective cutoff:

        T_eff(λ) = max(confidence_threshold, 1.0 − λ)

      - At λ = 0: cutoff = 1.0 — only boxes with score ≥ 1.0 (essentially
        none) survive, the prediction set is empty, the loss locks at 1.0.
      - As λ grows, cutoff falls linearly.
      - At λ ≥ 1 − confidence_threshold: cutoff = confidence_threshold,
        no further filtering happens — the full input passes through.

    Monotone non-decreasing in λ ✓ (cutoff falls → more boxes survive →
    larger / more inclusive set → loss can only stay equal or decrease).

    Reads the score from column 4 of `prediction_set`, per the
    `PredictionFunction` output contract. Geometric box coordinates
    (columns 0–3) are not modified.
    """
    if prediction_set.numel() == 0:
        return prediction_set
    effective_cutoff = max(confidence_threshold, 1.0 - lam)
    scores = prediction_set[:, 4]
    return prediction_set[scores >= effective_cutoff]


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_expansion: ExpansionFunction = confidence_filter_expansion
