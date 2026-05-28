"""Abstract framework for Conformal Risk Control calibration.

Defines the pluggable components every CRC pipeline needs — prediction,
expansion, per-image loss, and the set-level risk wrapper — plus the
`Calibrator` class that composes them into the finite-sample CRC procedure:

    λ̂ = inf { λ : (n / (n+1)) · R̂_n(λ) + 1/(n+1) ≤ α }

The loss vs. risk split is deliberate: the user defines the per-image
**LossFunction** (what failure means on one image); the set-level
**Risk** is just the mean of the loss over the calibration set —
`R̂_n = (1/n) Σ L_i`. The root search is fixed to Brent's method.

The **ExpansionFunction** turns a raw prediction set into a conformal
prediction set. Its signature `(prediction_set, lam, confidence_threshold)`
is wide enough to host confidence-aware variants (e.g. low-confidence
boxes inflated more, score-dependent margins) without changing the
Calibrator. Geometric-only expansions just ignore the threshold.

Concrete pixel-wise CRC components live under `conformal/loss`,
`conformal/expansion`, and `conformal/prediction`.
"""

from __future__ import annotations

from typing import Protocol

import torch
from scipy.optimize import brentq
from torch.utils.data import DataLoader


# ── Pluggable component types ────────────────────────────────────────────────

class PredictionFunction(Protocol):
    """Run a detector on a single image; return the raw prediction set.

    Output format — **strict contract**:

      - Shape `[P, 5]`, dtype `float32`.
      - Columns 0–3: `x_min, y_min, x_max, y_max` in **pixel coordinates**.
        Outputs MUST NOT be normalized to `[0, 1]` — the expansion and
        pixel-recall loss both assume pixel space.
      - Column 4: per-box confidence score, preserved end-to-end so
        confidence-aware expansions can read it.
      - Filtered at `confidence_threshold` (boxes below it dropped).
      - Empty results MUST be returned as `torch.zeros((0, 5))`, not `None`.

    Downstream consumers (expansion, loss) treat columns ≥ 4 as carried-through
    metadata: expansions preserve them; the pixel loss ignores them.
    """

    def __call__(
        self, image_path: str, confidence_threshold: float
    ) -> torch.Tensor: ...


class ExpansionFunction(Protocol):
    """Turn a raw prediction set into a conformal prediction set.

    Signature: `(prediction_set, lam, confidence_threshold) → expanded_set`.

    Must be monotone non-decreasing in `lam` (larger λ → larger / more
    inclusive sets) so the downstream risk is monotone non-increasing in
    λ — required for the CRC root search to be well-defined.

    `confidence_threshold` is provided so confidence-aware expansions can
    use it (e.g. a low-confidence box gets a larger margin). Purely
    geometric expansions are free to ignore it.
    """

    def __call__(
        self,
        prediction_set: torch.Tensor,
        lam: float,
        confidence_threshold: float,
    ) -> torch.Tensor: ...


# ── Loss (per-image) and Risk (set-level mean) ───────────────────────────────

class LossFunction(Protocol):
    """Per-image loss: `(prediction_set, ground_truth) -> float in [0, 1]`.

    Defines what failure means on a single image. Must be monotone
    non-increasing in the expansion applied upstream so the set-level
    risk is monotone non-increasing in λ — required for the CRC root
    search.
    """

    def __call__(
        self,
        prediction_set: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> float: ...


class Risk:
    """Set-level empirical risk = mean of a `LossFunction` over a sample.

    `R̂_n = (1/n) Σ_i L_i`. Constructing a new risk is just wrapping a
    different per-image loss — `Risk(other_loss)`.
    """

    def __init__(self, loss_fn: LossFunction) -> None:
        self.loss_fn = loss_fn

    def __call__(
        self,
        prediction_sets: list[torch.Tensor],
        ground_truth_sets: list[torch.Tensor],
    ) -> float:
        n = len(prediction_sets)
        if n == 0:
            return 0.0
        return sum(
            self.loss_fn(p, g)
            for p, g in zip(prediction_sets, ground_truth_sets)
        ) / n


# ── Calibrator ───────────────────────────────────────────────────────────────

class Calibrator:
    """Composes prediction, expansion, and risk to find λ̂ via Brent's method.

    The Calibrator holds no model — `prediction_fn` is injected. Given a
    calibration loader yielding `(image_paths, ground_truth)` per batch
    (whatever GT format the loss expects), `calibrate(...)` returns the
    calibrated multiplier `λ̂`. The `confidence_threshold` is held once
    and threaded into both `prediction_fn` and `expansion_fn` calls.
    """

    def __init__(
        self,
        prediction_fn: PredictionFunction,
        expansion_fn: ExpansionFunction,
        risk_fn: Risk,
        alpha: float,
        confidence_threshold: float,
    ) -> None:
        self.prediction_fn = prediction_fn
        self.expansion_fn = expansion_fn
        self.risk_fn = risk_fn
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

    # ── infer = predict ∘ expand (the two atomic steps reused by calibrate) ──

    def _predict_raw(self, image_path: str) -> torch.Tensor:
        """Step 1 of inference — raw predictions for one image."""
        return self.prediction_fn(image_path, self.confidence_threshold)

    def _apply_expansion(self, raw_predictions: torch.Tensor, lam: float) -> torch.Tensor:
        """Step 2 of inference — expand a raw prediction set at margin λ.

        Threads `self.confidence_threshold` through, so confidence-aware
        expansions get the same value used at prediction time.
        """
        return self.expansion_fn(raw_predictions, lam, self.confidence_threshold)

    def infer(self, image_path: str, lam: float) -> torch.Tensor:
        """Runtime inference (methodology §4.3) — `_apply_expansion(_predict_raw(img), λ)`.

        At deployment, pass `lam = λ̂` returned by `calibrate(...)`. Pass any
        other value to probe the prediction set at a non-calibrated margin.
        """
        return self._apply_expansion(self._predict_raw(image_path), lam)

    # ── calibration: predict once, expand per candidate λ ────────────────────

    def _collect(
        self, loader: DataLoader
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run `_predict_raw` once over the loader; return paired lists.

        Predictions are cached across the λ sweep — only `_apply_expansion`
        is re-applied per candidate, so calibration cost stays at N model
        calls regardless of how many λ values Brent's method probes.
        """
        preds: list[torch.Tensor] = []
        gts: list[torch.Tensor] = []
        for paths, gt_batch in loader:
            for path, gt in zip(paths, gt_batch):
                preds.append(self._predict_raw(path))
                gts.append(gt)
        return preds, gts

    def calibrate(
        self,
        calibration_loader: DataLoader,
        lambda_range: tuple[float, float] = (0.0, 2.0),
    ) -> float:
        """Return `λ̂` — the smallest λ satisfying the CRC bound at level α.

        Each candidate λ is evaluated on boxes produced by the same
        `_apply_expansion` step that `infer` uses, so the loss measured
        here matches the loss at deployment exactly.
        """
        preds, gts = self._collect(calibration_loader)
        n = len(preds)
        if n == 0:
            raise RuntimeError("Calibration set is empty.")

        def crc_gap(lam: float) -> float:
            expanded = [self._apply_expansion(p, lam) for p in preds]
            empirical_risk = self.risk_fn(expanded, gts)
            return crc_finite_sample_correction(empirical_risk, n) - self.alpha

        lo, hi = lambda_range
        if crc_gap(lo) <= 0:
            return lo
        if crc_gap(hi) > 0:
            locked = sum(1 for p, g in zip(preds, gts) if p.numel() == 0 and g.numel() > 0)
            raise RuntimeError(
                f"No λ in [{lo}, {hi}] satisfies the CRC bound at α={self.alpha}: "
                f"gap at λ={hi} is still positive. "
                f"{locked}/{n} calibration images have zero surviving predictions "
                "and lock the per-image loss at 1.0 regardless of λ "
                "(see methodology §5)."
            )
        return float(brentq(crc_gap, lo, hi, xtol=1e-4))


# ── CRC finite-sample correction ─────────────────────────────────────────────

def crc_finite_sample_correction(empirical_risk: float, n: int) -> float:
    """Inflate an empirical risk into a CRC-valid finite-sample upper bound.

    For `n` iid calibration samples and empirical risk `R̂_n`:

        R̃ = (n / (n+1)) · R̂_n + 1/(n+1)

    The CRC infimum rule (methodology Eq. 5) then selects
        λ̂ = inf { λ : R̃(λ) ≤ α }
    which gives the exchangeability-only guarantee `E[L(λ̂)] ≤ α`.

    As `n → ∞`, `R̃ → R̂_n` — the correction vanishes. For small `n` it
    matters: the `+ 1/(n+1)` term ensures the bound holds even when one
    extra worst-case sample could appear at test time.
    """
    return n / (n + 1) * empirical_risk + 1.0 / (n + 1)
