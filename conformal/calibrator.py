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

`Calibrator.evaluate(...)` is the test-time counterpart of `calibrate`:
given `λ̂`, it checks whether the CRC guarantee holds out of sample and at
what cost, returning an `EvaluationResult`. It is generic in the CRC
components (reads only the injected `risk_fn`), so the one thing that *is*
representation-specific — set size — is abstracted behind the
**EfficiencyMetric** protocol, mirroring `LossFunction`.

Concrete pixel-wise CRC components live under `conformal/loss`,
`conformal/expansion`, `conformal/efficiency`, and `conformal/prediction`.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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


# ── Efficiency (per-image set size) ──────────────────────────────────────────

class EfficiencyMetric(Protocol):
    """Per-image size of a prediction set: `(prediction_set) -> float ≥ 0`.

    Quantifies the *cost* of a conformal prediction set independently of
    whether it covers the GT — smaller is better, conditional on the risk
    guarantee holding. Where the loss says *is the GT covered?*, the
    efficiency metric says *how much did we have to predict to cover it?*

    Concrete metrics depend on the prediction-set representation (e.g.
    total box area in pixel space) and so live beside the matching
    loss/expansion components, not in this abstract module.
    """

    def __call__(self, prediction_set: torch.Tensor) -> float: ...


# ── Evaluation result ────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Outcome of evaluating a calibrated `λ` on a labeled split.

    Bundles the coverage check (`risk` vs `alpha`), the per-image loss
    distribution behind that mean, and — when an `EfficiencyMetric` is
    supplied — the size cost of the expansion. All fields are computed
    from a single prediction pass over the loader (see `Calibrator.evaluate`).
    """

    lam: float
    alpha: float
    n: int
    per_image_losses: list[float]
    risk: float                                  # R̂_n = mean per-image loss
    crc_bound: float                             # finite-sample corrected risk
    per_image_efficiency: list[float] | None = None
    mean_efficiency: float | None = None         # set size at `lam`
    mean_raw_efficiency: float | None = None     # set size of the raw (λ=0) set
    risk_curve: list[tuple[float, float]] | None = None
    # Optional non-loss diagnostics (e.g. FP/TP counts): name -> per-image list.
    extra: dict[str, list[float]] | None = None

    @property
    def coverage_satisfied(self) -> bool:
        """Whether the empirical test risk respects the target level α."""
        return self.risk <= self.alpha

    @property
    def slack(self) -> float:
        """`alpha - risk` — headroom (>0) or violation (<0) vs the target."""
        return self.alpha - self.risk

    @property
    def inflation_ratio(self) -> float | None:
        """Mean set size at `lam` relative to the raw set — the cost of λ.

        `None` unless an efficiency metric was supplied and the raw set has
        nonzero size.
        """
        if not self.mean_raw_efficiency:
            return None
        return self.mean_efficiency / self.mean_raw_efficiency

    @property
    def n_perfect(self) -> int:
        """Images with zero loss — GT fully covered."""
        return sum(1 for loss in self.per_image_losses if loss == 0.0)

    @property
    def n_locked(self) -> int:
        """Images with maximal loss (1.0) — typically zero surviving preds."""
        return sum(1 for loss in self.per_image_losses if loss == 1.0)

    @property
    def extra_totals(self) -> dict[str, float]:
        """Sum of each extra diagnostic over the split (e.g. total FP boxes)."""
        if not self.extra:
            return {}
        return {k: sum(v) for k, v in self.extra.items()}

    @property
    def extra_means(self) -> dict[str, float]:
        """Per-image mean of each extra diagnostic (e.g. mean FP per image)."""
        if not self.extra:
            return {}
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.extra.items()}

    # ── Reporting (text; ASCII-only so it is safe on any console/locale) ──────

    def summary(
        self,
        *,
        title: str | None = None,
        eff_name: str = "set size",
        eff_unit: str = "",
        include_curve: bool = True,
    ) -> str:
        """Human-readable multi-line report of this evaluation.

        Generic in the CRC components — it prints only what the result holds
        (risk vs alpha, the loss distribution, and, when present, the
        efficiency cost and risk curve). Callers pass `eff_name`/`eff_unit`
        to label the efficiency metric for their pipeline (e.g. "box area" /
        "px^2", "boxes admitted" / "boxes").
        """
        unit = f" {eff_unit}" if eff_unit else ""
        lines: list[str] = []
        if title:
            lines.append(title)
        lines.append(f"  test images (n)       : {self.n}")
        lines.append(f"  lambda                : {self.lam:.4f}")
        lines.append(f"  test risk  R(lambda)  : {self.risk:.4f}")
        lines.append(f"  target     alpha      : {self.alpha:.4f}")
        lines.append(f"  slack (alpha - risk)  : {self.slack:+.4f}")
        lines.append(f"  CRC finite-sample bnd : {self.crc_bound:.4f}")
        lines.append(f"  images fully covered  : {self.n_perfect}/{self.n}")
        lines.append(f"  images fully missed   : {self.n_locked}/{self.n} "
                     "(loss = 1.0)")
        if self.per_image_losses:
            lines.append("  per-image loss median : "
                         f"{statistics.median(self.per_image_losses):.4f}")
            lines.append("  per-image loss max    : "
                         f"{max(self.per_image_losses):.4f}")
        if self.mean_efficiency is not None:
            lines.append(f"  {('mean ' + eff_name):<22}: "
                         f"{self.mean_efficiency:,.0f}{unit}")
            if self.mean_raw_efficiency is not None:
                lines.append(f"  {('mean raw ' + eff_name):<22}: "
                             f"{self.mean_raw_efficiency:,.0f}{unit}")
            if self.inflation_ratio is not None:
                lines.append(f"  {'inflation ratio':<22}: "
                             f"{self.inflation_ratio:.2f}x")
        if self.extra:
            totals = self.extra_totals
            means = self.extra_means
            for name in self.extra:
                lines.append(f"  {name:<22}: {totals[name]:,.0f} total "
                             f"({means[name]:.3f} per image)")
        if include_curve and self.risk_curve:
            lines.append("")
            lines.append(f"  {'lambda':>8} | {'risk':>8}")
            lines.append(f"  {'-' * 8}-+-{'-' * 8}")
            half_step = self._curve_half_step()
            for lam, risk in self.risk_curve:
                mark = "  <- lambda-hat" if abs(lam - self.lam) < half_step else ""
                flag = " (<=alpha)" if risk <= self.alpha else ""
                lines.append(f"  {lam:>8.3f} | {risk:>8.4f}{flag}{mark}")
        return "\n".join(lines)

    def _curve_half_step(self) -> float:
        """Half the lambda grid spacing — tolerance for marking lambda-hat."""
        if not self.risk_curve or len(self.risk_curve) < 2:
            return 1e-6
        lams = [lam for lam, _ in self.risk_curve]
        return (max(lams) - min(lams)) / (2 * (len(lams) - 1))

    @property
    def verdict(self) -> str:
        """One-line PASS/FAIL banner for the coverage guarantee."""
        if self.coverage_satisfied:
            return (f"PASS -- test risk {self.risk:.4f} <= alpha {self.alpha:.4f}: "
                    "CRC guarantee held out of sample.")
        return (f"FAIL -- test risk {self.risk:.4f} > alpha {self.alpha:.4f}: "
                "guarantee violated on the test set.")

    @staticmethod
    def comparison(
        without: "EvaluationResult",
        with_: "EvaluationResult",
        *,
        eff_name: str = "set size",
        baseline_label: str = "without",
        calib_label: str = "with",
    ) -> str:
        """Side-by-side table of a baseline result vs a calibrated result.

        `without` is typically the model at its normal operating point and
        `with_` the result at lambda-hat; both must share the same alpha.
        """
        alpha = with_.alpha
        col_no = f"without ({baseline_label})"
        col_yes = f"with ({calib_label})"
        lines = [f"  {'metric':<24}{col_no:>22}{col_yes:>22}",
                 f"  {'-' * 24}{'-' * 22}{'-' * 22}",
                 f"  {'mean risk':<24}{without.risk:>22.4f}{with_.risk:>22.4f}",
                 f"  {'meets alpha?':<24}"
                 f"{str(without.risk <= alpha):>22}{str(with_.risk <= alpha):>22}",
                 f"  {'images fully covered':<24}"
                 f"{f'{without.n_perfect}/{without.n}':>22}"
                 f"{f'{with_.n_perfect}/{with_.n}':>22}",
                 f"  {'images fully missed':<24}"
                 f"{f'{without.n_locked}/{without.n}':>22}"
                 f"{f'{with_.n_locked}/{with_.n}':>22}"]
        if without.mean_efficiency is not None and with_.mean_efficiency is not None:
            lines.append(f"  {f'mean {eff_name}':<24}"
                         f"{without.mean_efficiency:>22,.0f}"
                         f"{with_.mean_efficiency:>22,.0f}")
        # Extra diagnostics (e.g. FP/TP): totals side by side, shared keys only.
        wt = without.extra_totals
        ct = with_.extra_totals
        for name in ct:
            if name in wt:
                lines.append(f"  {('total ' + name):<24}"
                             f"{wt[name]:>22,.0f}{ct[name]:>22,.0f}")
        risk_drop = without.risk - with_.risk
        fewer = without.n_locked - with_.n_locked
        lines.append("")
        lines.append(f"  calibration reduced test risk by {risk_drop:+.4f} "
                     f"({fewer:+d} fewer fully-missed images)")
        # If FP diagnostics are present, surface the precision cost explicitly.
        if "false positives" in wt and "false positives" in ct:
            dfp = ct["false positives"] - wt["false positives"]
            lines.append(f"  precision cost (FP)   : {dfp:+,.0f} false-positive "
                         "boxes vs the baseline")
        if "empty highlights" in wt and "empty highlights" in ct:
            deh = ct["empty highlights"] - wt["empty highlights"]
            lines.append(f"  precision cost (bg)   : {deh:+,.0f} background boxes "
                         "(highlight nothing) vs the baseline")
        return "\n".join(lines)


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
        lam = float(brentq(crc_gap, lo, hi, xtol=1e-6))
        # The CRC gap is a step function (discrete loss): brentq can land just
        # before a step where the gap is still slightly positive. If so, step
        # forward by 2×xtol — smaller than any meaningful loss step — to land
        # on the guaranteed side (gap ≤ 0) without adding measurable conservatism.
        if crc_gap(lam) > 0:
            lam = min(lam + 2e-6, hi)
        return lam

    # ── evaluation: measure the effect of a calibrated λ on a labeled split ──

    def _risk_curve(
        self,
        preds: list[torch.Tensor],
        gts: list[torch.Tensor],
        lambdas: Iterable[float],
    ) -> list[tuple[float, float]]:
        """`R̂(λ)` over a λ sweep on already-collected predictions.

        Reuses cached predictions — only `_apply_expansion` is re-run per λ,
        the same trick `calibrate` uses to keep cost at N model calls.
        """
        curve = []
        for lam in lambdas:
            expanded = [self._apply_expansion(p, float(lam)) for p in preds]
            curve.append((float(lam), self.risk_fn(expanded, gts)))
        return curve

    def evaluate(
        self,
        loader: DataLoader,
        lam: float,
        *,
        efficiency_fn: EfficiencyMetric | None = None,
        risk_curve_lambdas: Iterable[float] | None = None,
        extra_metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        | None = None,
    ) -> EvaluationResult:
        """Measure the effect of a calibrated `λ` on a labeled (test) split.

        This is the test-time counterpart of `calibrate`: pass `lam = λ̂` to
        check whether the CRC guarantee `E[L] ≤ α` actually holds out of
        sample, and at what cost. Everything is computed from a single
        prediction pass — predictions are cached and reused across the
        per-image losses, the efficiency metric, and the optional risk curve.

        Args:
            loader: yields `(image_paths, ground_truth)` — same format as the
                calibration loader.
            lam: the margin to evaluate, typically `λ̂` from `calibrate`.
            efficiency_fn: optional `EfficiencyMetric`. When given, the result
                carries the mean set size at `lam`, the raw (λ=0) set size, and
                their ratio (`inflation_ratio`) — the geometric cost of λ.
            risk_curve_lambdas: optional λ grid; when given, the result carries
                `R̂(λ)` over the grid for plotting against α and `λ̂`.
            extra_metrics: optional `{name: fn(prediction_set, ground_truth)}`
                of non-loss diagnostics (e.g. false-positive / true-positive
                counts). Computed on the same expanded predictions and carried
                on `EvaluationResult.extra` for the report — they never affect
                calibration.

        Returns:
            An `EvaluationResult`. Generic in the CRC components — it reads the
            injected `risk_fn` / per-image loss only, so it works for any loss
            (pixel, coverage-indicator, …) without modification.
        """
        preds, gts = self._collect(loader)
        n = len(preds)
        if n == 0:
            raise RuntimeError("Evaluation set is empty.")

        expanded = [self._apply_expansion(p, lam) for p in preds]
        per_image_losses = [
            self.risk_fn.loss_fn(e, g) for e, g in zip(expanded, gts)
        ]
        risk = sum(per_image_losses) / n

        per_image_eff = mean_eff = mean_raw_eff = None
        if efficiency_fn is not None:
            per_image_eff = [efficiency_fn(e) for e in expanded]
            mean_eff = sum(per_image_eff) / n
            mean_raw_eff = sum(efficiency_fn(p) for p in preds) / n

        curve = None
        if risk_curve_lambdas is not None:
            curve = self._risk_curve(preds, gts, risk_curve_lambdas)

        extra = None
        if extra_metrics:
            extra = {
                name: [fn(e, g) for e, g in zip(expanded, gts)]
                for name, fn in extra_metrics.items()
            }

        return EvaluationResult(
            lam=float(lam),
            alpha=self.alpha,
            n=n,
            per_image_losses=per_image_losses,
            risk=risk,
            crc_bound=crc_finite_sample_correction(risk, n),
            per_image_efficiency=per_image_eff,
            mean_efficiency=mean_eff,
            mean_raw_efficiency=mean_raw_eff,
            risk_curve=curve,
            extra=extra,
        )


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
