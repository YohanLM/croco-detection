"""Two-step SeqCRC calibration (spec Sections 5-7), single-class specialization.

The procedure collapses the paper's "1+2" to "1+1": one confidence step then
one localization step, reusing a single calibration split for both (the
`lambda_cnf_minus` estimator is what makes that valid without a second split).

  Step 1 (confidence) runs `calibrate_confidence` twice — with `B = 1`
  (`lambda_cnf_plus`, used at inference) and `B = 0` (`lambda_cnf_minus`, the
  optimistic estimator fed to Step 2). Both sweep `lambda_cnf` from 1 down over
  the sorted confidence scores, accumulating the conservative risk
  `R_tilde_cnf = max(R_cnf, R_loc(., lambda_bar_loc))` and monotonizing
  `L_loc` in `lambda_cnf` on the fly (running sup; spec Section 6).

  Step 2 (localization) runs `calibrate_localization`: a binary search over
  `lambda_loc in [0, lambda_bar_loc]`, each step recomputing the monotonized
  `R_loc(lambda_cnf_minus, lambda_loc)` by the same downward sweep, capped at
  `lambda_cnf_minus`.

The finite-sample test `n/(n+1) R + B/(n+1) <= alpha` reuses the project's
`crc_finite_sample_correction` (with `b in {0, 1}`).

`calibrate(...)` ties it together: predict + prefilter once, then run the three
sub-searches. Only the `_plus` values are used at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from conformal.calibrator import (
    PredictionFunction,
    crc_finite_sample_correction,
)
from conformal.seqcrc.config import SeqCRCConfig
from conformal.seqcrc.losses import l_cnf_image, l_loc_image


# ── Prediction collection (spec Section 7, pre-step) ──────────────────────────

def collect_predictions(
    predictor: PredictionFunction,
    loader: DataLoader,
    prefilter: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the detector once over `loader`; return `(predictions, ground_truth)`.

    The detector is run at the `prefilter` floor, so every box returned already
    satisfies the data-independent `conf >= prefilter` pre-step. Uses
    `predict_batch` when the predictor exposes it (one GPU call per batch).
    """
    use_batch = hasattr(predictor, "predict_batch")
    preds: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    for paths, gt_batch in loader:
        if use_batch:
            preds.extend(predictor.predict_batch(list(paths), prefilter))
        else:
            preds.extend(predictor(p, prefilter) for p in paths)
        gts.extend(gt_batch)
    return preds, gts


# ── Sorted-score sweep helper (shared by both steps) ──────────────────────────

def _sweep_candidates(
    predictions: list[torch.Tensor],
) -> tuple[list[float], list[int]]:
    """Flatten all confidence scores with their image index, sorted ascending,
    then shift-left the thresholds so each step *lowers* `lambda_cnf` past one
    box (spec Section 7.2/7.3: `scores = shift_left(scores); scores[last] = 1`).

    Returns `(candidate_scores, image_index)` where, at step `k`, evaluating at
    `lambda_cnf = 1 - candidate_scores[k]` removes the `k`-th box (owned by
    image `image_index[k]`) from its `Gamma_cnf`.
    """
    flat = [
        (float(score), i)
        for i, p in enumerate(predictions)
        if p.numel() > 0
        for score in p[:, 4].tolist()
    ]
    flat.sort(key=lambda t: t[0])               # ascending score
    image_index = [i for _, i in flat]
    scores = [s for s, _ in flat]
    candidates = scores[1:] + [1.0] if scores else []   # shift-left; last -> 1
    return candidates, image_index


# ── Step 1: confidence calibration (spec Section 7.2) ─────────────────────────

def calibrate_confidence(
    predictions: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    cfg: SeqCRCConfig,
    loss_bound: float,
) -> float:
    """Return `lambda_cnf` for finite-sample bound `B = loss_bound` (1 or 0).

    `B = 1` yields `lambda_cnf_plus` (inference); `B = 0` yields
    `lambda_cnf_minus` (optimistic, feeds Step 2). Sweeps `lambda_cnf` from 1
    downward and returns the infimum `lambda_cnf` still satisfying
    `n/(n+1) R_tilde_cnf + B/(n+1) <= alpha_cnf`. Convention: an empty feasible
    set returns `lambda_bar_cnf = 1`.
    """
    n = len(predictions)
    pf, abar = cfg.prefilter, cfg.lambda_bar_loc

    # Per-image loss arrays initialized at lambda_cnf = 1 (keep everything).
    l_cnf = [l_cnf_image(predictions[i], ground_truth[i], 1.0, pf) for i in range(n)]
    # L_loc monotonization base, evaluated at the maximum margin lambda_bar_loc.
    l_loc = [
        l_loc_image(predictions[i], ground_truth[i], 1.0, abar, pf,
                    cfg.tau_pix, cfg.margin_mode)
        for i in range(n)
    ]
    sum_cnf, sum_loc = sum(l_cnf), sum(l_loc)

    def risk() -> float:
        # R_tilde_cnf = max(R_cnf, R_loc(., lambda_bar_loc)) — spec Section 5.
        return max(sum_cnf / n, sum_loc / n)

    def violates(r: float) -> bool:
        return crc_finite_sample_correction(r, n, loss_bound) > cfg.alpha_cnf

    # Infeasible even at lambda_cnf = 1 (most boxes kept) -> lambda_bar_cnf = 1.
    if violates(risk()):
        return 1.0

    candidates, image_index = _sweep_candidates(predictions)
    lambda_cnf = 1.0
    for c, i in zip(candidates, image_index):
        prev = lambda_cnf
        lambda_cnf = 1.0 - c

        new_cnf = l_cnf_image(predictions[i], ground_truth[i], lambda_cnf, pf)
        sum_cnf += new_cnf - l_cnf[i]
        l_cnf[i] = new_cnf

        # Running sup => smallest monotone (in lambda_cnf) upper bound of L_loc.
        candidate_loc = l_loc_image(predictions[i], ground_truth[i], lambda_cnf,
                                    abar, pf, cfg.tau_pix, cfg.margin_mode)
        if candidate_loc > l_loc[i]:
            sum_loc += candidate_loc - l_loc[i]
            l_loc[i] = candidate_loc

        if violates(risk()):
            return prev                          # first violation -> infimum is prev
    return 0.0                                   # bound held all the way down


# ── Step 2: localization calibration (spec Section 7.3) ───────────────────────

def calibrate_localization(
    predictions: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    cfg: SeqCRCConfig,
    lambda_cnf_minus: float,
    loss_bound: float = 1.0,
) -> float:
    """Binary-search `lambda_loc_plus` with on-the-fly monotonization.

    For each candidate margin, `R_loc(lambda_cnf_minus, lambda_loc)` is the mean
    over images of the monotonized loss `sup_{lambda' >= lambda_cnf_minus}
    L_loc_i(lambda', lambda_loc)`, computed by sweeping `lambda'` from 1 down to
    `lambda_cnf_minus` with a running max. The feasible set is the upper
    interval `[lambda_loc_plus, lambda_bar_loc]`; bisection narrows toward its
    left endpoint (a slight upper-approximation of `lambda_loc_plus`).
    """
    n = len(predictions)
    pf = cfg.prefilter
    candidates, image_index = _sweep_candidates(predictions)

    def monotonized_risk(lambda_loc: float) -> float:
        l_loc = [
            l_loc_image(predictions[i], ground_truth[i], 1.0, lambda_loc, pf,
                        cfg.tau_pix, cfg.margin_mode)
            for i in range(n)
        ]
        for c, i in zip(candidates, image_index):
            lambda_prime = 1.0 - c
            value = l_loc_image(predictions[i], ground_truth[i], lambda_prime,
                                lambda_loc, pf, cfg.tau_pix, cfg.margin_mode)
            if value > l_loc[i]:
                l_loc[i] = value
            if lambda_prime <= lambda_cnf_minus:   # sup only over lambda' >= cnf_minus
                break
        return sum(l_loc) / n

    lo, hi = 0.0, cfg.lambda_bar_loc
    result: float | None = None
    for _ in range(cfg.bisection_steps):
        lambda_loc = (lo + hi) / 2
        risk = monotonized_risk(lambda_loc)
        if crc_finite_sample_correction(risk, n, loss_bound) <= cfg.alpha_loc:
            result = lambda_loc
            hi = lambda_loc                        # constraint holds -> try smaller
        else:
            lo = lambda_loc                        # constraint violated -> larger
    if result is None:
        raise RuntimeError(
            "No feasible localization margin in "
            f"[0, {cfg.lambda_bar_loc}]: even lambda_bar_loc fails the bound at "
            f"alpha_loc = {cfg.alpha_loc}. Raise lambda_bar_loc or alpha_loc."
        )
    return result


# ── Top-level orchestration (spec Section 7.1) ────────────────────────────────

@dataclass
class CalibrationResult:
    """Calibrated SeqCRC parameters and the diagnostics behind them.

    Only `lambda_cnf_plus` and `lambda_loc_plus` are used at inference;
    `lambda_cnf_minus` is reported for transparency (it gates Step 2). The
    risks are the empirical calibration-set risks at the calibrated values.
    """

    lambda_cnf_plus: float
    lambda_cnf_minus: float
    lambda_loc_plus: float
    n: int
    risk_cnf: float          # R_cnf(lambda_cnf_plus) on the calibration split
    risk_loc: float          # R_loc(lambda_cnf_plus, lambda_loc_plus), same split


def calibrate(
    predictor: PredictionFunction,
    loader: DataLoader,
    cfg: SeqCRCConfig,
) -> CalibrationResult:
    """Run the full two-step SeqCRC calibration; return the calibrated params.

    Validates the spec's `alpha_loc >= alpha_cnf + 1/(n+1)` relationship against
    the realized calibration size before searching, and fails loudly otherwise.
    """
    predictions, ground_truth = collect_predictions(predictor, loader, cfg.prefilter)
    n = len(predictions)
    if n == 0:
        raise RuntimeError("Calibration set is empty.")
    cfg.validate(n)

    lambda_cnf_plus = calibrate_confidence(predictions, ground_truth, cfg, loss_bound=1.0)
    lambda_cnf_minus = calibrate_confidence(predictions, ground_truth, cfg, loss_bound=0.0)
    lambda_loc_plus = calibrate_localization(
        predictions, ground_truth, cfg, lambda_cnf_minus)

    risk_cnf = confidence_risk(predictions, ground_truth, lambda_cnf_plus, cfg)
    risk_loc = localization_risk(
        predictions, ground_truth, lambda_cnf_plus, lambda_loc_plus, cfg)

    return CalibrationResult(
        lambda_cnf_plus=lambda_cnf_plus,
        lambda_cnf_minus=lambda_cnf_minus,
        lambda_loc_plus=lambda_loc_plus,
        n=n,
        risk_cnf=risk_cnf,
        risk_loc=risk_loc,
    )


# ── Empirical risks (for evaluation / reporting) ──────────────────────────────

def confidence_risk(
    predictions: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    lambda_cnf: float,
    cfg: SeqCRCConfig,
) -> float:
    """`R_cnf(lambda_cnf)` — mean per-image FNR over the sample."""
    n = len(predictions)
    if n == 0:
        return 0.0
    return sum(
        l_cnf_image(predictions[i], ground_truth[i], lambda_cnf, cfg.prefilter)
        for i in range(n)
    ) / n


def localization_risk(
    predictions: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    lambda_cnf: float,
    lambda_loc: float,
    cfg: SeqCRCConfig,
) -> float:
    """`R_loc(lambda_cnf, lambda_loc)` — mean per-image localization loss."""
    n = len(predictions)
    if n == 0:
        return 0.0
    return sum(
        l_loc_image(predictions[i], ground_truth[i], lambda_cnf, lambda_loc,
                    cfg.prefilter, cfg.tau_pix, cfg.margin_mode)
        for i in range(n)
    ) / n
