"""Evaluation metrics for the median-smoothing add-on.

These answer "how does smoothing behave?" from four angles, all read off the same
per-image `SmoothingSamples` record (one Monte-Carlo pass — no extra inference):

  1. Stability / dispersion (no GT) — how much the box wiggles or blinks across
     the `N` noisy copies. This is the *instability detector*: a prediction that
     moves a lot under tiny noise is one that "could change with little
     modification".
  2. Accuracy vs GT — the precision the median actually delivers.
  3. Monte-Carlo estimation quality — whether `N` is large enough that the
     numbers reflect the predictor, not sampling noise.
  4. Certified robustness — guaranteed worst-case behaviour under an l2 attack
     (delegated to `conformal.smoothing.certificate`).

Each metric is a small pure function of `(samples)` or `(samples, gt)`.
`evaluate_image` bundles them into a flat scalar dict, and `sweep` averages that
over a dataset for each `sigma` — the table the evaluation script plots.

GT format matches `CalibrationDataset` (`conformal/dataset.py`): pixel-xyxy
`[G, 4]`. NaN is used for "undefined on this image" (e.g. IoU when the smoothed
box is empty) so aggregation can skip it cleanly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from conformal.loss.coverage import image_coverage_indicator_loss
from conformal.smoothing.certificate import (
    certified_detection_radius,
    certified_iou_lower_bound,
    certified_radius_px,
    max_certified_radius,
)
from conformal.smoothing.predictor import (
    ArrayPredictor,
    SmoothedTop1Predictor,
    SmoothingSamples,
    collect_samples,
)


_NAN = float("nan")


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of two single boxes `[>=4]` (pixel xyxy); 0 if either is degenerate."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ba = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _best_gt(box: torch.Tensor, gt: torch.Tensor) -> torch.Tensor | None:
    """The GT box (row of `[G, 4]`) with highest IoU to `box`; None if no GT."""
    if gt.numel() == 0:
        return None
    gts = gt.reshape(-1, gt.shape[-1])
    ious = [_iou(box, g) for g in gts]
    return gts[int(max(range(len(ious)), key=lambda i: ious[i]))]


# ── 1. Stability / dispersion (no GT needed) ─────────────────────────────────

def detection_rate(s: SmoothingSamples) -> float:
    """Fraction of the `N` noisy copies that produced a box — the "vote share".

    The most brittle instability signal: a low rate means the detection itself
    flickers in and out under noise, not just that the box moves.
    """
    return s.n_detected / s.n if s.n else 0.0


def coordinate_dispersion(s: SmoothingSamples) -> torch.Tensor:
    """Per-edge std (px) of the top-1 box across the detecting copies — `[4]`.

    How far each box edge travels under noise. NaN per edge if `< 2` detections.
    """
    det = s.detected_coords
    if det.shape[0] < 2:
        return torch.full((4,), _NAN)
    return det.std(dim=0, unbiased=True)


def box_jitter(s: SmoothingSamples) -> float:
    """Scalar instability score: mean per-edge std (px). NaN if undefined."""
    disp = coordinate_dispersion(s)
    return float(disp.mean()) if not torch.isnan(disp).any() else _NAN


def score_dispersion(s: SmoothingSamples) -> float:
    """Std of the top-1 confidence across detecting copies. NaN if `< 2`."""
    sc = s.detected_scores
    return float(sc.std(unbiased=True)) if sc.shape[0] >= 2 else _NAN


def self_consistency_iou(s: SmoothingSamples) -> float:
    """Mean IoU of each detecting copy's box vs the median box.

    High = the sample cloud is tight and unimodal (stable); low = the detector
    is torn between interpretations (the median sits between two modes). NaN if
    there is no median box.
    """
    if s.median.numel() == 0:
        return _NAN
    med = s.median[0]
    det = s.detected_coords
    if det.shape[0] == 0:
        return _NAN
    return sum(_iou(box, med) for box in det) / det.shape[0]


# ── 2. Accuracy vs ground truth ──────────────────────────────────────────────

def smoothed_iou(s: SmoothingSamples, gt: torch.Tensor) -> float:
    """IoU of the smoothed (median) box vs the best-matching GT. NaN if no box/GT."""
    if s.median.numel() == 0 or gt.numel() == 0:
        return _NAN
    g = _best_gt(s.median[0], gt)
    return _iou(s.median[0], g) if g is not None else _NAN


def coordinate_error(s: SmoothingSamples, gt: torch.Tensor) -> torch.Tensor:
    """Per-edge `|median - GT|` (px) vs the best-matching GT — `[4]`. NaN if none."""
    if s.median.numel() == 0 or gt.numel() == 0:
        return torch.full((4,), _NAN)
    g = _best_gt(s.median[0], gt)
    if g is None:
        return torch.full((4,), _NAN)
    return (s.median[0, :4] - g[:4]).abs()


def center_error(s: SmoothingSamples, gt: torch.Tensor) -> float:
    """L2 distance (px) between the median box center and the best GT center."""
    if s.median.numel() == 0 or gt.numel() == 0:
        return _NAN
    g = _best_gt(s.median[0], gt)
    if g is None:
        return _NAN
    mc = s.median[0]
    pcx, pcy = (mc[0] + mc[2]) / 2, (mc[1] + mc[3]) / 2
    gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return float(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5)


def size_error(s: SmoothingSamples, gt: torch.Tensor) -> float:
    """Relative area error `|area_pred - area_gt| / area_gt`. NaN if no box/GT."""
    if s.median.numel() == 0 or gt.numel() == 0:
        return _NAN
    g = _best_gt(s.median[0], gt)
    if g is None:
        return _NAN
    mc = s.median[0]
    pa = max(0.0, float(mc[2] - mc[0])) * max(0.0, float(mc[3] - mc[1]))
    ga = max(0.0, float(g[2] - g[0])) * max(0.0, float(g[3] - g[1]))
    return abs(pa - ga) / ga if ga > 0 else _NAN


def coverage_indicator(s: SmoothingSamples, gt: torch.Tensor) -> float:
    """75%-coverage-indicator loss of the median box vs GT (the live CRC loss).

    Reuses `image_coverage_indicator_loss` so this number is directly comparable
    to what the conformal calibration optimizes. `1.0` when the box is empty but
    a GT exists (a full miss), `0.0` when there is no GT.
    """
    return image_coverage_indicator_loss(s.median, gt)


# ── 3. Monte-Carlo estimation quality ────────────────────────────────────────

def mc_standard_error(s: SmoothingSamples) -> torch.Tensor:
    """Standard error of the median estimate per edge (px) — `[4]`.

    Uses the large-sample median SE `~= 1.2533 * std / sqrt(m)`. Separates "the
    prediction is intrinsically unstable" (large dispersion) from "I just under-
    sampled" (large SE that more `N` would shrink). NaN per edge if `< 2`.
    """
    disp = coordinate_dispersion(s)
    m = s.n_detected
    if m < 2 or torch.isnan(disp).any():
        return torch.full((4,), _NAN)
    return 1.2533 * disp / math.sqrt(m)


def median_repeatability(
    predictor: SmoothedTop1Predictor,
    image_path: str,
    confidence_threshold: float,
    *,
    repeats: int = 5,
) -> torch.Tensor:
    """Empirical SE of the median: re-estimate `repeats` times, take the std — `[4]`.

    A direct, assumption-free check on whether `N` is adequate: re-run the whole
    smoothing pass `repeats` times (fresh noise each time) and measure how much
    the median box itself moves. Small = `N` is enough; large = raise `N`. NaN if
    fewer than 2 repeats yielded a box.

    Note: this deliberately bypasses the predictor's per-image seed by drawing
    fresh noise per repeat, so the repeats are genuinely independent.
    """
    meds = []
    for _ in range(repeats):
        # Draw genuinely independent noise per repeat (generator=None) instead of
        # the predictor's deterministic per-image seed, so the repeats are i.i.d.
        s = collect_samples(
            predictor.base, image_path, predictor.sigma, predictor.n_samples,
            noise_fn=predictor.noise_fn, conf_floor=predictor.conf_floor,
            quorum=predictor.quorum, conf_threshold=confidence_threshold,
            generator=None,
        )
        if s.median.numel():
            meds.append(s.median[0, :4])
    if len(meds) < 2:
        return torch.full((4,), _NAN)
    return torch.stack(meds).std(dim=0, unbiased=True)


# ── Per-image bundle ─────────────────────────────────────────────────────────

def evaluate_image(
    s: SmoothingSamples,
    gt: torch.Tensor,
    *,
    cert_epsilon: float = 0.1,
    iou_target: float = 0.5,
    cert_conf: float = 0.0,
) -> dict[str, float]:
    """All scalar metrics for one image, flattened for easy aggregation.

    Vector metrics are reduced to their mean (`box_jitter`, `mc_se`); the
    certified-robustness metrics are evaluated at `cert_epsilon` / `iou_target`.
    """
    mc_se = mc_standard_error(s)
    return {
        "detection_rate": detection_rate(s),
        "has_box": 1.0 if s.median.numel() else 0.0,
        "box_jitter_px": box_jitter(s),
        "score_dispersion": score_dispersion(s),
        "self_consistency_iou": self_consistency_iou(s),
        "smoothed_iou": smoothed_iou(s, gt),
        "center_error_px": center_error(s, gt),
        "size_error_rel": size_error(s, gt),
        "coverage_loss": coverage_indicator(s, gt),
        "mc_se_px": _NAN if torch.isnan(mc_se).any() else float(mc_se.mean()),
        # GT-FREE per-output certificate (what each live prediction can carry):
        "cert_detection_radius": certified_detection_radius(s, conf=cert_conf),
        "cert_localization_radius_px": certified_radius_px(s, tol_px=2.0, conf=cert_conf),
        # GT-COUPLED certificate (eval-only — needs ground truth):
        "certified_iou": certified_iou_lower_bound(s, gt, cert_epsilon, conf=cert_conf),
        "certified_radius_vs_gt": max_certified_radius(
            s, gt, iou_target=iou_target, conf=cert_conf
        ),
    }


# ── Split-level aggregation ──────────────────────────────────────────────────

def _nanmean(values: list[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else _NAN


def sweep(
    base: ArrayPredictor,
    dataset: Iterable[tuple[str, torch.Tensor]],
    sigmas: Iterable[float],
    n_samples: int,
    *,
    confidence_threshold: float = 0.30,
    quorum: float = 0.5,
    conf_floor: float = 0.05,
    seed: int | None = 0,
    cert_epsilon: float = 0.1,
    iou_target: float = 0.5,
    cert_conf: float = 0.0,
    noise_fn=None,
) -> dict[float, dict[str, float]]:
    """Mean of every per-image metric over `dataset`, for each `sigma`.

    One smoothing predictor is built per `sigma`; the dataset is iterated once
    per sigma. Returns `{sigma: {metric_name: mean_over_split}}` — the table the
    evaluation script renders and plots. `dataset` yields `(image_path, gt_xyxy)`
    exactly as `CalibrationDataset` does.
    """
    items = list(dataset)
    kwargs = {} if noise_fn is None else {"noise_fn": noise_fn}
    out: dict[float, dict[str, float]] = {}
    for sigma in sigmas:
        predictor = SmoothedTop1Predictor(
            base, n_samples, sigma, quorum=quorum, conf_floor=conf_floor,
            seed=seed, **kwargs,
        )
        rows: list[dict[str, float]] = []
        for path, gt in items:
            s = predictor.samples_for(path, confidence_threshold)
            rows.append(evaluate_image(
                s, gt, cert_epsilon=cert_epsilon, iou_target=iou_target,
                cert_conf=cert_conf,
            ))
        keys = rows[0].keys() if rows else []
        out[float(sigma)] = {k: _nanmean([r[k] for r in rows]) for k in keys}
    return out


def mc_se_vs_n(
    base: ArrayPredictor,
    dataset: Iterable[tuple[str, torch.Tensor]],
    sigma: float,
    n_values: Iterable[int],
    *,
    confidence_threshold: float = 0.30,
    quorum: float = 0.5,
    conf_floor: float = 0.05,
    seed: int | None = 0,
) -> dict[int, float]:
    """Mean Monte-Carlo SE (px) of the median vs `N`, at a fixed `sigma`.

    The diagnostic for choosing `N`: SE should fall like `1/sqrt(N)`; pick the
    `N` where it drops below the precision you care about. Returns `{N: mean_se}`.
    """
    items = list(dataset)
    out: dict[int, float] = {}
    for n in n_values:
        predictor = SmoothedTop1Predictor(
            base, int(n), sigma, quorum=quorum, conf_floor=conf_floor, seed=seed,
        )
        ses: list[float] = []
        for path, _gt in items:
            s = predictor.samples_for(path, confidence_threshold)
            se = mc_standard_error(s)
            if not torch.isnan(se).any():
                ses.append(float(se.mean()))
        out[int(n)] = _nanmean(ses)
    return out
