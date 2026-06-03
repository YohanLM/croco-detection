"""Median-smoothing robustness certificate (Chiang et al., 2020).

This is the formal version of "could this prediction change under a small input
change?" Median smoothing certifies that the *smoothed* box can only move within
a computed band when the input is perturbed by a bounded l2 amount.

Theory. For percentile `p`, the smoothed predictor `g_p(x)` is the `p`-th
percentile of `f(x + delta)` with `delta ~ N(0, sigma^2 I)`. Under an l2
perturbation of size `epsilon`, the attainable percentile shifts by at most a
Gaussian amount, giving, per coordinate, the sandwich

    g_{p-}(x)  <=  g_p(x + e)  <=  g_{p+}(x),
    p- = Phi(Phi^{-1}(p) - epsilon/sigma),
    p+ = Phi(Phi^{-1}(p) + epsilon/sigma).

For the median (`p = 0.5`): `p- = Phi(-epsilon/sigma)`, `p+ = Phi(epsilon/sigma)`.
We estimate those percentiles from the `N` Monte-Carlo samples via order
statistics. With `conf > 0` a Gaussian-binomial rank shift turns the empirical
percentile into a high-probability bound (a one-sided nonparametric tolerance
interval), so the band holds with probability `>= 1 - conf` over the sampling.

What you get per box edge is a certified interval `[lower, upper]`; the box is
guaranteed to stay inside that band for **any** attack with `||e||_2 <= epsilon`.
From the band we derive a guaranteed worst-case IoU floor against the GT and the
largest radius that keeps that floor above a target — a per-image stability
score with a guarantee attached.

All inputs/outputs are pixel-space, matching the rest of the pipeline. `epsilon`
is in the same normalized `[0, 1]` pixel-value units as `sigma` (see `noise.py`),
because that is the space the Gaussian noise lives in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.stats import norm

from conformal.smoothing.predictor import SmoothingSamples


def _quantile_index(m: int, p: float, conf: float, side: str) -> int:
    """Order-statistic index into `m` sorted values for percentile `p`.

    With `conf == 0` this is the plain empirical-percentile rank `round(p*(m-1))`.
    With `conf > 0` the rank is shifted outward by `z * sqrt(m p (1-p))` (Gaussian
    approximation to Binomial(m, p)), so the chosen order statistic bounds the
    true percentile with probability `>= 1 - conf`:

      - `side="lower"` shifts the rank *down* (a smaller value) — a valid lower
        bound on the true `p`-quantile,
      - `side="upper"` shifts the rank *up* — a valid upper bound.
    """
    base = p * (m - 1)
    if conf > 0.0 and 0.0 < p < 1.0:
        z = float(norm.ppf(1.0 - conf))
        margin = z * math.sqrt(m * p * (1.0 - p))
        base = base - margin if side == "lower" else base + margin
    return int(min(max(round(base), 0), m - 1))


def coordinate_certificate(
    samples: SmoothingSamples,
    epsilon: float,
    *,
    conf: float = 0.0,
) -> torch.Tensor:
    """Certified `[lower, upper]` band per box edge under `||e||_2 <= epsilon`.

    Returns a `[4, 2]` tensor (x_min, y_min, x_max, y_max; columns lower/upper).
    The smoothed box at any input within `epsilon` is guaranteed (w.p. `>=1-conf`)
    to have each edge inside its row's interval.

    Returns all-NaN if there is no detection to certify (`epsilon` past which the
    box may vanish is a separate, detection-rate question).
    """
    sigma = samples.sigma
    det = samples.detected_coords                    # [m, 4]
    m = det.shape[0]
    band = torch.full((4, 2), float("nan"))
    if m == 0 or sigma <= 0.0:
        return band

    p_lower = float(norm.cdf(-epsilon / sigma))      # = Phi(-eps/sigma)
    p_upper = float(norm.cdf(epsilon / sigma))       # = Phi(+eps/sigma)
    k_lower = _quantile_index(m, p_lower, conf, "lower")
    k_upper = _quantile_index(m, p_upper, conf, "upper")
    for j in range(4):
        col, _ = torch.sort(det[:, j])
        band[j, 0] = col[k_lower]
        band[j, 1] = col[k_upper]
    return band


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of two single boxes `[4]` (pixel xyxy); 0 if either is degenerate."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def certified_iou_lower_bound(
    samples: SmoothingSamples,
    ground_truth: torch.Tensor,
    epsilon: float,
    *,
    conf: float = 0.0,
) -> float:
    """Guaranteed worst-case IoU of the smoothed box vs GT under `||e||_2<=epsilon`.

    A conservative but valid lower bound: it pairs the *smallest* box the band
    allows (minimizing intersection) with the *largest* box it allows (maximizing
    union), so the result is `<= IoU` for every box inside the certified band.

    GT is `[G, 4]` (or `[4]`); for multiple GTs the best-matching GT is used (the
    box is certified against whichever target it is closest to). Returns `0.0`
    when nothing can be certified (no detection, or `sigma == 0`).
    """
    band = coordinate_certificate(samples, epsilon, conf=conf)
    if torch.isnan(band).any():
        return 0.0
    if ground_truth.numel() == 0:
        return 0.0
    gts = ground_truth.reshape(-1, ground_truth.shape[-1])

    lo = band[:, 0]
    hi = band[:, 1]
    # Smallest admissible box (inner) minimizes intersection; largest (outer)
    # maximizes union. Clamp so the inner box stays non-degenerate.
    inner = torch.tensor([hi[0], hi[1], lo[2], lo[3]])
    inner[2] = max(float(inner[2]), float(inner[0]))
    inner[3] = max(float(inner[3]), float(inner[1]))
    outer = torch.tensor([lo[0], lo[1], hi[2], hi[3]])
    outer_area = max(0.0, float(outer[2] - outer[0])) * max(0.0, float(outer[3] - outer[1]))

    best = 0.0
    for g in gts:
        gx1 = max(float(inner[0]), float(g[0]))
        gy1 = max(float(inner[1]), float(g[1]))
        gx2 = min(float(inner[2]), float(g[2]))
        gy2 = min(float(inner[3]), float(g[3]))
        min_inter = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
        g_area = max(0.0, float(g[2] - g[0])) * max(0.0, float(g[3] - g[1]))
        max_union = outer_area + g_area - min_inter
        iou_lb = min_inter / max_union if max_union > 0 else 0.0
        best = max(best, iou_lb)
    return best


# ── GT-free, per-output certification (attaches to a live prediction) ────────
#
# Everything above can be evaluated at a chosen radius; everything below answers
# "what radius is this single output certified to?" WITHOUT ground truth, so it
# can ride along with each prediction at inference time.

def certified_detection_radius(
    samples: SmoothingSamples, *, conf: float = 0.0
) -> float:
    """Certified l2 radius over which the *detection itself* provably persists.

    The detect / no-detect vote is a binary smoothed classifier, so the standard
    randomized-smoothing classification certificate applies: if a fraction
    `p > 1/2` of the `N` noisy copies detect the clip, the smoothed detection
    decision is certified for any `||e||_2 <= sigma * Phi^{-1}(p)`. With
    `conf > 0`, `p` is replaced by a one-sided lower confidence bound (normal
    approximation) so the radius holds with probability `>= 1 - conf`.

    Returns `0.0` when the vote is at or below the quorum boundary (`p <= 1/2`)
    or when there is no smoothing noise — i.e. existence is not certifiable.
    """
    sigma = samples.sigma
    if sigma <= 0.0 or samples.n == 0:
        return 0.0
    p = samples.n_detected / samples.n
    if conf > 0.0:
        z = float(norm.ppf(1.0 - conf))
        p = p - z * math.sqrt(max(p * (1.0 - p), 0.0) / samples.n)
    if p <= 0.5:
        return 0.0
    return sigma * float(norm.ppf(min(p, 1.0 - 1e-9)))


def _max_band_halfwidth(
    samples: SmoothingSamples, epsilon: float, conf: float
) -> float:
    """Largest per-edge half-width of the certified band at `epsilon` (px)."""
    band = coordinate_certificate(samples, epsilon, conf=conf)
    if torch.isnan(band).any():
        return float("inf")
    return float(((band[:, 1] - band[:, 0]) / 2.0).max())


def certified_radius_px(
    samples: SmoothingSamples,
    tol_px: float = 2.0,
    *,
    conf: float = 0.0,
    eps_max: float = 0.5,
    tol: float = 1e-3,
) -> float:
    """GT-free localization radius: largest `epsilon` keeping the box within `tol_px`.

    "Within `tol_px`" means every certified edge band has half-width `<= tol_px`,
    i.e. under any `||e||_2 <= epsilon` no edge of the smoothed box can move more
    than `tol_px` pixels from its certified center. The band half-width grows
    monotonically with `epsilon`, so this bisects on `[0, eps_max]`.

    This is the per-output stability score the predictor can ship at inference
    time — no ground truth needed. Returns `0.0` if there is no smoothed box.
    """
    if samples.median.numel() == 0 or samples.sigma <= 0.0:
        return 0.0
    if _max_band_halfwidth(samples, eps_max, conf) <= tol_px:
        return eps_max
    lo, hi = 0.0, eps_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _max_band_halfwidth(samples, mid, conf) <= tol_px:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class CertifiedPrediction:
    """A smoothed prediction that carries its own robustness certificate.

    This is the per-output object: the robust decision (`box`) bundled with a
    GT-free guarantee about how far the input can be perturbed before that output
    can change. `band` is the certified edge envelope at the reported `epsilon`;
    the two radii summarize the rest as scalars.

    Attributes:
        box: `[1, 5]` smoothed box (xyxy + score) or `[0, 5]` if not detected.
        band: `[4, 2]` certified `[lower, upper]` per edge at `epsilon` (NaN if
            no detection).
        epsilon: the l2 radius `band` is certified at.
        detection_rate: the vote share (existence stability, pre-certificate).
        detection_radius: certified l2 radius over which the detection persists.
        localization_radius_px: certified l2 radius keeping every edge within
            `tol_px` of its certified position.
        tol_px: the pixel tolerance used for `localization_radius_px`.
    """

    box: torch.Tensor
    band: torch.Tensor
    epsilon: float
    detection_rate: float
    detection_radius: float
    localization_radius_px: float
    tol_px: float

    @property
    def certified(self) -> bool:
        """Whether there is a detected box to certify at all."""
        return self.box.numel() > 0


def certify_samples(
    samples: SmoothingSamples,
    *,
    epsilon: float = 0.1,
    tol_px: float = 2.0,
    conf: float = 0.0,
) -> CertifiedPrediction:
    """Assemble the full per-output certificate from one `SmoothingSamples` record.

    GT-free: every field is computable at inference time. `epsilon` sets the
    radius the explicit `band` is reported at; `tol_px` sets the tolerance for
    the scalar localization radius; `conf` (>0) makes both radii hold with
    probability `>= 1 - conf` over the Monte-Carlo sampling.
    """
    return CertifiedPrediction(
        box=samples.median,
        band=coordinate_certificate(samples, epsilon, conf=conf),
        epsilon=epsilon,
        detection_rate=(samples.n_detected / samples.n if samples.n else 0.0),
        detection_radius=certified_detection_radius(samples, conf=conf),
        localization_radius_px=certified_radius_px(samples, tol_px, conf=conf),
        tol_px=tol_px,
    )


def max_certified_radius(
    samples: SmoothingSamples,
    ground_truth: torch.Tensor,
    *,
    iou_target: float = 0.5,
    conf: float = 0.0,
    eps_max: float = 0.5,
    tol: float = 1e-3,
) -> float:
    """Largest `epsilon` keeping the certified IoU floor `>= iou_target`.

    The certified IoU lower bound is monotone non-increasing in `epsilon` (a
    wider band can only loosen the bound), so this bisects on `epsilon` in
    `[0, eps_max]`. Returns `0.0` if even `epsilon -> 0` fails the target (e.g.
    the clean smoothed box already misses), `eps_max` if the whole range passes.
    """
    if certified_iou_lower_bound(samples, ground_truth, 0.0, conf=conf) < iou_target:
        return 0.0
    if certified_iou_lower_bound(samples, ground_truth, eps_max, conf=conf) >= iou_target:
        return eps_max
    lo, hi = 0.0, eps_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if certified_iou_lower_bound(samples, ground_truth, mid, conf=conf) >= iou_target:
            lo = mid
        else:
            hi = mid
    return lo
