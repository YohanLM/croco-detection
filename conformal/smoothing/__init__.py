"""Median-smoothing add-on for the croco-clip CRC stack.

A self-contained package that wraps the base detector in a randomized-smoothing
layer (`SmoothedTop1Predictor`) and characterizes it. It touches nothing in the
core framework: the wrapper satisfies the `PredictionFunction` contract, so it
drops into the existing `Calibrator` later with a one-line swap.

  - `noise`       : pluggable input perturbations (Gaussian is the certified one).
  - `predictor`   : `collect_samples` (one MC pass -> full sample record) and the
                    `SmoothedTop1Predictor` drop-in.
  - `certificate` : median-smoothing certified box bands + IoU radius.
  - `metrics`     : per-image stability / accuracy / MC-quality / robustness
                    metrics and the sigma `sweep`.
  - `attack`      : l2-PGD for the empirical robustness check.
"""

from conformal.smoothing.attack import pgd_l2
from conformal.smoothing.certificate import (
    CertifiedPrediction,
    certified_detection_radius,
    certified_iou_lower_bound,
    certified_radius_px,
    certify_samples,
    coordinate_certificate,
    max_certified_radius,
)
from conformal.smoothing.metrics import (
    box_jitter,
    center_error,
    coordinate_dispersion,
    coordinate_error,
    coverage_indicator,
    detection_rate,
    evaluate_image,
    mc_se_vs_n,
    mc_standard_error,
    median_repeatability,
    score_dispersion,
    self_consistency_iou,
    size_error,
    smoothed_iou,
    sweep,
)
from conformal.smoothing.noise import (
    NoiseFunction,
    gaussian_noise,
    impulse_noise,
    uniform_noise,
)
from conformal.smoothing.predictor import (
    ArrayPredictor,
    SmoothedTop1Predictor,
    SmoothingSamples,
    collect_samples,
    collect_samples_tensor,
    load_image_chw01,
)

__all__ = [
    # noise
    "NoiseFunction",
    "gaussian_noise",
    "uniform_noise",
    "impulse_noise",
    # predictor
    "ArrayPredictor",
    "SmoothedTop1Predictor",
    "SmoothingSamples",
    "collect_samples",
    "collect_samples_tensor",
    "load_image_chw01",
    # certificate
    "coordinate_certificate",
    "certified_iou_lower_bound",
    "max_certified_radius",
    "certified_detection_radius",
    "certified_radius_px",
    "CertifiedPrediction",
    "certify_samples",
    # metrics
    "detection_rate",
    "coordinate_dispersion",
    "box_jitter",
    "score_dispersion",
    "self_consistency_iou",
    "smoothed_iou",
    "coordinate_error",
    "center_error",
    "size_error",
    "coverage_indicator",
    "mc_standard_error",
    "median_repeatability",
    "evaluate_image",
    "sweep",
    "mc_se_vs_n",
    # attack
    "pgd_l2",
]
