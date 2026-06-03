# Median Smoothing Layer

A randomized-smoothing add-on for the croco-clip detector that turns each
prediction into a **robust, certified output**: instead of trusting one YOLO
forward pass, it runs the detector on `N` Gaussian-noised copies of the image,
takes the coordinate-wise **median** of the top-1 boxes (the robust decision),
and attaches a **per-output robustness certificate** (how far the input can be
perturbed before that output can change).

It is a self-contained package, `conformal/smoothing/`, that touches nothing in
the core CRC framework — the smoothed predictor satisfies the existing
`PredictionFunction` contract, so it drops into the `Calibrator` later with a
one-line swap.

---

## 1. Why

The base detector is fragile to small input perturbations (sensor noise, lens
dirt, compression speckle, adversarial patches on orthophotos). Standard
randomized smoothing (plurality vote) is built for classification; bounding-box
regression needs a different aggregator. Taking the **mean** of box coordinates
across noisy copies is wrecked by outliers, so we use the **median**, which
votes out spurious noise-triggered boxes while staying provably stable
(Chiang et al., 2020, *Detection as Regression: Certified Object Detection by
Median Smoothing*).

Two things come out of the same `N`-sample computation:

- **The decision** — the median box (a robust point estimate).
- **The instability signal / certificate** — the dispersion and order statistics
  of the sample cloud, which say how much that box could move (or vanish) under a
  bounded input change.

The median alone would throw away the second half. The layer keeps both.

---

## 2. Architecture at a glance

```
image ─► N noisy copies ─► base YOLO (one batched pass) ─► N top-1 boxes
                                                              │
                                       ┌──────────────────────┴───────────────┐
                                       ▼                                       ▼
                            coordinate-wise median                  order statistics + vote
                            (+ detection quorum)                    (dispersion / certificate)
                                       │                                       │
                                       ▼                                       ▼
                               robust box  [1,5]                    CertifiedPrediction
                             (PredictionFunction)                  (band + radii, GT-free)
```

A single Monte-Carlo pass (`collect_samples`) produces a `SmoothingSamples`
record holding **every** copy's box plus a detection mask. The predictor returns
only the median; the metrics and the certificate read the whole record. No
duplicated inference.

---

## 3. Package layout — `conformal/smoothing/`

| Module | Contents |
|---|---|
| `noise.py` | `NoiseFunction` protocol + `gaussian_noise` (the **certified** one), `uniform_noise`, `impulse_noise`. `sigma` is in normalized `[0,1]` pixel-value units. |
| `predictor.py` | `collect_samples` / `collect_samples_tensor` (one MC pass → `SmoothingSamples`) and `SmoothedTop1Predictor` — the drop-in `PredictionFunction`. |
| `certificate.py` | The median-smoothing certificate: GT-free per-output radii + bands, and GT-coupled (eval-only) IoU bounds. |
| `metrics.py` | 14+ per-image evaluation functions across four families, plus the `sweep` / `mc_se_vs_n` aggregators. |
| `attack.py` | `pgd_l2` — l2-bounded PGD for the empirical robustness check. |

Supporting change outside the package: `YoloPredictor.predict_arrays`
(`conformal/prediction/yolo.py`) lets the base detector run on in-memory image
tensors, because noise must be injected at the pixel level (file paths are not
enough). Existing `YoloPredictor` methods, `calibrator.py`, and `seqcrc.py` are
untouched. Public names are re-exported through `conformal/__init__.py`.

---

## 4. The predictor — decision vs certificate

`SmoothedTop1Predictor` wraps any base with `predict_arrays` (e.g.
`YoloPredictor`) and exposes two deliberately separated surfaces:

| Method | Returns | Role |
|---|---|---|
| `__call__(path, conf)` / `predict_batch` | `[P,5]` box only | The robust **decision**. Same strict contract as `YoloPredictor`, so the `Calibrator` cannot tell the difference. |
| `certify(path, conf, *, epsilon, tol_px, conf)` / `certify_batch` | `CertifiedPrediction` | The box **+ its per-output certificate**. |
| `samples_for(path, conf)` | `SmoothingSamples` | The raw `N`-sample record, for custom metrics. |

Key config: `n_samples` (`N`), `noise_scale` (`sigma`), `noise_fn`, `quorum`,
`conf_floor`, `seed`.

### Detection quorum (the no-vote rule)
Some noisy copies detect nothing. If fewer than a `quorum` fraction of the `N`
copies produce a box, the smoothed prediction is **empty** (`[0,5]`) — the
median "vote" is no-detection. Otherwise the median is taken over the detecting
copies only. The base runs at a permissive `conf_floor` so weak-but-real
detections can vote; the operating `confidence_threshold` is applied to the
**median score** afterwards.

### Reproducibility
Noise is seeded **per image** (stable hash of the path), so `predict_batch` and
per-image `__call__` give identical results regardless of call order, and a
`sigma` sweep can reuse the same noise to isolate the effect of `sigma`.

---

## 5. The certificate (per output, GT-free)

A `CertifiedPrediction` bundles the robust box with guarantees that hold at
**inference time, without ground truth** — so every live output ships certified.

| Field | Meaning |
|---|---|
| `box` | `[1,5]` smoothed box (or `[0,5]`). |
| `band` `[4,2]` | Certified `[lower, upper]` per edge: under any `‖δ‖₂ ≤ epsilon`, each edge provably stays in its interval. |
| `detection_radius` | l2 radius over which the **detection itself** provably persists — the classification-smoothing certificate on the detect/no-detect vote, `σ·Φ⁻¹(p_detect)`. |
| `localization_radius_px` | l2 radius keeping **every edge within `tol_px`** of its certified position. |
| `detection_rate` | the raw vote share (existence stability, pre-certificate). |

### Theory
For percentile `p`, the smoothed predictor `g_p(x)` is the `p`-th percentile of
`f(x + δ)`, `δ ~ N(0, σ²I)`. Under an l2 perturbation of size `ε`, the attainable
percentile shifts by a bounded Gaussian amount, giving per coordinate

```
g_{p-}(x) ≤ g_p(x+e) ≤ g_{p+}(x),   p- = Φ(Φ⁻¹(p) − ε/σ),  p+ = Φ(Φ⁻¹(p) + ε/σ).
```

Those percentiles are estimated from the `N` samples via **order statistics**.
With `conf > 0`, a Gaussian-binomial rank shift turns the empirical percentile
into a high-probability bound, so the band holds with probability `≥ 1 − conf`.

### GT-free vs GT-coupled
- **GT-free (per-output, ship with each prediction):** `coordinate_certificate`,
  `certified_detection_radius`, `certified_radius_px`, assembled by
  `certify_samples` / `SmoothedTop1Predictor.certify`.
- **GT-coupled (evaluation only — needs ground truth):**
  `certified_iou_lower_bound` (worst-case IoU vs GT inside the band) and
  `max_certified_radius` (largest `ε` keeping certified IoU above a target).

> `sigma` and `epsilon` share the normalized `[0,1]` pixel-value scale. The PGD
> attack's `epsilon` is a different, coarser quantity — the **total** l2 norm
> over the whole image tensor (`epsilon_total ≈ sigma·√(3·H·W)`).

---

## 6. Evaluation metrics — `metrics.py`

All are pure functions of a `SmoothingSamples` (+ GT where needed), grouped by
question. `evaluate_image` bundles them into a flat scalar dict; `sweep` averages
that over a split for each `sigma`.

**Stability / dispersion (no GT) — the instability detector**
- `detection_rate` — vote share; the most brittle signal (box blinking in/out).
- `coordinate_dispersion` / `box_jitter` — per-edge / mean std (px) under noise.
- `score_dispersion` — std of the confidence across copies.
- `self_consistency_iou` — agreement of each copy's box with the median.

**Accuracy vs GT**
- `smoothed_iou`, `coordinate_error`, `center_error`, `size_error`.
- `coverage_indicator` — reuses the live 75%-coverage CRC loss, so the number is
  directly comparable to what calibration optimizes.

**Monte-Carlo estimation quality (choose `N`)**
- `mc_standard_error` — analytic median SE per edge (`≈ 1.2533·std/√m`).
- `median_repeatability` — empirical SE: re-estimate `R` times, measure spread.
- `mc_se_vs_n` — SE vs `N` curve (should fall like `1/√N`).

**Certified robustness**
- GT-free: `cert_detection_radius`, `cert_localization_radius_px`.
- GT-coupled: `certified_iou`, `certified_radius_vs_gt`.

---

## 7. Scripts — `scripts/smoothing/`

| Script | What it does | Output |
|---|---|---|
| `smooth_predict_demo.py` | Sanity overlays: raw top-1 vs smoothed median vs the noisy-sample cloud vs GT; prints each output's certificate; contract check. | `outputs/smoothing_demo/` |
| `evaluate_smoothing.py` | `sigma` sweep over the test split → full metric table + plots (IoU, jitter, detection rate, certified radii vs `sigma`; MC-SE vs `N`). No calibration. | `outputs/smoothing_eval/` |
| `attack_smoothing.py` | l2-PGD on each test image; compares raw vs smoothed IoU drop under attack (robustness gain); optional CRC-risk-under-attack if a calibrated `lambda` is set. | `outputs/smoothing_attack/` |

Run from the project root:

```bash
python scripts/smoothing/smooth_predict_demo.py
python scripts/smoothing/evaluate_smoothing.py     # MAX_IMAGES caps a quick run; None = full split
python scripts/smoothing/attack_smoothing.py       # PGD is slow on CPU; keep MAX_IMAGES small
```

Each script has a CONFIG block at the top (weights `models/best.pt`, splits
`data/splits/`, `sigma`, `N`, quorum, thresholds) and mirrors all console output
to a `results.txt` beside its plots.

### Expected trends
As `sigma` grows: IoU **decreases** (precision cost of noise), box jitter and the
certified radii **increase** (more robustness), detection rate eventually drops.
Under PGD: the smoothed IoU drop should be **smaller** than the raw drop.

---

## 8. Practical constraints

- **Input size.** Tensor inference requires images divisible by 32. The
  synthetic 640² test split is fine; the attack script resizes to 640.
  Rectangular configs (e.g. 570×100) would need padding.
- **Tensor normalization.** `predict_arrays` passes a `[0,1]` RGB tensor;
  Ultralytics applies `/255` only to numpy inputs, so a float tensor is taken
  as-is (and letterboxing is skipped) — boxes come back in the input pixel scale.
- **Cost.** Inference is `N×` the base detector. `N` MC samples are batched into
  one forward pass per image; pick `N` from the `mc_se_vs_n` curve.

---

## 9. Future: feeding the calibrator

Because `SmoothedTop1Predictor` satisfies `PredictionFunction` + `predict_batch`,
conformal calibration over smoothed predictions is a one-line predictor swap. The
only hook needed is to let `scripts/crc_common.run_pipeline` accept an injected
predictor instead of hardcoding `YoloPredictor` (`crc_common.py:330`). A richer
option is to let the **certificate feed back into the conformal expansion**
(e.g. expand low-certified-radius boxes more, or refuse boxes below a detection-
radius threshold) — left open by design, since it couples certification into
calibration.
```
