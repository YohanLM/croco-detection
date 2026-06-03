# Calibration Pipeline

This document describes the conformal calibration pipeline used to produce certified uncertainty quantification for crocodile-clip detection.

---

## Overview

The pipeline wraps a YOLOv11n detector with **Conformal Risk Control (CRC)**: a statistical framework that finds a threshold parameter λ̂ such that the expected per-image loss on unseen data is provably bounded by a user-specified risk budget α.

The core guarantee is:

```
E[L(λ̂)] ≤ α
```

where L is a per-image loss function and the expectation is over new test images drawn from the same distribution as the calibration set.

---

## Input Data

### Images

Orthogonal LiDAR height-map projections of a railway track, resolution **640 × 640 pixels**. Each image contains one or more crocodile clips — narrow, elongated fasteners roughly 65–75 px wide × 4–5 px tall.

### Labels

Stored in YOLO format (one `.txt` per image):

```
<class> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

Coordinates are **normalized to [0, 1]** relative to image width and height.

### Dataset Splits

Plain text files listing one image path per line:

- `data/splits/calibration.txt` — used to find λ̂
- `data/splits/test.txt` — used to evaluate the calibrated system

---

## Coordinate System

All operations downstream of data loading use **pixel-space xyxy** coordinates:

| Stage | Format | Coordinates |
|-------|--------|-------------|
| YOLO labels on disk | `cls cx cy w h` | Normalized [0, 1] |
| `CalibrationDataset` output | `[G, 4]` tensor | Pixel xyxy |
| YOLO predictor output | `[P, 5]` tensor | Pixel xyxy + confidence |
| Expansion function output | `[P, 5]` tensor | Pixel xyxy + confidence |
| Loss function input | `[G, 4]` and `[P, 5]` | Pixel xyxy |

**xyxy convention:** `[x_min, y_min, x_max, y_max]`, origin at top-left, x increases right, y increases down.

The conversion from YOLO normalized to pixel xyxy is:

```python
x_min = (cx_norm - w_norm / 2) * img_width
y_min = (cy_norm - h_norm / 2) * img_height
x_max = (cx_norm + w_norm / 2) * img_width
y_max = (cy_norm + h_norm / 2) * img_height
```

There are **no real-world metric coordinates** in this pipeline; everything stays in pixel space.

---

## Batch Processing

### During Calibration and Evaluation

A PyTorch `DataLoader` yields batches of `batch_size=16` images. Because each image has a different number of ground-truth boxes, the collate function returns variable-length ground-truth tensors as a Python list rather than a stacked tensor:

```python
paths: list[str]          # length = batch_size
gts:   list[Tensor[G, 4]] # G varies per image
```

**Prediction caching** is a key efficiency decision: the detector runs once over the entire dataset (O(N) forward passes), and the resulting raw predictions are cached in memory. During the Brent root search — which may probe 30–50 candidate λ values — only the cheap expansion step is re-applied, not the detector. This makes calibration cost **O(N) model calls + O(N × M_brent) expansions** instead of O(N × M_brent) model calls.

### During Inference (Single Image)

At deployment time the pipeline processes one image at a time:

1. Run the detector to get raw boxes `[P, 5]`.
2. Apply the expansion function at the calibrated λ̂.
3. Return the certified prediction set.

---

## Pipeline Steps

### Step 1 — Data Loading

`CalibrationDataset` reads each image path and its label file, converts YOLO-normalized labels to pixel-xyxy tensors, and yields `(image_path, gt_xyxy_pixels)` pairs. The `DataLoader` batches these into `(list[path], list[Tensor])`.

### Step 2 — Model Setup

```
YoloPredictor(weights_path)
    ↓  (optional)
TopKPredictor(predictor, k=1)   # single-object regime: keep only top-1 box
```

`YoloPredictor` wraps Ultralytics YOLO and returns raw predictions as `[P, 5]` (pixel xyxy + confidence). `TopKPredictor` filters to the k highest-confidence boxes before the expansion step.

### Step 3 — Calibration (Finding λ̂)

`Calibrator.calibrate(calib_loader, lambda_range)` runs Brent's root-finding method on the function:

```
gap(λ) = CRC_bound(risk(λ), n) − α
```

where the finite-sample CRC bound is:

```
CRC_bound(R̂_n, n) = (n / (n+1)) × R̂_n + 1/(n+1)
```

This correction ensures the guarantee holds even for small calibration sets (as n → ∞ it vanishes).

The search finds the **smallest** λ for which `gap(λ) ≤ 0`, i.e., the tightest expansion that still satisfies the risk budget.

**Inner loop per λ candidate:**

```
for each image i in calibration set:
    expanded_i = expansion_fn(preds_i, λ)
    loss_i     = loss_fn(expanded_i, gt_i)

risk(λ) = mean(loss_i)
```

Predictions `preds_i` are already cached from a single prior pass.

### Step 4 — Evaluation

`Calibrator.evaluate(test_loader, λ̂)` runs the same loss computation on the held-out test set. It optionally sweeps additional λ values to produce a risk curve `R(λ)` for plotting, and collects efficiency metrics (predicted box area or box count).

### Step 5 — Reporting

Results are written to `results.txt` and a set of PNG plots:

- `risk_curve.png` — R(λ) vs α with λ̂ marked
- `loss_histogram.png` — per-image loss distribution
- `efficiency_cost.png` — expansion cost at baseline vs λ̂
- `calib_vs_test.png` — calibration / test risk generalization gap
- Example overlays — side-by-side raw vs expanded predictions on the 6 most-improved images

---

## Expansion Functions

Each expansion function takes `(preds: [P, 5], λ: float, conf_threshold: float)` and returns `[P, 5]` with modified box coordinates. All are monotone non-decreasing in λ (required for Brent's method to work).

| Name | Effect |
|------|--------|
| **Multiplicative** | Grows each box by `w·λ` / `h·λ` on all sides |
| **Additive** | Adds λ pixels to all four sides (fixed margin) |
| **Asymmetric multiplicative** | Vertical: `h·λ`, Horizontal: `w·λ/3` (reduces lateral spread) |
| **Confidence filter** | Lowers admission threshold to `max(conf_floor, 1 − λ)`; admits more boxes rather than growing them |

---

## Loss Functions

Each loss function maps `(expanded_preds: [P, 5], gt: [G, 4])` → scalar ∈ [0, 1]. All are monotone non-increasing in λ.

| Name | Formula | Meaning |
|------|---------|---------|
| **Pixel recall** | `1 − mean_k(coverage_k)` | Fraction of GT pixels not covered by any prediction |
| **Coverage indicator** | Fraction of GT boxes with < 75% pixel coverage | Binary "found / not found" per clip |
| **Detection miss** | Fraction of GT boxes hit by no prediction | Did the detector fire near the target at all? |

**Pixel coverage** (used by pixel-recall and coverage-indicator losses) is computed via local-mask rasterization: for each GT box, a binary mask of its dimensions is allocated, each predicted box is clipped to the GT region and OR-ed into the mask, then the fraction of True pixels gives the coverage. This is exact at pixel resolution without inclusion-exclusion.

---

## SeqCRC: Two-Phase Pipeline

For harder configurations (very low α, thin clips) a two-phase sequential composition is available in `conformal/seqcrc.py` and `scripts/calibrate_seqcrc.py`.

The global budget α is split between two phases:

```
α_cnf = 0.35 × α    (Phase 1: detection)
α_loc = 0.65 × α    (Phase 2: localization)
```

By a union bound, if each phase independently satisfies its sub-budget, the composed pipeline satisfies the full budget α.

### Phase 1 — Confidence Calibration

- **Loss:** detection-miss loss (did YOLO fire near the target?)
- **Expansion:** confidence filter (lower admission threshold)
- **Output:** λ_cnf → effective threshold `T_eff = max(conf_floor, 1 − λ_cnf)`

### Phase 2 — Localization Calibration

- **Dataset:** only the "survivor" frames where Phase 1 found the target at T_eff
- **Loss:** 75%-coverage indicator
- **Expansion:** additive margin (λ_loc pixels per side)
- **Output:** λ_loc

### Deployment

```
image → YOLO at T_eff → additive_expansion(λ_loc) → certified prediction set
```

The `SeqCRCInferencer` object bundles the predictor, T_eff, and λ_loc for inference.

---

## Key Parameters

| Parameter | Typical value | Meaning |
|-----------|---------------|---------|
| `alpha` | 0.09 | Global risk budget |
| `confidence_threshold` | 0.30 (geometric), 0.001 (filter) | Detector admission floor |
| `lambda_range` | (0.0, 2.0) multiplicative; (0.0, 100.0) additive | Brent search interval |
| `batch_size` | 16 | DataLoader batch size |
| `num_workers` | 4 | DataLoader parallel workers |
| SeqCRC `ALPHA_CNF_FRACTION` | 0.35 | Phase-1 share of risk budget |
| SeqCRC `CONF_FLOOR` | 0.001 | Minimum YOLO confidence threshold |
| SeqCRC `TOP_K` | 1 | Boxes kept per frame |
