# 1-Lipschitz HKR clip classifier

A standalone, image-level binary classifier that answers a single question:
**is there a crocodile clip in this image or not?** Since there is at most one clip per
image, this is a clean binary task. It is a port of deel-torchlip's Wasserstein/HKR
classifier (`robustness/wasserstein_classification_MNIST08.ipynb`, originally MNIST
0-vs-8) to the croco-clip domain.

It does **not** touch the YOLO / CRC / SeqCRC detection pipelines — it is additive and
localises nothing. It answers presence only.

## Why this framework: certified margins for free

The network is built **1-Lipschitz** (constant `L = 1`) w.r.t. the input L2 norm, using
deel-torchlip's spectrally-normalised layers. The decision is `sign(f(x))`, and the
signed output is a **certified margin**:

> if `y · f(x) = m > 0`, then no perturbation `δ` with `‖δ‖₂ < m` can change the
> prediction — because `|f(x+δ) − f(x)| ≤ ‖δ‖₂`.

So `|f(x)|` *is* a provable L2 robustness radius, computed in a single forward pass —
no Monte-Carlo. Images are in `[0, 1]` RGB (same as `conformal/smoothing/predictor.py`),
so this radius is in the **same pixel units** as the smoothing `sigma`. This is the
deterministic, image-level counterpart to the randomized-smoothing certificate in
`conformal/smoothing`.

## Layout

```
lipschitz/
  data.py     ClipClassificationDataset, loaders, balanced synthetic builder
  model.py    build_lip_classifier (SpectralConv2d → GroupSort2 → ScaledL2NormPool2d
              ×5, adaptive pool, FrobeniusLinear head) + checkpoint / vanilla-export
  metrics.py  binary_accuracy, certified_radius, certified_accuracy_curve, confusion
  engine.py   HKR train loop + evaluate
scripts/lipschitz/
  train_clip_classifier.py   generate balanced data → train → save weights/curves
  eval_clip_classifier.py    eval on data/splits/test.txt → certified-accuracy curve
```

Labels are derived from the existing YOLO label files (reusing
`conformal.dataset._parse_yolo_label`): a non-empty label file → `+1` (clip),
empty/missing → `−1` (no clip).

## Data

- **Train/val**: balanced synthetic generated on the fly via the project's square
  generator (`data_generation/dataset_synthetic_square.py`) with `p_clip ≈ 0.5`,
  written under `data/dataset/lipschitz/`. Reproducible by seed (train and val use
  disjoint seeds).
- **Test**: `data/splits/test.txt` (800 imgs, 235 clip / 565 none) — the same split
  the YOLO detector is measured on, for an apples-to-apples image-level comparison.

## Running (GPU machine only)

The dev box has no ML deps; training/eval run on the server.

```bash
source .venv/bin/activate
pip install deel-torchlip          # pin a torch-compatible version

# Train (≈ MNIST analogue reached ~99% val acc)
CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/train_clip_classifier.py \
    --epochs 30 --out outputs/lipschitz/run1

# Evaluate on the YOLO test split
CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/eval_clip_classifier.py \
    --run outputs/lipschitz/run1
```

Training writes `best.pt`, `vanilla.pt` (parametrisation-folded for fast inference),
`config.json`, `curves.png`, `results.txt`. Evaluation writes `eval_results.txt`
(accuracy, TP/FP/FN/TN, mean/median certified radius) and `cert_curve.png` (certified
accuracy vs radius).

## Notes

- **Lipschitz sanity check** (notebook §4.1): empirical constant should be ≈ 1 via
  `deel.torchlip.utils.evaluate_lip_const(model, x)`.
- **Architecture is tolerant to torchlip version drift**: `model._first_available`
  resolves each layer role (activation / pool) against the installed release; every
  fallback is itself 1-Lipschitz, so the global guarantee holds either way.
- **HKR loss** `HKRLoss(alpha, min_margin)` trades the Kantorovich-Rubinstein term
  (maximise margin) against the hinge term (enforce `min_margin`); `alpha→1` favours
  the hinge. Defaults `alpha=0.98`, `min_margin=1` match the reference notebook.
