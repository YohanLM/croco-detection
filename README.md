# Conformal Risk Control for croco-clip detection

A YOLO-based croco-clip detector wrapped in a **Conformal Risk Control (CRC)** layer that turns raw bounding-box predictions into certified prediction sets. The expected per-image loss is provably bounded by a user-chosen risk level α, with no distributional assumptions beyond exchangeability.

## What it does

Given a trained YOLO model and a calibration set of labelled images, the pipeline calibrates a scalar margin λ̂ such that:

```
E[L(λ̂)] ≤ α
```

where L is a per-image loss (pixel-wise recall or 75%-coverage indicator). At deployment the model runs at λ̂, producing enlarged prediction sets that carry the guarantee.

Two pipelines are implemented:

| Pipeline | Knob | Recovers |
|---|---|---|
| **Single-knob CRC** | geometric expansion (multiplicative / additive / asymmetric) or confidence threshold | localisation slack or missed detections — not both |
| **SeqCRC** | Phase 1: confidence threshold → Phase 2: additive expansion | both, under a Bonferroni budget split α = α_cnf + α_loc |

Best result: SeqCRC at α=0.09 achieves e2e test risk **0.039**, reducing fully-missed clips from 94/800 → 31/800 at 1.19× box inflation.

## Repository layout

```
conformal/          core library
  calibrator.py     Calibrator engine + EvaluationResult
  seqcrc.py         SeqCRC two-phase composer + SeqCRCInferencer
  dataset.py        GT data loading (image paths + YOLO labels → pixel xyxy)
  prediction/       YOLO wrapper, TopKPredictor
  expansion/        multiplicative, additive, asymmetric, confidence-filter
  loss/             pixel recall, 75%-coverage indicator, detection-miss
  efficiency/       box area, box count
  diagnostics/      TP/FP/empty-highlight counters

scripts/
  calibration/      one script per experiment (calibrate_crc*.py, calibrate_seqcrc.py)
  evaluation/       val_crc_test.py, eval_test_split.py, eval_seqcrc_test.py,
                    check_disjointness.py, generate_clean_splits.py

outputs/            experiment results (results.txt + diagnostic PNGs per run)
  experiments_summary.txt   all runs compared in one file
  base_results/     YOLO training output (results.csv, curves, weights)
  crc/              single-knob CRC runs
  seqcrc/           SeqCRC runs
  raw_vs_seqcrc/    image-level TP/FP/FN/TN comparison

docs/report/        conformal_report.tex  (LaTeX report of all experiments)
models/             best.pt (trained YOLO11n weights)
data/splits/        calibration.txt, test.txt  (800 images each, disjoint from training)
```

## Environment

Training and calibration run on a remote GPU machine. This Windows repo is for authoring and analysis only.

```
GPU machine:  /home/data/home/lemorhedec-y/croco_detection/
Model:        models/best.pt   (YOLO11n, 25 epochs, imgsz=640)
```

Activate the venv before running any script on the server:
```bash
source .venv/bin/activate
```

## Data splits

| Split | File | Size | Notes |
|---|---|---|---|
| Calibration | `data/splits/calibration.txt` | 800 | disjoint from train and val |
| Test | `data/splits/test.txt` | 800 | 235 positive + 565 negative |
| Training val | `dataset_800.yaml` val key | 800 | used to select `best.pt` — see disjointness check |

To verify disjointness before trusting CRC results:
```bash
python scripts/evaluation/check_disjointness.py
```

## Running experiments

All scripts run from the project root on the GPU machine with `CUDA_VISIBLE_DEVICES=N`.

```bash
# Single-knob CRC (one script per loss/expansion combination)
python scripts/calibration/calibrate_crc.py                  # pixel loss, multiplicative
python scripts/calibration/calibrate_crc_coverage.py          # coverage, multiplicative
python scripts/calibration/calibrate_crc_additive_coverage.py # coverage, additive
python scripts/calibration/calibrate_crc_confidence.py        # coverage, confidence filter

# SeqCRC (edit alpha split inside the script)
python scripts/calibration/calibrate_seqcrc.py

# YOLO val on the CRC test split
python scripts/evaluation/val_crc_test.py

# Image-level TP/FP/FN/TN
python scripts/evaluation/eval_test_split.py      # raw YOLO
python scripts/evaluation/eval_seqcrc_test.py     # SeqCRC pipeline
```

Each run writes `outputs/<name>/results.txt` and diagnostic PNGs. A cross-run summary is at `outputs/experiments_summary.txt`.

## Key design decisions

**`B/(n+1)` correction.** The finite-sample correction uses the loss upper bound B (carried by `Risk.loss_upper_bound`), not a hardcoded 1. All current losses have B=1.

**Top-1 selection in SeqCRC.** Phase 1 lowers the confidence threshold toward 0.001 to recover missed clips. Without top-1 selection this floods frames with spurious boxes; `TopKPredictor(k=1)` keeps only the highest-confidence box so the recovered detections are the right ones.

**75%-coverage criterion.** The Phase-2 loss and image-level evaluation both use pixel coverage (fraction of GT area covered), not IoU. IoU penalises boxes that are too large; coverage does not — which is appropriate here because expansion deliberately grows boxes.

## Report

```bash
# On the server: regenerate cross-experiment figures
python docs/report/make_report_figures.py

# Compile (Overleaf or local LaTeX)
pdflatex docs/report/conformal_report.tex
```
