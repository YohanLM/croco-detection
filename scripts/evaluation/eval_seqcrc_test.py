"""Image-level TP / FP / FN / TN for the calibrated SeqCRC pipeline.

Lambda values are taken from the already-run experiment seqcrc_09_top1_3
(outputs/seqcrc/seqcrc_09_top1_3/results.txt):

    lambda_cnf = 0.9006  ->  T_eff = max(0.001, 1 - 0.9006) = 0.0994
    lambda_loc = 0.4379 px per side (additive expansion)
    top-k      = 1

The calibrated pipeline for each image:
    1. Run YOLO at T_eff=0.0994, keep top-1 box by confidence (Phase 1)
    2. Grow that box by +0.4379 px per side (Phase 2)

Classification criterion: 75 % pixel coverage indicator (identical to the
Phase-2 loss).  A GT clip is "found" if ≥75 % of its pixel area is covered
by the union of the (expanded) predicted boxes.  This is the right criterion
here because box expansion directly increases coverage ratios — unlike IoU,
which penalises boxes that are too large.

Definitions (image level):
  TP  image has ≥1 GT clip AND every GT clip is covered ≥75 %
  FN  image has ≥1 GT clip AND at least one GT clip is covered <75 %
  FP  image has 0 GT clips AND the pipeline emits ≥1 box  (false alarm)
  TN  image has 0 GT clips AND the pipeline emits 0 boxes

Run from the croco_detection project root:
    cd /home/data/home/lemorhedec-y/croco_detection
    CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/eval_seqcrc_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from conformal.dataset import CalibrationDataset
from conformal.prediction.yolo import YoloPredictor
from conformal.prediction.top1 import TopKPredictor
from conformal.seqcrc import SeqCRCInferencer
from conformal.loss.pixel import _gt_pixel_area, _union_pixel_area_inside

WEIGHTS  = ROOT / "models" / "best.pt"
TEST_TXT = ROOT / "data" / "splits" / "test.txt"
DEVICE   = "cuda"

# --- calibrated values from seqcrc_09_top1_3 ---
T_EFF      = 0.0994   # max(0.001, 1 - 0.9006)
LAMBDA_LOC = 0.4379   # px per side (additive expansion)

COVERAGE_THRESHOLD = 0.75


def per_gt_coverages(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    """Fraction of each GT box's pixels covered by the union of predicted boxes."""
    ratios = []
    for k in range(gt.shape[0]):
        area = _gt_pixel_area(gt[k])
        if area == 0:
            continue
        covered = _union_pixel_area_inside(gt[k], pred)
        ratios.append(covered / area)
    return ratios


def classify(pred: torch.Tensor, gt: torch.Tensor) -> tuple[str, list[float]]:
    """Return (label, [coverage_ratio_per_GT])."""
    has_gt   = gt.numel() > 0
    has_pred = pred.numel() > 0

    if has_gt:
        ratios = per_gt_coverages(pred, gt)
        if not ratios:
            return "TN", []
        found = all(r >= COVERAGE_THRESHOLD for r in ratios)
        return ("TP" if found else "FN"), ratios
    return ("FP" if has_pred else "TN"), []


def main() -> None:
    base     = YoloPredictor(str(WEIGHTS))
    base.model.to(DEVICE)
    top1     = TopKPredictor(base, k=1)
    pipeline = SeqCRCInferencer(predictor=top1, t_eff=T_EFF, lambda_loc=LAMBDA_LOC)

    dataset  = CalibrationDataset(TEST_TXT)
    counts   = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    examples: dict[str, list[str]] = {"TP": [], "FP": [], "FN": [], "TN": []}
    coverages: dict[str, list[float]] = {"TP": [], "FN": []}

    for i, (img_path, gt) in enumerate(dataset):
        pred  = pipeline(img_path)
        label, ratios = classify(pred, gt)
        counts[label] += 1
        if label in coverages:
            coverages[label].extend(ratios)
        if len(examples[label]) < 3:
            examples[label].append(Path(img_path).name)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dataset)} ...", flush=True)

    n  = len(dataset)
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]
    n_pos = tp + fn
    n_neg = fp + tn

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else float("nan"))

    all_cov  = coverages["TP"] + coverages["FN"]
    mean_cov_tp = sum(coverages["TP"]) / len(coverages["TP"]) if coverages["TP"] else float("nan")
    mean_cov_fn = sum(coverages["FN"]) / len(coverages["FN"]) if coverages["FN"] else float("nan")
    mean_cov    = sum(all_cov) / len(all_cov) if all_cov else float("nan")

    print(f"\n{'='*60}")
    print(f"SeqCRC pipeline  —  test split  (n={n})")
    print(f"  T_eff={T_EFF}  lambda_loc={LAMBDA_LOC} px  top-k=1")
    print(f"Coverage criterion: ≥{int(COVERAGE_THRESHOLD*100)}% of GT pixels covered")
    print(f"{'='*60}")
    print(f"  Positive images (has clip) : {n_pos}")
    print(f"  Negative images (no clip)  : {n_neg}")
    print()
    print(f"  TP  (clip present, covered ≥{int(COVERAGE_THRESHOLD*100)}%)  : {tp:>4}  /  {n_pos}")
    print(f"  FN  (clip present, covered <{int(COVERAGE_THRESHOLD*100)}%)  : {fn:>4}  /  {n_pos}")
    print(f"  FP  (no clip, false alarm)           : {fp:>4}  /  {n_neg}")
    print(f"  TN  (no clip, correctly silent)      : {tn:>4}  /  {n_neg}")
    print()
    print(f"  Image-level precision      : {precision:.4f}")
    print(f"  Image-level recall         : {recall:.4f}")
    print(f"  Image-level F1             : {f1:.4f}")
    print()
    print(f"  Mean GT coverage (TP imgs) : {mean_cov_tp:.4f}")
    print(f"  Mean GT coverage (FN imgs) : {mean_cov_fn:.4f}  (0.0 = no prediction)")
    print(f"  Mean GT coverage (all pos) : {mean_cov:.4f}")
    print(f"{'='*60}")
    print("Example filenames per category:")
    for cat, fnames in examples.items():
        print(f"  {cat}: {fnames}")


if __name__ == "__main__":
    main()
