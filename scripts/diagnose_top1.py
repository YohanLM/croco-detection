"""Decide single-phase-top1 vs two-phase SeqCRC for a single-object detector.

The worry: lowering the confidence threshold makes many boxes appear even
though there is one object per frame. This script answers the two questions
that settle the design, over the calibration set:

  1. Is the flood real AT THE OPERATING POINT? The floor (CONF_FLOOR) is only
     the candidate pool; Phase 1 deploys at T_eff = max(floor, 1 - lambda_cnf),
     the HIGHEST threshold meeting the recall guarantee. We report boxes-per-
     frame at the floor AND at T_eff so you see the count you actually ship.

  2. Does lowering the threshold buy recall, or is top-1 enough? For each
     frame with a target we classify it as:

       A  top-1 hits        -- the single most-confident box already finds the
                               target. Single-phase CRC on top-1 covers it.
       B  top-1 misses, some other box hits -- only threshold-lowering /
                               multi-box recovers it. This is what Phase 1 buys.
       C  no box hits at all -- unrecoverable by either design (model miss);
                               a hard floor on the achievable miss rate.

  Verdict: if B is rare, collapse to single-phase CRC on the top-1 box (no
  flood, one region). If B is sizeable, keep two-phase SeqCRC. If C exceeds
  alpha_cnf, Phase 1 is infeasible regardless.

    cd ~/croco_detection
    python scripts/diagnose_top1.py

Reuses the CONFIG in calibrate_seqcrc.py so the threshold matches the pipeline.
Results (PNG + results.txt) land in outputs/seqcrc_diag.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from conformal.calibrator import Calibrator
from conformal.dataset import CalibrationDataset, make_calibration_loader
from conformal.expansion.confidence_filter import confidence_filter_expansion
from conformal.loss.detection import (
    make_detection_miss_loss,
    make_detection_risk,
    make_iou_hit,
    nonzero_overlap_hit,
)
from conformal.prediction.yolo import YoloPredictor
from conformal.seqcrc import effective_threshold

# Reuse the pipeline's CONFIG so the diagnostic matches what we'd deploy.
from calibrate_seqcrc import (
    ALPHA,
    ALPHA_CNF_FRACTION,
    CONF_FLOOR,
    CONF_LAMBDA_RANGE,
    HIT_CRITERION,
    IOU_MATCH,
    SPLITS,
    WEIGHTS,
)
from crc_common import _Tee, check_paths, count_lines, rule


def _hit_fn():
    """The per-(gt, preds) hit predicate for the configured criterion."""
    if HIT_CRITERION == "iou":
        return make_iou_hit(IOU_MATCH), f"IoU >= {IOU_MATCH}"
    if HIT_CRITERION == "overlap":
        return nonzero_overlap_hit, "nonzero overlap"
    raise ValueError(f"HIT_CRITERION must be 'overlap' or 'iou', got {HIT_CRITERION!r}")


def _top1(preds: torch.Tensor) -> torch.Tensor:
    """The single highest-confidence box (col 4), shape [1, 5]; empty if none."""
    if preds.numel() == 0:
        return preds
    idx = int(torch.argmax(preds[:, 4]).item())
    return preds[idx:idx + 1]


def _hits_any_gt(box: torch.Tensor, gt: torch.Tensor, hit_fn) -> bool:
    """Does `box` (>=1 row) hit at least one valid GT under `hit_fn`?"""
    if box.numel() == 0 or gt.numel() == 0:
        return False
    for k in range(gt.shape[0]):
        g = gt[k]
        if (g[2] - g[0]) <= 0 or (g[3] - g[1]) <= 0:
            continue
        if hit_fn(g, box):
            return True
    return False


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} ({(100.0 * num / den if den else 0.0):5.1f}%)"


def main() -> None:
    out_dir = ROOT / "outputs" / "seqcrc_diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.txt"
    original_stdout = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original_stdout, fh)
        try:
            _run(out_dir)
        finally:
            sys.stdout = original_stdout
    print(f"\n(full log written to {results_path})")


ROOT = Path(__file__).resolve().parent.parent


def _run(out_dir: Path) -> None:
    calib_path = SPLITS / "calibration.txt"
    alpha_cnf = ALPHA * ALPHA_CNF_FRACTION
    hit_fn, hit_label = _hit_fn()
    detection_loss = make_detection_miss_loss(hit_fn)

    rule("CONFIG")
    print(f"  weights              : {WEIGHTS}")
    print(f"  calibration split    : {calib_path}")
    print(f"  alpha / alpha_cnf    : {ALPHA} / {alpha_cnf:.4f}")
    print(f"  confidence floor     : {CONF_FLOOR}")
    print(f"  hit criterion        : {hit_label}")
    check_paths(WEIGHTS, calib_path)
    print(f"  calibration images   : {count_lines(calib_path)}")

    rule("LOADING MODEL")
    predictor = YoloPredictor(str(WEIGHTS))
    print("  YOLO model loaded")

    # ── Phase 1 calibration -> the real operating threshold T_eff ─────────────
    rule("PHASE 1 -- find the operating threshold T_eff")
    phase1 = Calibrator(
        prediction_fn=predictor,
        expansion_fn=confidence_filter_expansion,
        risk_fn=make_detection_risk(hit_fn),
        alpha=alpha_cnf,
        confidence_threshold=CONF_FLOOR,
    )
    calib_loader = make_calibration_loader(calib_path, batch_size=16, num_workers=4)
    t_eff = None
    try:
        lam_cnf = phase1.calibrate(calib_loader, lambda_range=CONF_LAMBDA_RANGE)
        t_eff = effective_threshold(CONF_FLOOR, lam_cnf)
        print(f"  lambda_cnf = {lam_cnf:.4f}  ->  T_eff = {t_eff:.4f}")
    except RuntimeError as e:
        print(f"  Phase 1 INFEASIBLE at alpha_cnf={alpha_cnf:.4f}: {e}")
        print("  (continuing the top-1 analysis at the floor only)")

    # ── One pass: box counts + A/B/C breakdown at floor and at T_eff ──────────
    rule("SCANNING CALIBRATION SET")
    dataset = CalibrationDataset(calib_path)
    counts_floor: list[int] = []
    counts_teff: list[int] = []
    n_gt = 0
    a_floor = b_floor = c_floor = 0          # top-1 hit / recovered / unrecoverable @ floor
    a_teff = b_teff = c_teff = 0             # same @ operating threshold
    n_empty_teff = 0                         # frames with NO box at T_eff

    for i in range(len(dataset)):
        path, gt = dataset[i]
        preds = predictor(path, CONF_FLOOR)
        counts_floor.append(int(preds.shape[0]) if preds.numel() else 0)

        teff_preds = (preds[preds[:, 4] >= t_eff] if (t_eff is not None and preds.numel())
                      else (preds if t_eff is None else preds[:0]))
        counts_teff.append(int(teff_preds.shape[0]) if teff_preds.numel() else 0)

        if gt.numel() == 0:
            continue
        n_gt += 1

        # @ floor (full candidate pool): is top-1 enough, or does Phase 1 buy recall?
        top1_hit = _hits_any_gt(_top1(preds), gt, hit_fn)
        all_hit = detection_loss(preds, gt) == 0.0
        if top1_hit:
            a_floor += 1
        elif all_hit:
            b_floor += 1
        else:
            c_floor += 1

        # @ operating threshold (what we'd actually ship)
        if t_eff is not None:
            if teff_preds.numel() == 0:
                n_empty_teff += 1
            top1_hit_t = _hits_any_gt(_top1(teff_preds), gt, hit_fn)
            all_hit_t = detection_loss(teff_preds, gt) == 0.0
            if top1_hit_t:
                a_teff += 1
            elif all_hit_t:
                b_teff += 1
            else:
                c_teff += 1

    # ── Box-count report ──────────────────────────────────────────────────────
    rule("BOXES PER FRAME")

    def _count_stats(label: str, counts: list[int]) -> None:
        if not counts:
            return
        multi = sum(1 for c in counts if c > 1)
        print(f"  {label}")
        print(f"     mean {statistics.mean(counts):.2f} | "
              f"median {statistics.median(counts):.0f} | "
              f"max {max(counts)} | "
              f"frames with >1 box: {_pct(multi, len(counts))}")

    _count_stats(f"at floor (conf >= {CONF_FLOOR})", counts_floor)
    if t_eff is not None:
        _count_stats(f"at operating point T_eff = {t_eff:.4f}", counts_teff)
        print("\n  --> the flood at the floor is the candidate pool, NOT what you "
              "ship.\n      Judge precision by the T_eff row above.")

    # ── A/B/C report ──────────────────────────────────────────────────────────
    rule(f"TOP-1 vs MULTI-BOX  ({n_gt} frames with a target)")
    print("  At the floor (full candidate pool):")
    print(f"     A  top-1 already hits            : {_pct(a_floor, n_gt)}")
    print(f"     B  top-1 misses, another box hits: {_pct(b_floor, n_gt)}  <- what Phase 1 buys")
    print(f"     C  no box hits at all            : {_pct(c_floor, n_gt)}  <- unrecoverable")
    if t_eff is not None:
        print(f"\n  At the operating point T_eff = {t_eff:.4f}:")
        print(f"     A  top-1 hits                    : {_pct(a_teff, n_gt)}")
        print(f"     B  top-1 misses, another box hits: {_pct(b_teff, n_gt)}")
        print(f"     C  no box hits at all            : {_pct(c_teff, n_gt)}")
        print(f"     (frames with NO box at T_eff     : {_pct(n_empty_teff, n_gt)})")

    # ── Verdict ───────────────────────────────────────────────────────────────
    rule("VERDICT")
    b_rate = b_floor / n_gt if n_gt else 0.0
    c_rate = c_floor / n_gt if n_gt else 0.0
    alpha_loc = ALPHA - alpha_cnf
    if c_rate > alpha_cnf:
        print(f"  ! C = {c_rate:.3f} > alpha_cnf = {alpha_cnf:.3f}: the model misses "
              "more targets than the\n    Phase-1 budget allows -- no threshold "
              "fixes this. Model/data work needed,\n    or raise alpha_cnf.")
    if b_rate <= alpha_loc:
        print(f"  -> RECOMMEND single-phase CRC on top-1: B = {b_rate:.3f} is within "
              f"the localization\n     budget alpha_loc = {alpha_loc:.3f}, so the most-"
              "confident box is essentially always\n     the target. Drop Phase 1; "
              "calibrate the additive margin on top-1 boxes.")
    else:
        print(f"  -> RECOMMEND keeping two-phase SeqCRC: B = {b_rate:.3f} exceeds "
              f"alpha_loc = {alpha_loc:.3f},\n     so lowering the threshold genuinely "
              "recovers targets top-1 would miss.\n     Consider HIT_CRITERION='iou' to "
              "stop spurious overlaps counting as hits.")

    # ── Plot: boxes-per-frame histogram, floor vs T_eff ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hi = max(max(counts_floor, default=0), max(counts_teff, default=0))
    bins = range(0, hi + 2)
    ax.hist(counts_floor, bins=bins, align="left", alpha=0.6,
            color="#9ecae1", label=f"floor ({CONF_FLOOR})")
    if t_eff is not None:
        ax.hist(counts_teff, bins=bins, align="left", alpha=0.6,
                color="#1f77b4", label=f"T_eff ({t_eff:.3f})")
    ax.set_xlabel("boxes per frame")
    ax.set_ylabel("number of frames")
    ax.set_title("Boxes per frame: candidate pool vs operating point")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = out_dir / "boxes_per_frame.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {plot_path}")
    rule()


if __name__ == "__main__":
    main()
