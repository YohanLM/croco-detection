"""Two-phase Sequential Conformal Risk Control (SeqCRC) end to end.

Composes the two existing single-knob CRC phases under the Bonferroni union
bound `alpha = alpha_cnf + alpha_loc`:

  Phase 1 (CONFIDENCE)  -- detection-miss loss + confidence-filter knob.
      Lowers the objectness threshold until <= alpha_cnf of frames have an
      *undetected* target (no surviving box near it). Yields lambda_cnf and
      the effective threshold T_eff = max(floor, 1 - lambda_cnf).

  Phase 2 (LOCALIZATION) -- 75 %-coverage-indicator loss + additive margin.
      On the SURVIVOR frames only (target detected at T_eff), grows boxes by
      an additive pixel margin until <= alpha_loc of survivors cover < 75 %
      of their target. Yields lambda_loc.

The two phases control complementary failure modes, so the union bound is a
genuine partition:

    {final coverage < 75%}  subset of  {target missed}  union  {detected but < 75% covered}

End-to-end VALIDATION (the real proof): run the composed pipeline -- detector
at T_eff, then additive +lambda_loc -- over the full test set and confirm the
empirical 75 %-coverage risk respects alpha.

    cd ~/croco_detection
    python scripts/calibrate_seqcrc.py

Results (PNGs + results.txt) land in outputs/seqcrc_<alpha%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` and `import crc_common` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.calibrator import Calibrator
from conformal.dataset import make_calibration_loader
from conformal.diagnostics.detection import make_detection_counters
from conformal.efficiency.box_area import total_box_area
from conformal.expansion.additive import additive_expansion
from conformal.expansion.confidence_filter import confidence_filter_expansion
from conformal.loss.coverage import COVERAGE_FRACTION, coverage_risk
from conformal.loss.detection import (
    detection_miss_loss,
    detection_risk,
    make_detection_miss_loss,
    make_iou_detection_risk,
    make_iou_hit,
)
from conformal.prediction.top1 import TopKPredictor
from conformal.prediction.yolo import YoloPredictor
from conformal.seqcrc import (
    SeqCRCInferencer,
    build_survivor_split,
    effective_threshold,
)

from crc_common import (
    _Tee,
    check_paths,
    count_lines,
    rule,
    save_example_overlays,
    save_plots,
)


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

# Global risk budget and its union-bound split. alpha = alpha_cnf + alpha_loc.
ALPHA = 0.09
ALPHA_CNF_FRACTION = 0.35     # fraction of the budget given to Phase 1. Smaller values
                              # force a tighter Phase-1 guarantee -> lower T_eff -> fewer
                              # missed detections. Must stay above the irreducible miss
                              # rate at the floor (~0.029 observed); if Phase 1 fails,
                              # raise this slightly.

# Phase-1 hit criterion. "overlap" = any nonzero pixel overlap counts as a hit
# (lenient); "iou" = require IoU >= IOU_MATCH (strict, cheaper Phase 2).
HIT_CRITERION = "overlap"     # "overlap" | "iou"
IOU_MATCH = 0.10              # used only when HIT_CRITERION == "iou"

# Single-object regime: keep only the TOP_K highest-confidence boxes per frame
# so lowering the threshold never floods duplicates/false positives for the one
# object. With TOP_K=1, Phase 1 is feasible only if alpha_cnf exceeds the rate
# at which top-1 is NOT the target (run diagnose_top1.py to read that rate).
TOP_K = 1                     # set to None to disable top-k selection (keep all boxes)

CONF_FLOOR = 0.001            # Phase-1 admission floor: YOLO runs here
STD_THRESHOLD = 0.30          # model's normal operating threshold (system baseline)
CONF_LAMBDA_RANGE = (0.0, 1.0)        # T_eff = max(floor, 1 - lambda): lambda in [0, 1]
ADDITIVE_LAMBDA_RANGE = (0.0, 100.0)  # pixels per side; well above any clip dim

# Paste saved values to skip recalibration of either phase.
LAMBDA_CNF_OVERRIDE = None
LAMBDA_LOC_OVERRIDE = None


def _phase1_risk_and_loss():
    """Return `(risk_fn, loss_fn, label)` for the chosen Phase-1 hit criterion."""
    if HIT_CRITERION == "iou":
        hit = make_iou_hit(IOU_MATCH)
        return (make_iou_detection_risk(IOU_MATCH),
                make_detection_miss_loss(hit),
                f"detection-miss (IoU >= {IOU_MATCH})")
    if HIT_CRITERION == "overlap":
        return detection_risk, detection_miss_loss, "detection-miss (nonzero overlap)"
    raise ValueError(f"HIT_CRITERION must be 'overlap' or 'iou', got {HIT_CRITERION!r}")


def main() -> None:
    alpha_cnf = ALPHA * ALPHA_CNF_FRACTION
    alpha_loc = ALPHA - alpha_cnf
    output_dir = ROOT / "outputs" / f"seqcrc_{round(ALPHA * 100):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.txt"

    original_stdout = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original_stdout, fh)
        try:
            _run(alpha_cnf, alpha_loc, output_dir)
        finally:
            sys.stdout = original_stdout
    print(f"\n(full log written to {results_path})")


def _run(alpha_cnf: float, alpha_loc: float, output_dir: Path) -> None:
    calib_path = SPLITS / "calibration.txt"
    test_path = SPLITS / "test.txt"
    phase1_risk, phase1_loss, phase1_label = _phase1_risk_and_loss()

    # ── CONFIG echo ───────────────────────────────────────────────────────────
    rule("CONFIG")
    print(f"  weights              : {WEIGHTS}")
    print(f"  calibration split    : {calib_path}")
    print(f"  test split           : {test_path}")
    print(f"  alpha (global budget): {ALPHA}")
    print(f"  union-bound split    : alpha_cnf={alpha_cnf:.4f} + "
          f"alpha_loc={alpha_loc:.4f} = {alpha_cnf + alpha_loc:.4f}")
    print(f"  Phase-1 loss         : {phase1_label}")
    print(f"  Phase-2 loss         : coverage-indicator ({COVERAGE_FRACTION:.0%})")
    print(f"  top-k per frame      : {TOP_K if TOP_K is not None else 'all boxes'}")
    print(f"  confidence floor     : {CONF_FLOOR}")
    print(f"  conf lambda range    : {CONF_LAMBDA_RANGE}")
    print(f"  additive lambda range: {ADDITIVE_LAMBDA_RANGE} (px/side)")

    check_paths(WEIGHTS, calib_path, test_path)
    print(f"  calibration images   : {count_lines(calib_path)}")
    print(f"  test images          : {count_lines(test_path)}")

    rule("LOADING MODEL")
    predictor = YoloPredictor(str(WEIGHTS))
    if TOP_K is not None:
        predictor = TopKPredictor(predictor, k=TOP_K)
        print(f"  YOLO model loaded (keeping top-{TOP_K} box per frame)")
    else:
        print("  YOLO model loaded (all boxes per frame)")
    calib_loader = make_calibration_loader(calib_path, batch_size=16, num_workers=4)
    test_loader = make_calibration_loader(test_path, batch_size=16, num_workers=4)

    # ── PHASE 1: confidence calibration (detection-miss loss) ─────────────────
    rule("PHASE 1 -- CONFIDENCE CALIBRATION")
    phase1 = Calibrator(
        prediction_fn=predictor,
        expansion_fn=confidence_filter_expansion,
        risk_fn=phase1_risk,
        alpha=alpha_cnf,
        confidence_threshold=CONF_FLOOR,
    )
    if LAMBDA_CNF_OVERRIDE is not None:
        lam_cnf = float(LAMBDA_CNF_OVERRIDE)
        print(f"  using saved lambda_cnf = {lam_cnf:.4f} (override set)")
    else:
        print("  running detector at floor + Brent root search on detection-miss ...")
        try:
            lam_cnf = phase1.calibrate(calib_loader, lambda_range=CONF_LAMBDA_RANGE)
        except RuntimeError as e:
            print(f"\n  PHASE 1 FAILED -- {e}")
            sys.exit(2)
    t_eff = effective_threshold(CONF_FLOOR, lam_cnf)
    print(f"\n  lambda_cnf            = {lam_cnf:.4f}")
    print(f"  -> effective threshold T_eff = max({CONF_FLOOR}, 1 - {lam_cnf:.4f}) "
          f"= {t_eff:.4f}")
    p1_eval = phase1.evaluate(calib_loader, lam_cnf)
    print(f"  Phase-1 calib risk    = {p1_eval.risk:.4f}  "
          f"(CRC bound {p1_eval.crc_bound:.4f} <= alpha_cnf {alpha_cnf})")

    # ── Build the Phase-2 survivor subset (n_loc) ─────────────────────────────
    rule("PHASE 2 -- SURVIVOR SUBSET")
    survivor_split = output_dir / "survivors_calibration.txt"
    print("  selecting frames whose target was detected at T_eff ...")
    n_total, n_loc, _ = build_survivor_split(
        predictor, calib_path, t_eff, phase1_loss, survivor_split)
    print(f"  survivors (n_loc)     = {n_loc} / {n_total} calibration frames")
    print(f"  -> written to {survivor_split}")
    if n_loc == 0:
        print("\n  PHASE 2 ABORTED -- no survivors to calibrate localization on.")
        sys.exit(2)

    # ── PHASE 2: localization calibration (75% coverage, additive margin) ─────
    rule("PHASE 2 -- LOCALIZATION CALIBRATION")
    phase2 = Calibrator(
        prediction_fn=predictor,
        expansion_fn=additive_expansion,
        risk_fn=coverage_risk,
        alpha=alpha_loc,
        confidence_threshold=t_eff,        # detector now runs at the Phase-1 threshold
    )
    survivor_loader = make_calibration_loader(
        survivor_split, batch_size=16, num_workers=4)
    if LAMBDA_LOC_OVERRIDE is not None:
        lam_loc = float(LAMBDA_LOC_OVERRIDE)
        print(f"  using saved lambda_loc = {lam_loc:.4f} (override set)")
    else:
        print("  running detector at T_eff over survivors + Brent on coverage ...")
        try:
            lam_loc = phase2.calibrate(
                survivor_loader, lambda_range=ADDITIVE_LAMBDA_RANGE)
        except RuntimeError as e:
            print(f"\n  PHASE 2 FAILED -- {e}")
            sys.exit(2)
    print(f"\n  lambda_loc            = {lam_loc:.4f} px per side")
    p2_eval = phase2.evaluate(survivor_loader, lam_loc)
    print(f"  Phase-2 survivor risk = {p2_eval.risk:.4f}  "
          f"(CRC bound {p2_eval.crc_bound:.4f} <= alpha_loc {alpha_loc})")

    # ── END-TO-END VALIDATION on the full test set ────────────────────────────
    rule("END-TO-END VALIDATION (test set)")
    print("  composed pipeline: detector at T_eff -> additive +lambda_loc")
    print("  measuring 75%-coverage risk over the FULL test set ...")
    # The end-to-end pipeline IS a Calibrator with the additive knob run at
    # T_eff: YOLO@T_eff applies the Phase-1 filter, additive applies Phase 2.
    e2e = Calibrator(
        prediction_fn=predictor,
        expansion_fn=additive_expansion,
        risk_fn=coverage_risk,
        alpha=ALPHA,
        confidence_threshold=t_eff,
    )
    # Dense near 0 (where λ_loc typically lands for well-localised detectors),
    # then coarser up to the ceiling so the full range is visible.
    lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
               10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    detection_counters = make_detection_counters(IOU_MATCH)
    res = e2e.evaluate(
        test_loader, lam_loc, efficiency_fn=total_box_area,
        risk_curve_lambdas=lambdas, extra_metrics=detection_counters)
    print(res.summary(eff_name="box area", eff_unit="px^2"))

    # ── System baseline: standard threshold, no expansion ─────────────────────
    rule("WITH vs WITHOUT SeqCRC (test set)")
    print(f"  baseline: standard threshold {STD_THRESHOLD}, no expansion (raw) ...")
    baseline = Calibrator(
        prediction_fn=predictor,
        expansion_fn=additive_expansion,
        risk_fn=coverage_risk,
        alpha=ALPHA,
        confidence_threshold=STD_THRESHOLD,
    )
    res_raw = baseline.evaluate(
        test_loader, 0.0, efficiency_fn=total_box_area,
        extra_metrics=detection_counters)
    print()
    print(res.comparison(
        res_raw, res, eff_name="box area",
        baseline_label=f"std thr {STD_THRESHOLD}", calib_label="SeqCRC"))

    # ── Plots + overlays (Phase-2 effect on survivors, at T_eff) ──────────────
    rule("PLOTS")
    saved = save_plots(
        res, res_raw, lam_loc, p2_eval.risk, output_dir,
        eff_name="box area", eff_unit="px^2",
        baseline_label=f"std thr {STD_THRESHOLD}", calib_label="SeqCRC")
    for path in saved:
        print(f"  saved {path}")

    rule("EXAMPLE OVERLAYS (Phase-2 expansion effect, detector at T_eff)")
    # Overlays share one calibrator: e2e at lambda=0 (filtered, unexpanded)
    # vs lambda_loc (expanded), so the panels isolate what Phase 2 adds.
    res_filtered = e2e.evaluate(test_loader, 0.0)
    examples = save_example_overlays(
        e2e, test_path, lam_loc, 0.0,
        res_filtered.per_image_losses, res.per_image_losses,
        output_dir, n_examples=6,
        baseline_label="filtered, no expand", calib_label="+lambda_loc")
    for path in examples:
        print(f"  saved {path}")

    # ── Calibrated runtime handle (composed inferencer) ───────────────────────
    inferencer = SeqCRCInferencer(predictor, t_eff, lam_loc)
    _ = inferencer  # constructed here to document the deployment entry point

    # ── Verdict ───────────────────────────────────────────────────────────────
    rule("VERDICT")
    print(f"  Phase 1: lambda_cnf = {lam_cnf:.4f} -> T_eff = {t_eff:.4f} "
          f"(detection-miss <= {alpha_cnf:.4f})")
    print(f"  Phase 2: lambda_loc = {lam_loc:.4f} px "
          f"(survivor coverage-miss <= {alpha_loc:.4f})")
    print(f"  union bound: global failure <= alpha_cnf + alpha_loc = {ALPHA:.4f}")
    print(f"  end-to-end test risk  = {res.risk:.4f}  vs  alpha = {ALPHA:.4f}")
    print(f"  {res.verdict}")
    rule()


if __name__ == "__main__":
    main()
