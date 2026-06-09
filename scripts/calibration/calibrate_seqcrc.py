"""Two-step Sequential Conformal Risk Control (SeqCRC), single-class, end to end.

Implements the procedure in `docs/seqcrc_single_class_spec.md`:

  Step 1 (CONFIDENCE)   FNR (box-count) loss + confidence threshold.
      Calibrates two estimators on the sorted confidence scores:
      lambda_cnf_plus  (B=1, used at inference) and
      lambda_cnf_minus (B=0, feeds Step 2). The conservative risk
      R_tilde_cnf = max(R_cnf, R_loc(., lambda_bar_loc)) is monotonized in
      lambda_cnf on the fly.

  Step 2 (LOCALIZATION) pixel-superposition-threshold loss + margin expansion.
      Binary-searches lambda_loc_plus so the monotonized
      R_loc(lambda_cnf_minus, lambda_loc) clears alpha_loc.

Inference filters at conf >= 1 - lambda_cnf_plus, then expands each surviving
box by lambda_loc_plus. The end-to-end guarantee is E[L_loc] <= alpha_loc,
valid when alpha_loc >= alpha_cnf + 1/(n+1) (checked at startup).

    cd ~/croco_detection
    python scripts/calibration/calibrate_seqcrc.py

Results (PNGs + results.txt) land in outputs/seqcrc_<alpha_loc%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` and `import crc_common` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.dataset import make_calibration_loader
from conformal.efficiency.box_area import total_box_area
from conformal.prediction.top1 import TopKPredictor
from conformal.prediction.yolo import YoloPredictor
from conformal.seqcrc import (
    SeqCRCConfig,
    SeqCRCInferencer,
    calibrate,
    collect_predictions,
    confidence_risk,
    localization_risk,
    localization_set,
)

from crc_common import _Tee, check_paths, count_lines, rule
from seqcrc_report import save_overlays, save_risk_curves


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

# Target error rates, chosen a priori (NEVER from data). The guarantee needs
# alpha_loc >= alpha_cnf + 1/(n+1); validated at startup against the realized n.
ALPHA_CNF = 0.03      # confidence (FNR) budget
ALPHA_LOC = 0.09      # localization budget = the operative end-to-end target

# Localization loss / expansion, fixed a priori.
TAU_PIX = 0.75               # pixel-superposition threshold in (0, 1]
MARGIN_MODE = "additive"     # "additive" (px/side) | "multiplicative" (scale)
LAMBDA_BAR_LOC = 100.0       # upper bound of Lambda_loc (px for additive)

PREFILTER = 1e-3             # data-independent confidence floor (pre-step)
BISECTION_STEPS = 25         # binary-search steps for lambda_loc_plus

# Single-object regime: keep only the top-k highest-confidence boxes per frame.
TOP_K = 1                    # set to None to keep all boxes

# Paste saved values to skip recalibration.
LAMBDA_CNF_OVERRIDE = None
LAMBDA_LOC_OVERRIDE = None


def main() -> None:
    output_dir = ROOT / "outputs" / f"seqcrc_{round(ALPHA_LOC * 100):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.txt"

    original_stdout = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original_stdout, fh)
        try:
            _run(output_dir)
        finally:
            sys.stdout = original_stdout
    print(f"\n(full log written to {results_path})")


def _run(output_dir: Path) -> None:
    calib_path = SPLITS / "calibration.txt"
    test_path = SPLITS / "test.txt"
    cfg = SeqCRCConfig(
        alpha_cnf=ALPHA_CNF,
        alpha_loc=ALPHA_LOC,
        tau_pix=TAU_PIX,
        margin_mode=MARGIN_MODE,
        lambda_bar_loc=LAMBDA_BAR_LOC,
        prefilter=PREFILTER,
        bisection_steps=BISECTION_STEPS,
    )

    # ── CONFIG echo ───────────────────────────────────────────────────────────
    rule("CONFIG")
    print(f"  weights              : {WEIGHTS}")
    print(f"  calibration split    : {calib_path}")
    print(f"  test split           : {test_path}")
    print(f"  alpha_cnf (FNR)      : {cfg.alpha_cnf}")
    print(f"  alpha_loc (target)   : {cfg.alpha_loc}")
    print(f"  tau_pix              : {cfg.tau_pix}")
    print(f"  margin mode          : {cfg.margin_mode}")
    print(f"  lambda_bar_loc       : {cfg.lambda_bar_loc}")
    print(f"  prefilter floor      : {cfg.prefilter}")
    print(f"  bisection steps      : {cfg.bisection_steps}")
    print(f"  top-k per frame      : {TOP_K if TOP_K is not None else 'all boxes'}")

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

    # ── CALIBRATION (two-step SeqCRC) ─────────────────────────────────────────
    rule("CALIBRATION -- two-step SeqCRC")
    if LAMBDA_CNF_OVERRIDE is not None and LAMBDA_LOC_OVERRIDE is not None:
        lam_cnf = float(LAMBDA_CNF_OVERRIDE)
        lam_loc = float(LAMBDA_LOC_OVERRIDE)
        print(f"  using saved lambda_cnf_plus = {lam_cnf:.4f}, "
              f"lambda_loc_plus = {lam_loc:.4f} (overrides set)")
        cfg.validate(count_lines(calib_path))
    else:
        print("  predicting at prefilter floor + two-step search ...")
        try:
            result = calibrate(predictor, calib_loader, cfg)
        except (RuntimeError, ValueError) as e:
            print(f"\n  CALIBRATION FAILED -- {e}")
            sys.exit(2)
        lam_cnf = result.lambda_cnf_plus
        lam_loc = result.lambda_loc_plus
        print(f"\n  calibration images (n) = {result.n}")
        print(f"  lambda_cnf_plus        = {lam_cnf:.4f}  "
              f"-> conf cutoff 1 - lambda = {1.0 - lam_cnf:.4f}")
        print(f"  lambda_cnf_minus       = {result.lambda_cnf_minus:.4f}  "
              "(optimistic, fed to Step 2)")
        print(f"  lambda_loc_plus        = {lam_loc:.4f} "
              f"({'px/side' if cfg.margin_mode == 'additive' else 'scale'})")
        print(f"  calib R_cnf(plus)      = {result.risk_cnf:.4f}  (<= alpha_cnf "
              f"{cfg.alpha_cnf})")
        print(f"  calib R_loc(plus,plus) = {result.risk_loc:.4f}  (<= alpha_loc "
              f"{cfg.alpha_loc})")

    # ── END-TO-END VALIDATION on the test set ─────────────────────────────────
    rule("END-TO-END VALIDATION (test set)")
    print("  predicting test set at prefilter floor ...")
    preds, gts = collect_predictions(predictor, test_loader, cfg.prefilter)
    test_r_cnf = confidence_risk(preds, gts, lam_cnf, cfg)
    test_r_loc = localization_risk(preds, gts, lam_cnf, lam_loc, cfg)

    fields = [localization_set(p, lam_cnf, lam_loc, cfg.prefilter, cfg.margin_mode)
              for p in preds]
    raw_fields = [localization_set(p, lam_cnf, 0.0, cfg.prefilter, cfg.margin_mode)
                  for p in preds]
    n_test = len(preds)
    mean_area = sum(total_box_area(f) for f in fields) / n_test
    mean_area_raw = sum(total_box_area(f) for f in raw_fields) / n_test

    print(f"  test images (n)        : {n_test}")
    print(f"  test R_cnf(plus)       : {test_r_cnf:.4f}  "
          f"(target alpha_cnf {cfg.alpha_cnf}, slack {cfg.alpha_cnf - test_r_cnf:+.4f})")
    print(f"  test R_loc(plus,plus)  : {test_r_loc:.4f}  "
          f"(target alpha_loc {cfg.alpha_loc}, slack {cfg.alpha_loc - test_r_loc:+.4f})")
    print(f"  mean Gamma_loc area    : {mean_area:,.0f} px^2  "
          f"(raw, no margin: {mean_area_raw:,.0f} px^2, "
          f"inflation {mean_area / mean_area_raw:.2f}x)" if mean_area_raw > 0
          else f"  mean Gamma_loc area    : {mean_area:,.0f} px^2")

    # ── PLOTS + OVERLAYS ──────────────────────────────────────────────────────
    rule("PLOTS")
    for path in save_risk_curves(preds, gts, lam_cnf, lam_loc, cfg, output_dir):
        print(f"  saved {path}")

    rule("EXAMPLE OVERLAYS (calibrated Gamma_loc field)")
    inferencer = SeqCRCInferencer.from_config(predictor, lam_cnf, lam_loc, cfg)
    for path in save_overlays(inferencer, test_path, output_dir, n_examples=6):
        print(f"  saved {path}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    rule("VERDICT")
    print(f"  Step 1: lambda_cnf_plus = {lam_cnf:.4f}  (FNR <= {cfg.alpha_cnf})")
    print(f"  Step 2: lambda_loc_plus = {lam_loc:.4f}  "
          f"(localization <= {cfg.alpha_loc})")
    verdict = ("PASS" if test_r_loc <= cfg.alpha_loc else "FAIL")
    print(f"  end-to-end test R_loc = {test_r_loc:.4f}  vs  alpha_loc = "
          f"{cfg.alpha_loc:.4f}  ->  {verdict}")
    rule()


if __name__ == "__main__":
    main()
