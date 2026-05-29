"""CRC run that tunes the CONFIDENCE THRESHOLD instead of box geometry.

Same coverage-indicator loss (75%) as calibrate_crc_coverage.py, but the
conformal knob is the confidence-filter expansion: lambda lowers the
effective confidence cutoff `T_eff = max(floor, 1 - lambda)`, ADMITTING
lower-confidence detections rather than growing boxes. So calibration here
answers: how far must the confidence threshold drop to guarantee that
<= alpha of clips miss the 75% coverage bar?

Key difference from the geometric pipelines (see NOTE below):

  - The detector is run at a LOW floor confidence (CONF_FLOOR) so a wide
    candidate pool exists for lambda to admit from. In this pipeline the
    `confidence_threshold` parameter therefore plays the role of the
    admission FLOOR, not the operating point.
  - lambda=0 gives an EMPTY set (cutoff 1.0), so "without calibration" is
    NOT lambda=0. The baseline is the lambda that reproduces the model's
    STANDARD threshold: 1 - STD_THRESHOLD.

    cd ~/croco_detection
    python scripts/calibrate_crc_confidence.py

Results (PNGs + results.txt) land in outputs/crc_conf_<alpha%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` and `import crc_common` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.diagnostics.detection import make_detection_counters
from conformal.efficiency.box_count import box_count
from conformal.expansion.confidence_filter import confidence_filter_expansion
from conformal.loss.coverage import COVERAGE_FRACTION, coverage_risk

from crc_common import run_pipeline


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

ALPHA = 0.05                  # bound on the fraction of clips missing the 75% bar
CONF_FLOOR = 0.001            # admission floor: YOLO runs here, lambda sweeps up from it
STD_THRESHOLD = 0.30          # the model's normal operating threshold (the baseline)
LAMBDA_RANGE = (0.0, 1.0)     # T_eff = max(floor, 1 - lambda): lambda in [0, 1]
IOU_MATCH = 0.10              # IoU above which a predicted box counts as a true positive
LAMBDA_HAT_OVERRIDE = None    # paste a saved lambda-hat to skip recalibration


def main() -> None:
    run_pipeline(
        weights=WEIGHTS,
        calib_path=SPLITS / "calibration.txt",
        test_path=SPLITS / "test.txt",
        risk_fn=coverage_risk,
        expansion_fn=confidence_filter_expansion,
        efficiency_fn=box_count,
        alpha=ALPHA,
        confidence_threshold=CONF_FLOOR,     # the admission floor (see NOTE)
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / f"crc_conf_{round(ALPHA * 100):02d}",
        baseline_lambda=1.0 - STD_THRESHOLD,  # reproduces the standard threshold
        baseline_label=f"std thr {STD_THRESHOLD}",
        calib_label="calibrated thr",
        loss_name=f"coverage-indicator ({COVERAGE_FRACTION:.0%})",
        expansion_name="confidence-filter (threshold knob)",
        eff_name="boxes admitted",
        eff_unit="boxes",
        report_effective_threshold=True,
        extra_metrics=make_detection_counters(IOU_MATCH),
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
