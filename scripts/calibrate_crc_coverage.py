"""CRC run with the COVERAGE-INDICATOR loss (75%) + multiplicative expansion.

Same geometric knob as scripts/calibrate_crc.py (boxes grow by w*lambda /
h*lambda), but the loss is binary per GT: a clip counts as covered only if
>= 75% of its pixel area is inside the predicted set, otherwise it is a full
miss. So here alpha bounds the FRACTION OF CLIPS that fail the 75% bar,
rather than mean pixel slippage. Compare lambda-hat and the verdict against
the pixel-loss run.

    cd ~/croco_detection
    python scripts/calibrate_crc_coverage.py

Results (PNGs + results.txt) land in outputs/crc_cov_<alpha%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` and `import crc_common` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.efficiency.box_area import total_box_area
from conformal.expansion.multiplicative import multiplicative_expansion
from conformal.loss.coverage import COVERAGE_FRACTION, coverage_risk

from crc_common import run_pipeline


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

ALPHA = 0.09                  # bound on the fraction of clips missing the 75% bar
CONFIDENCE_THRESHOLD = 0.30
LAMBDA_RANGE = (0.0, 2.0)
LAMBDA_HAT_OVERRIDE = None    # paste a saved lambda-hat to skip recalibration


def main() -> None:
    run_pipeline(
        weights=WEIGHTS,
        calib_path=SPLITS / "calibration.txt",
        test_path=SPLITS / "test.txt",
        risk_fn=coverage_risk,
        expansion_fn=multiplicative_expansion,
        efficiency_fn=total_box_area,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / f"crc_cov_{round(ALPHA * 100):02d}",
        baseline_lambda=0.0,                 # multiplicative at lambda=0 = raw model
        baseline_label="raw",
        calib_label="lambda-hat",
        loss_name=f"coverage-indicator ({COVERAGE_FRACTION:.0%})",
        expansion_name="multiplicative C_lambda",
        eff_name="box area",
        eff_unit="px^2",
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
