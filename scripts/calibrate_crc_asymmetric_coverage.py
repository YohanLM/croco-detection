"""CRC run with COVERAGE-INDICATOR loss (75%) + asymmetric multiplicative expansion.

Uses the asymmetric multiplicative expansion (horizontal factor 1/3) and
measures coverage quality via the binary 75%-coverage indicator: a clip is a
miss if less than 75% of its GT pixel area falls inside the predicted set.
Compare lambda-hat and cost against calibrate_crc_coverage.py (full
multiplicative) to quantify the efficiency gain from the reduced horizontal
margin under this coarser loss.

    cd ~/croco_detection
    python scripts/calibrate_crc_asymmetric_coverage.py

Results (PNGs + results.txt) land in outputs/crc_asym_cov_<alpha%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.efficiency.box_area import total_box_area
from conformal.expansion.asymmetric_multiplicative import asymmetric_multiplicative_expansion
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
        expansion_fn=asymmetric_multiplicative_expansion,
        efficiency_fn=total_box_area,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / f"crc_asym_cov_{round(ALPHA * 100):02d}",
        baseline_lambda=0.0,
        baseline_label="raw",
        calib_label="lambda-hat",
        loss_name=f"coverage-indicator ({COVERAGE_FRACTION:.0%})",
        expansion_name="asymmetric multiplicative (h×1, w×1/3)",
        eff_name="box area",
        eff_unit="px^2",
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
