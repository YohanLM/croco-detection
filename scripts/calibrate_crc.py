"""CRC run with the PIXEL-WISE RECALL loss + multiplicative expansion.

Calibrates the multiplicative margin lambda-hat on the calibration split,
then evaluates it on the held-out test split with the pixel-wise recall
loss. The printed report shows the calibrated lambda-hat, the test risk vs
the target alpha, the finite-sample CRC bound, the box-area inflation cost,
a risk curve, and a PASS/FAIL verdict.

This is the reference pixel-loss run; the coverage / confidence variants
share the same calibrate -> evaluate -> report pipeline (see crc_common).

    cd ~/croco_detection
    python scripts/calibrate_crc.py

Results (PNGs + results.txt) land in outputs/crc_<alpha%>, e.g. crc_08 for 8%.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` and `import crc_common` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.efficiency.box_area import total_box_area
from conformal.expansion.multiplicative import multiplicative_expansion
from conformal.loss.pixel import pixel_risk

from crc_common import run_pipeline


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

ALPHA = 0.09                  # target risk: allow <= 9% mean pixel miss
CONFIDENCE_THRESHOLD = 0.30   # methodology section 4.2
LAMBDA_RANGE = (0.0, 2.0)
LAMBDA_HAT_OVERRIDE = None     # paste a saved lambda-hat to skip recalibration


def main() -> None:
    run_pipeline(
        weights=WEIGHTS,
        calib_path=SPLITS / "calibration.txt",
        test_path=SPLITS / "test.txt",
        risk_fn=pixel_risk,
        expansion_fn=multiplicative_expansion,
        efficiency_fn=total_box_area,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / f"crc_{round(ALPHA * 100):02d}",
        baseline_lambda=0.0,                 # multiplicative at lambda=0 = raw model
        baseline_label="raw",
        calib_label="lambda-hat",
        loss_name="pixel-wise recall (image_pixel_loss)",
        expansion_name="multiplicative C_lambda",
        eff_name="box area",
        eff_unit="px^2",
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
