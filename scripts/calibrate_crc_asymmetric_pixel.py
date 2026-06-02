"""CRC run with PIXEL-WISE RECALL loss + asymmetric multiplicative expansion.

Uses the asymmetric multiplicative expansion (horizontal factor 1/3) and
measures coverage quality via the pixel-wise recall loss. Compare lambda-hat
and cost against calibrate_crc.py (full multiplicative) to see how much
efficiency is gained by reducing horizontal margin.

    cd ~/croco_detection
    python scripts/calibrate_crc_asymmetric_pixel.py

Results (PNGs + results.txt) land in outputs/crc_asym_<alpha%>.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.efficiency.box_area import total_box_area
from conformal.expansion.asymmetric_multiplicative import asymmetric_multiplicative_expansion
from conformal.loss.pixel import pixel_risk

from crc_common import run_pipeline


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

ALPHA = 0.09                  # target risk: allow <= 9% mean pixel miss
CONFIDENCE_THRESHOLD = 0.30
LAMBDA_RANGE = (0.0, 2.0)
LAMBDA_HAT_OVERRIDE = None    # paste a saved lambda-hat to skip recalibration


def main() -> None:
    run_pipeline(
        weights=WEIGHTS,
        calib_path=SPLITS / "calibration.txt",
        test_path=SPLITS / "test.txt",
        risk_fn=pixel_risk,
        expansion_fn=asymmetric_multiplicative_expansion,
        efficiency_fn=total_box_area,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / f"crc_asym_{round(ALPHA * 100):02d}",
        baseline_lambda=0.0,
        baseline_label="raw",
        calib_label="lambda-hat",
        loss_name="pixel-wise recall (image_pixel_loss)",
        expansion_name="asymmetric multiplicative (h×1, w×1/3)",
        eff_name="box area",
        eff_unit="px^2",
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
