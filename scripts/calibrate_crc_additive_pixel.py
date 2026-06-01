"""CRC run with PIXEL-WISE RECALL loss + ADDITIVE expansion (alpha = 9%).

Same pixel-loss guarantee as calibrate_crc.py but the conformal knob is an
additive pixel margin: every box grows by λ pixels on each side, regardless
of box size. Compare lambda-hat and the box-area cost against the multiplicative
run (crc_09) — the difference reveals whether clip sizes are uniform enough
that a fixed pixel margin is as efficient as a scale-proportional one.

    cd ~/croco_detection
    python scripts/calibrate_crc_additive_pixel.py

Results land in outputs/crc_add_pixel_09/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conformal.efficiency.box_area import total_box_area
from conformal.expansion.additive import additive_expansion
from conformal.loss.pixel import pixel_risk

from crc_common import run_pipeline


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"

ALPHA = 0.09
CONFIDENCE_THRESHOLD = 0.30
LAMBDA_RANGE = (0.0, 100.0)   # pixels; upper bound well above any clip dimension
LAMBDA_HAT_OVERRIDE = None


def main() -> None:
    run_pipeline(
        weights=WEIGHTS,
        calib_path=SPLITS / "calibration.txt",
        test_path=SPLITS / "test.txt",
        risk_fn=pixel_risk,
        expansion_fn=additive_expansion,
        efficiency_fn=total_box_area,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        lambda_range=LAMBDA_RANGE,
        output_dir=ROOT / "outputs" / "crc_add_pixel_09",
        baseline_lambda=0.0,
        baseline_label="raw",
        calib_label="lambda-hat",
        loss_name="pixel-wise recall (image_pixel_loss)",
        expansion_name="additive +λ px per side",
        eff_name="box area",
        eff_unit="px^2",
        lambda_hat_override=LAMBDA_HAT_OVERRIDE,
    )


if __name__ == "__main__":
    main()
