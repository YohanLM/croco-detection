"""End-to-end Pixel-Wise Conformal Risk Control calibration.

Wires the prediction / margin / risk building blocks from `conformal/` into
a `Calibrator` and runs calibration on the held-out calibration split.
Prints the calibrated multiplier `λ̂`. Test-set evaluation lives elsewhere.

Run from the project root once the splits are populated:

    python scripts/calibrate_crc.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformal import (
    Calibrator,
    YoloPredictor,
    make_calibration_loader,
    multiplicative_expansion,
    pixel_risk,
)


# ── Configuration ────────────────────────────────────────────────────────────

WEIGHTS              = "models/best.pt"
ALPHA                = 0.10          # target upper bound on E[L_pixel]
CONFIDENCE_THRESHOLD = 0.30          # methodology §4.2
BATCH_SIZE           = 16
SPLITS_DIR           = Path("data/splits")
LAMBDA_RANGE         = (0.0, 2.0)


def main() -> None:
    cal_loader = make_calibration_loader(SPLITS_DIR / "calib.txt", batch_size=BATCH_SIZE)

    print(f"Calibrating on {len(cal_loader.dataset)} images (α = {ALPHA}) ...")
    calibrator = Calibrator(
        prediction_fn        = YoloPredictor(WEIGHTS),
        expansion_fn         = multiplicative_expansion,
        risk_fn              = pixel_risk,
        alpha                = ALPHA,
        confidence_threshold = CONFIDENCE_THRESHOLD,
    )
    lambda_hat = calibrator.calibrate(cal_loader, lambda_range=LAMBDA_RANGE)
    print(f"  → λ̂ = {lambda_hat:.4f}")


if __name__ == "__main__":
    main()
