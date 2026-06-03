"""Sanity overlay: raw top-1 vs smoothed top-1 (median) on a few test images.

Confirms the `SmoothedTop1Predictor` output contract and shows, visually, what
the smoothing does: the faint cloud is the per-copy top-1 boxes under noise (the
instability the median averages out), blue is the raw (no-noise) top-1, orange
is the smoothed median, green is the GT.

    cd ~/croco_detection
    python scripts/smoothing/smooth_predict_demo.py

Output (PNGs + results.txt) lands in outputs/smoothing_demo/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from conformal.dataset import CalibrationDataset
from conformal.prediction.yolo import YoloPredictor
from conformal.smoothing.predictor import SmoothedTop1Predictor

from crc_common import _Tee, check_paths, count_lines, rule


# ── CONFIG ───────────────────────────────────────────────────────────────────
WEIGHTS = ROOT / "models" / "best.pt"
TEST_SPLIT = ROOT / "data" / "splits" / "test.txt"
OUT_DIR = ROOT / "outputs" / "smoothing_demo"

SIGMA = 0.08            # normalized [0,1] pixel-value noise scale
N_SAMPLES = 50
QUORUM = 0.5
CONF_THRESHOLD = 0.30
CONF_FLOOR = 0.05
SEED = 0
N_EXAMPLES = 6


def _draw(ax, boxes, color, lw=2.0, ls="-", alpha=1.0):
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
            linewidth=lw, linestyle=ls, alpha=alpha, zorder=3))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.txt"
    original = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original, fh)
        try:
            _run()
        finally:
            sys.stdout = original
    print(f"\n(full log written to {results_path})")


def _run() -> None:
    rule("CONFIG")
    print(f"  weights        : {WEIGHTS}")
    print(f"  test split     : {TEST_SPLIT}")
    print(f"  sigma / N      : {SIGMA} / {N_SAMPLES}")
    print(f"  quorum         : {QUORUM}")
    print(f"  conf threshold : {CONF_THRESHOLD}")
    check_paths(WEIGHTS, TEST_SPLIT)
    print(f"  test images    : {count_lines(TEST_SPLIT)}")

    rule("LOADING MODEL")
    base = YoloPredictor(str(WEIGHTS))
    smoothed = SmoothedTop1Predictor(
        base, N_SAMPLES, SIGMA, quorum=QUORUM, conf_floor=CONF_FLOOR, seed=SEED)
    print("  YOLO + smoothing wrapper ready")

    rule("CONTRACT CHECK")
    ds = CalibrationDataset(TEST_SPLIT)
    path0, _ = ds[0]
    out = smoothed(path0, CONF_THRESHOLD)
    ok = (out.dtype.is_floating_point and out.ndim == 2 and out.shape[1] == 5
          and out.shape[0] in (0, 1))
    print(f"  smoothed(path) -> shape {tuple(out.shape)}, dtype {out.dtype}: "
          f"{'OK' if ok else 'FAIL'} ([P,5], P in {{0,1}})")

    rule("EXAMPLE OVERLAYS (each output carries its own certificate)")
    for rank in range(min(N_EXAMPLES, len(ds))):
        path, gt = ds[rank]
        samples = smoothed.samples_for(path, CONF_THRESHOLD)
        cert = smoothed.certify(path, CONF_THRESHOLD, epsilon=0.1, tol_px=2.0)
        print(f"  {Path(path).name}: detect {samples.n_detected}/{samples.n}  "
              f"| cert detection radius {cert.detection_radius:.3f}  "
              f"| cert localization radius {cert.localization_radius_px:.3f} (tol 2px)")
        raw = base(path, CONF_THRESHOLD)        # no-noise detections
        raw_top1 = raw[int(raw[:, 4].argmax()):int(raw[:, 4].argmax()) + 1] if raw.numel() else raw

        with Image.open(path) as im:
            img = im.convert("RGB")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img)
        ax.axis("off")
        _draw(ax, samples.detected_coords, "grey", lw=0.6, alpha=0.35)  # the cloud
        _draw(ax, raw_top1, "deepskyblue", lw=2.0, ls="--")
        _draw(ax, samples.median, "orange", lw=2.4)
        _draw(ax, gt, "lime", lw=2.0)
        ax.set_title(
            f"{Path(path).name}\n"
            f"detect {samples.n_detected}/{samples.n}  |  "
            f"cert radius: detection {cert.detection_radius:.3f} / "
            f"localization {cert.localization_radius_px:.3f} (l2, 2px tol)",
            fontsize=9)
        ax.legend(handles=[
            mpatches.Patch(edgecolor="lime", facecolor="none", label="GT"),
            mpatches.Patch(edgecolor="deepskyblue", facecolor="none", label="raw top-1"),
            mpatches.Patch(edgecolor="orange", facecolor="none", label="smoothed median"),
            mpatches.Patch(edgecolor="grey", facecolor="none", label="noisy samples"),
        ], loc="upper right", fontsize=8, framealpha=0.85)
        fig.tight_layout()
        out_path = OUT_DIR / f"demo_{rank:02d}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"  saved {out_path}")

    rule()


if __name__ == "__main__":
    main()
