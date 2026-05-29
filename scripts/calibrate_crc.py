"""One end-to-end Conformal Risk Control run on the croco-detection model.

Calibrates the multiplicative margin lambda-hat on the calibration split,
then evaluates it on the held-out test split with the pixel-wise recall
loss. Everything is printed so you can see at a glance whether the
guarantee held: the calibrated lambda-hat, the test risk vs the target
alpha, the finite-sample CRC bound, the box-area inflation cost, a risk
curve, and a PASS/FAIL verdict.

Edit the CONFIG block below to change the run; takes nothing from the CLI.
Mirrors the structure of the other scripts in this folder.

    cd ~/croco_detection
    python scripts/calibrate_crc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import conformal` work when launched as `python scripts/calibrate_crc.py`
# (Python puts scripts/ on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless backend: save PNGs, no display needed
import matplotlib.pyplot as plt

from conformal.calibrator import Calibrator
from conformal.dataset import make_calibration_loader
from conformal.efficiency.box_area import total_box_area
from conformal.expansion.multiplicative import multiplicative_expansion
from conformal.loss.pixel import pixel_risk
from conformal.prediction.yolo import YoloPredictor


# ── CONFIG ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "best.pt"
SPLITS = ROOT / "data" / "splits"
CALIB_FILE = "calibration.txt"
TEST_FILE = "test.txt"

ALPHA = 0.09                  # target risk: allow <= 8% mean pixel miss
CONFIDENCE_THRESHOLD = 0.30   # methodology section 4.2
LAMBDA_RANGE = (0.0, 2.0)
BATCH_SIZE = 16
NUM_WORKERS = 4

# Paste the lambda-hat from results.txt here to skip recalibration and jump
# straight to evaluation / comparison / overlays. Leave None to recalibrate.
LAMBDA_HAT_OVERRIDE = None

MAKE_PLOTS = True
SAVE_EXAMPLES = True
N_EXAMPLES = 6                # test images to draw box overlays for

# Results (PNGs + results.txt) land in outputs/crc_<alpha%>, e.g. crc_08 for 8%.
OUTPUT_DIR = ROOT / "outputs" / f"crc_{round(ALPHA * 100):02d}"


# ── Helpers ──────────────────────────────────────────────────────────────────

def rule(title: str = "") -> None:
    if not title:
        print("\n" + "-" * 70)
    else:
        print(f"\n-- {title} " + "-" * max(0, 66 - len(title)))


def check_paths(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("ERROR -- required input(s) not found:")
        for m in missing:
            print(f"   - {m}")
        print("\nFix the path(s) in the CONFIG block and re-run.")
        sys.exit(1)


def count_lines(path: Path) -> int:
    return sum(1 for ln in path.read_text().splitlines() if ln.strip())


class _Tee:
    """Mirror writes to several streams at once (stdout + the results.txt file)."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# ── Plots ────────────────────────────────────────────────────────────────────

def save_plots(res, lam_hat: float, calib_risk: float, out_dir: Path) -> list[Path]:
    """Save the four diagnostic plots as PNGs; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 1. Risk curve R(lambda) vs alpha, with the calibrated lambda-hat marked.
    lams = [lam for lam, _ in res.risk_curve]
    risks = [risk for _, risk in res.risk_curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, risks, marker="o", ms=3, color="#1f77b4", label="test risk R(lambda)")
    ax.axhline(res.alpha, color="red", ls="--", lw=1.2, label=f"alpha = {res.alpha:.2f}")
    ax.axvline(lam_hat, color="green", ls="--", lw=1.2,
               label=f"lambda-hat = {lam_hat:.3f}")
    ax.scatter([lam_hat], [res.risk], color="green", zorder=5)
    ax.set_xlabel("lambda (expansion margin)")
    ax.set_ylabel("empirical risk")
    ax.set_title("Risk curve on the test set")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "risk_curve.png")

    # 2. Per-image loss distribution at lambda-hat.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(res.per_image_losses, bins=20, range=(0.0, 1.0),
            color="#1f77b4", edgecolor="white")
    ax.axvline(res.risk, color="red", ls="--", lw=1.2,
               label=f"mean risk = {res.risk:.3f}")
    ax.set_xlabel("per-image pixel loss (1 - recall)")
    ax.set_ylabel("number of test images")
    ax.set_title(f"Loss distribution at lambda-hat = {lam_hat:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "loss_histogram.png")

    # 3. Box-area inflation: raw vs expanded mean area.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["raw (lambda=0)", f"expanded (lambda-hat)"],
                  [res.mean_raw_efficiency, res.mean_efficiency],
                  color=["#9ecae1", "#1f77b4"])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel("mean box area (px^2)")
    title = "Box-area inflation"
    if res.inflation_ratio is not None:
        title += f"  ({res.inflation_ratio:.2f}x)"
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "area_inflation.png")

    # 4. Calibration vs test risk at lambda-hat, against alpha.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["calibration", "test"], [calib_risk, res.risk],
                  color=["#9ecae1", "#1f77b4"])
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(res.alpha, color="red", ls="--", lw=1.2, label=f"alpha = {res.alpha:.2f}")
    ax.set_ylabel("empirical risk at lambda-hat")
    ax.set_title("Generalization gap: calibration vs test")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    _save(fig, "calib_vs_test.png")

    return saved


def save_comparison_plot(res_raw, res, out_dir: Path) -> Path:
    """Grouped bars: raw (no calibration) vs calibrated, on shared [0,1] axes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = ["mean risk", "frac covered", "frac missed"]
    raw_vals = [res_raw.risk, res_raw.n_perfect / res_raw.n, res_raw.n_locked / res_raw.n]
    cal_vals = [res.risk, res.n_perfect / res.n, res.n_locked / res.n]

    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], raw_vals, w,
                label="without calibration (raw)", color="#9ecae1")
    b2 = ax.bar([i + w / 2 for i in x], cal_vals, w,
                label="with calibration (lambda-hat)", color="#1f77b4")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.axhline(res.alpha, color="red", ls="--", lw=1.2, label=f"alpha = {res.alpha:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("value (0-1)")
    ax.set_title("With vs without calibration (test set)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "with_vs_without.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _draw_boxes(ax, boxes, edgecolor, ls="-", lw=2.0, fill=False):
    """Draw xyxy boxes as rectangles on an axis."""
    import matplotlib.patches as mpatches
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=fill, facecolor=(edgecolor if fill else "none"),
            alpha=0.18 if fill else 1.0,
            edgecolor=edgecolor, linewidth=lw, linestyle=ls, zorder=3))


def _zoom_window(box_groups, img_w, img_h, pad_frac=0.6, min_pad=30):
    """Union bbox of all given boxes, padded so tiny clips fill the frame."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for boxes in box_groups:
        for b in boxes:
            xs1.append(float(b[0])); ys1.append(float(b[1]))
            xs2.append(float(b[2])); ys2.append(float(b[3]))
    if not xs1:
        return 0.0, 0.0, float(img_w), float(img_h)
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    pad = max(min_pad, pad_frac * max(x2 - x1, y2 - y1))
    return (max(0.0, x1 - pad), max(0.0, y1 - pad),
            min(float(img_w), x2 + pad), min(float(img_h), y2 + pad))


def save_example_overlays(calibrator, test_path: Path, lam_hat: float,
                          raw_losses, cal_losses, out_dir: Path,
                          n_examples: int) -> list[Path]:
    """Side-by-side WITHOUT vs WITH calibration on the test images it helped most.

    Each figure has two zoomed panels over the same crop so the boxes are not
    superposed: left shows the raw detection (blue, dashed) vs GT (green);
    right shows the calibrated/expanded box (orange, filled) vs GT. Images are
    ranked by loss drop (raw -> calibrated), so the expansion's effect is
    visible. Saved as outputs/crc_<alpha>/examples/example_NN.png.
    """
    import matplotlib.patches as mpatches
    from PIL import Image

    from conformal.dataset import CalibrationDataset

    ds = CalibrationDataset(test_path)
    n = min(len(ds), len(raw_losses), len(cal_losses))
    improvement = [raw_losses[i] - cal_losses[i] for i in range(n)]
    chosen = sorted(range(n), key=lambda i: improvement[i], reverse=True)[:n_examples]

    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    gt_patch = mpatches.Patch(edgecolor="lime", facecolor="none", label="ground truth")
    raw_patch = mpatches.Patch(edgecolor="deepskyblue", facecolor="none",
                               label="raw prediction")
    exp_patch = mpatches.Patch(edgecolor="orange", facecolor="none",
                               label=f"expanded (lambda-hat={lam_hat:.2f})")

    saved: list[Path] = []
    for rank, idx in enumerate(chosen):
        path, gt = ds[idx]
        raw = calibrator._predict_raw(path)
        expanded = calibrator._apply_expansion(raw, lam_hat)

        with Image.open(path) as im:
            img = im.convert("RGB")
        zx1, zy1, zx2, zy2 = _zoom_window([gt, raw, expanded], img.width, img.height)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
        for ax in (axL, axR):
            ax.imshow(img)
            ax.set_xlim(zx1, zx2)
            ax.set_ylim(zy2, zy1)   # inverted y: image coords run top-down
            ax.axis("off")

        # Left: without calibration (raw detector output).
        _draw_boxes(axL, raw, "deepskyblue", ls="--", lw=2.2)
        _draw_boxes(axL, gt, "lime", ls="-", lw=2.2)
        axL.set_title(f"WITHOUT calibration\npixel loss = {raw_losses[idx]:.2f}",
                      fontsize=11)
        axL.legend(handles=[gt_patch, raw_patch], loc="upper right",
                   fontsize=8, framealpha=0.85)

        # Right: with calibration (expanded box, translucent fill).
        _draw_boxes(axR, expanded, "orange", ls="-", lw=2.6, fill=True)
        _draw_boxes(axR, gt, "lime", ls="-", lw=2.2)
        axR.set_title(f"WITH calibration (lambda-hat={lam_hat:.2f})\n"
                      f"pixel loss = {cal_losses[idx]:.2f}", fontsize=11)
        axR.legend(handles=[gt_patch, exp_patch], loc="upper right",
                   fontsize=8, framealpha=0.85)

        fig.suptitle(Path(path).name, fontsize=10)
        fig.tight_layout()
        out = ex_dir / f"example_{rank:02d}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        saved.append(out)
    return saved


# ── Run ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Tee all output to OUTPUT_DIR/results.txt, then run the pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.txt"
    original_stdout = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original_stdout, fh)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    print(f"\n(full log written to {results_path})")


def _run() -> None:
    calib_path = SPLITS / CALIB_FILE
    test_path = SPLITS / TEST_FILE

    rule("CONFIG")
    print(f"  weights              : {WEIGHTS}")
    print(f"  calibration split    : {calib_path}")
    print(f"  test split           : {test_path}")
    print(f"  alpha (target risk)  : {ALPHA}")
    print(f"  confidence threshold : {CONFIDENCE_THRESHOLD}")
    print(f"  lambda search range  : {LAMBDA_RANGE}")
    print(f"  loss                 : pixel-wise recall (image_pixel_loss)")
    print(f"  expansion            : multiplicative C_lambda")

    check_paths(WEIGHTS, calib_path, test_path)
    print(f"  calibration images   : {count_lines(calib_path)}")
    print(f"  test images          : {count_lines(test_path)}")

    # ── Build the CRC pipeline ────────────────────────────────────────────────
    rule("LOADING MODEL")
    predictor = YoloPredictor(str(WEIGHTS))
    print("  YOLO model loaded")

    calibrator = Calibrator(
        prediction_fn=predictor,
        expansion_fn=multiplicative_expansion,
        risk_fn=pixel_risk,
        alpha=ALPHA,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    calib_loader = make_calibration_loader(
        calib_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = make_calibration_loader(
        test_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # ── Calibrate (or reuse a saved lambda-hat) ───────────────────────────────
    rule("CALIBRATION")
    if LAMBDA_HAT_OVERRIDE is not None:
        lam_hat = float(LAMBDA_HAT_OVERRIDE)
        print(f"  using saved lambda-hat = {lam_hat:.4f} "
              "(LAMBDA_HAT_OVERRIDE set -- skipping root search)")
    else:
        print("  running detector over calibration set + Brent root search ...")
        try:
            lam_hat = calibrator.calibrate(calib_loader, lambda_range=LAMBDA_RANGE)
        except RuntimeError as e:
            print(f"\n  CALIBRATION FAILED -- {e}")
            sys.exit(2)
        print(f"\n  lambda-hat (calibrated margin) = {lam_hat:.4f}")

    # In-sample side of the calib<->test gap.
    calib_eval = calibrator.evaluate(calib_loader, lam_hat)
    print(f"  calibration-set risk at lambda-hat = {calib_eval.risk:.4f}  "
          f"(CRC bound {calib_eval.crc_bound:.4f} <= alpha {ALPHA})")

    # ── Evaluate on the test set ──────────────────────────────────────────────
    rule("TEST EVALUATION")
    print("  running detector over test set at lambda-hat ...")
    lo, hi = LAMBDA_RANGE
    span = hi - lo
    lambdas = [lo + i * span / 20 for i in range(21)]
    res = calibrator.evaluate(
        test_loader, lam_hat,
        efficiency_fn=total_box_area,
        risk_curve_lambdas=lambdas,
    )

    # Reporting lives on EvaluationResult (summary + risk-curve table).
    print(res.summary(eff_name="box area", eff_unit="px^2"))

    # ── With vs without calibration ───────────────────────────────────────────
    # "Without" = lambda 0: the multiplicative expansion is the identity there,
    # so this is the raw detector output, evaluated with the same loss.
    rule("WITH vs WITHOUT CALIBRATION (test set)")
    print("  re-running test set at lambda=0 (raw model) ...")
    res_raw = calibrator.evaluate(test_loader, 0.0, efficiency_fn=total_box_area)
    print()
    print(res.comparison(res_raw, res, eff_name="box area",
                         baseline_label="raw", calib_label="lambda-hat"))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if MAKE_PLOTS:
        rule("PLOTS")
        saved = save_plots(res, lam_hat, calib_eval.risk, OUTPUT_DIR)
        saved.append(save_comparison_plot(res_raw, res, OUTPUT_DIR))
        for path in saved:
            print(f"  saved {path}")

    # ── Box-overlay examples on test images ───────────────────────────────────
    if SAVE_EXAMPLES:
        rule("EXAMPLE OVERLAYS (where calibration helped most)")
        examples = save_example_overlays(
            calibrator, test_path, lam_hat,
            res_raw.per_image_losses, res.per_image_losses,
            OUTPUT_DIR, N_EXAMPLES)
        for path in examples:
            print(f"  saved {path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    rule()
    print(f"  {res.verdict}")
    rule()


if __name__ == "__main__":
    main()
