"""Characterize the median-smoothing add-on across noise scales (no calibration).

Sweeps `sigma`, runs the full per-image metric suite over the test split, and
emits the robustness-vs-efficiency picture the methodology asks for:

  - accuracy degrades with sigma          -> iou_vs_sigma.png
  - the box gets jumpier with sigma       -> jitter_vs_sigma.png
  - detections start to blink out          -> detection_rate_vs_sigma.png
  - certified stability grows with sigma   -> certified_radius_vs_sigma.png
  - Monte-Carlo error shrinks with N       -> mc_se_vs_N.png

plus a full metric table (also written to results.txt). This is pure evaluation:
it never calibrates lambda, but the `SmoothedTop1Predictor` it builds is the same
object you would later hand to the `Calibrator`.

    cd ~/croco_detection
    python scripts/smoothing/evaluate_smoothing.py

Output lands in outputs/smoothing_eval/. Set MAX_IMAGES=None for the full split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conformal.dataset import CalibrationDataset
from conformal.prediction.yolo import YoloPredictor
from conformal.smoothing import metrics

from crc_common import _Tee, check_paths, count_lines, rule


# ── CONFIG ───────────────────────────────────────────────────────────────────
WEIGHTS = ROOT / "models" / "best.pt"
TEST_SPLIT = ROOT / "data" / "splits" / "test.txt"
OUT_DIR = ROOT / "outputs" / "smoothing_eval"

DEVICE = "cuda"          # "cuda" / "mps" / "cpu"

SIGMAS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20]
N_SAMPLES = 50
QUORUM = 0.5
CONF_THRESHOLD = 0.30
CONF_FLOOR = 0.05
SEED = 0

CERT_EPSILON = 0.10          # l2 attack radius the certified IoU is reported at
CERT_IOU_TARGET = 0.50       # IoU floor used for the certified radius
CERT_CONF = 0.0              # >0 -> finite-sample confidence-corrected band

MC_SIGMA = 0.08              # sigma for the N-curve
MC_N_VALUES = [5, 10, 20, 40, 80]

MAX_IMAGES = 100             # cap for a quick run; set None to use the whole split


# Metrics to render as "metric vs sigma" line plots: (key, ylabel, filename).
_SIGMA_PLOTS = [
    ("smoothed_iou", "mean IoU(smoothed, GT)", "iou_vs_sigma.png"),
    ("box_jitter_px", "mean box jitter (px)", "jitter_vs_sigma.png"),
    ("detection_rate", "mean detection rate", "detection_rate_vs_sigma.png"),
    ("cert_localization_radius_px", "mean certified localization radius (l2, tol=2px)",
     "certified_radius_vs_sigma.png"),
    ("cert_detection_radius", "mean certified detection radius (l2)",
     "certified_detection_radius_vs_sigma.png"),
]


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
    print(f"  device         : {DEVICE}")
    print(f"  sigmas         : {SIGMAS}")
    print(f"  N samples      : {N_SAMPLES}")
    print(f"  quorum         : {QUORUM}")
    print(f"  conf threshold : {CONF_THRESHOLD}")
    print(f"  cert epsilon   : {CERT_EPSILON}  (IoU target {CERT_IOU_TARGET})")
    print(f"  max images     : {MAX_IMAGES if MAX_IMAGES is not None else 'all'}")
    check_paths(WEIGHTS, TEST_SPLIT)
    print(f"  test images    : {count_lines(TEST_SPLIT)}")

    rule("LOADING MODEL")
    base = YoloPredictor(str(WEIGHTS))
    print("  YOLO model loaded")

    ds = CalibrationDataset(TEST_SPLIT)
    items = [ds[i] for i in range(len(ds))]
    if MAX_IMAGES is not None:
        items = items[:MAX_IMAGES]
    print(f"  evaluating on {len(items)} images")

    # ── Sigma sweep: the full metric table ────────────────────────────────────
    rule("SIGMA SWEEP")
    print(f"  running {len(SIGMAS)} sigmas x {len(items)} images x N={N_SAMPLES} ...")
    table = metrics.sweep(
        base, items, SIGMAS, N_SAMPLES,
        confidence_threshold=CONF_THRESHOLD, quorum=QUORUM, conf_floor=CONF_FLOOR,
        seed=SEED, cert_epsilon=CERT_EPSILON, iou_target=CERT_IOU_TARGET,
        cert_conf=CERT_CONF, device=DEVICE,
    )
    _print_table(table)

    # ── Monte-Carlo SE vs N ───────────────────────────────────────────────────
    rule(f"MONTE-CARLO SE vs N  (sigma={MC_SIGMA})")
    se_curve = metrics.mc_se_vs_n(
        base, items, MC_SIGMA, MC_N_VALUES,
        confidence_threshold=CONF_THRESHOLD, quorum=QUORUM, conf_floor=CONF_FLOOR,
        seed=SEED, device=DEVICE,
    )
    for n, se in se_curve.items():
        print(f"     N={n:>4}  ->  mean median SE = {se:.3f} px")

    # ── Plots ─────────────────────────────────────────────────────────────────
    rule("PLOTS")
    saved = _save_plots(table, se_curve)
    for p in saved:
        print(f"  saved {p}")

    rule()


def _print_table(table: dict[float, dict[str, float]]) -> None:
    sigmas = sorted(table)
    keys = list(next(iter(table.values())).keys())
    print(f"\n  {'metric':<22}" + "".join(f"{f's={s:g}':>12}" for s in sigmas))
    print("  " + "-" * (22 + 12 * len(sigmas)))
    for k in keys:
        row = "".join(f"{table[s][k]:>12.4f}" for s in sigmas)
        print(f"  {k:<22}{row}")


def _save_plots(table, se_curve) -> list[Path]:
    sigmas = sorted(table)
    saved: list[Path] = []

    for key, ylabel, fname in _SIGMA_PLOTS:
        ys = [table[s][key] for s in sigmas]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(sigmas, ys, marker="o", color="#1f77b4")
        ax.set_xlabel("noise scale sigma")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs sigma")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = OUT_DIR / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # MC SE vs N (with the 1/sqrt(N) reference).
    ns = sorted(se_curve)
    ses = [se_curve[n] for n in ns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, ses, marker="o", color="#d62728", label="measured")
    if ses and ses[0] == ses[0]:  # not NaN
        ref = [ses[0] * (ns[0] / n) ** 0.5 for n in ns]
        ax.plot(ns, ref, ls="--", color="grey", label="1/sqrt(N) reference")
    ax.set_xlabel("N (Monte-Carlo samples)")
    ax.set_ylabel("mean median SE (px)")
    ax.set_title(f"Monte-Carlo error vs N (sigma={MC_SIGMA})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "mc_se_vs_N.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    return saved


if __name__ == "__main__":
    main()
