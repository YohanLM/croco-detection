"""Shared calibrate -> evaluate -> report pipeline for the CRC scripts.

Each pipeline script (coverage / confidence / ...) is a thin CONFIG block
that calls `run_pipeline(...)` with its own loss, expansion, efficiency
metric, and baseline semantics. Keeping the reporting and plotting in one
place avoids duplicating ~300 lines per pipeline; everything that differs
between pipelines is a parameter.

Two notions are deliberately generic here:

  - The "without calibration" baseline is whatever lambda reproduces the
    model's normal operating point. For the multiplicative expansion that is
    lambda=0 (the identity -> raw detector). For the confidence-filter
    expansion lambda=0 is the EMPTY set, so the baseline is the lambda that
    reproduces the standard confidence threshold instead. Pass it via
    `baseline_lambda`.

  - The efficiency metric (box area vs box count) is injected, so the cost
    plot is labelled per pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: save PNGs, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from conformal.calibrator import Calibrator
from conformal.dataset import CalibrationDataset, make_calibration_loader


# ── Small helpers ─────────────────────────────────────────────────────────────

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


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(res, res_raw, lam_hat, calib_risk, out_dir,
               eff_name, eff_unit, baseline_label, calib_label):
    """Save the diagnostic plots as PNGs; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    def _save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 1. Risk curve R(lambda) vs alpha, with lambda-hat marked.
    if res.risk_curve:
        lams = [lam for lam, _ in res.risk_curve]
        risks = [risk for _, risk in res.risk_curve]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(lams, risks, marker="o", ms=3, color="#1f77b4",
                label="test risk R(lambda)")
        ax.axhline(res.alpha, color="red", ls="--", lw=1.2,
                   label=f"alpha = {res.alpha:.2f}")
        ax.axvline(lam_hat, color="green", ls="--", lw=1.2,
                   label=f"lambda-hat = {lam_hat:.3f}")
        ax.scatter([lam_hat], [res.risk], color="green", zorder=5)
        ax.set_xlabel("lambda")
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
    ax.set_xlabel("per-image loss")
    ax.set_ylabel("number of test images")
    ax.set_title(f"Loss distribution at lambda-hat = {lam_hat:.3f}")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, "loss_histogram.png")

    # 3. Efficiency cost: baseline vs at lambda-hat.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    base_eff = res_raw.mean_efficiency
    cal_eff = res.mean_efficiency
    bars = ax.bar([baseline_label, calib_label], [base_eff, cal_eff],
                  color=["#9ecae1", "#1f77b4"])
    ax.bar_label(bars, fmt="%.0f")
    ax.set_ylabel(f"mean {eff_name} ({eff_unit})")
    title = f"{eff_name} cost"
    if base_eff:
        title += f"  ({cal_eff / base_eff:.2f}x)"
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "efficiency_cost.png")

    # 4. Calibration vs test risk at lambda-hat, against alpha.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["calibration", "test"], [calib_risk, res.risk],
                  color=["#9ecae1", "#1f77b4"])
    ax.bar_label(bars, fmt="%.3f")
    ax.axhline(res.alpha, color="red", ls="--", lw=1.2,
               label=f"alpha = {res.alpha:.2f}")
    ax.set_ylabel("empirical risk at lambda-hat")
    ax.set_title("Generalization gap: calibration vs test")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    _save(fig, "calib_vs_test.png")

    # 5. With vs without calibration (grouped bars).
    cats = ["mean risk", "frac covered", "frac missed"]
    raw_vals = [res_raw.risk, res_raw.n_perfect / res_raw.n,
                res_raw.n_locked / res_raw.n]
    cal_vals = [res.risk, res.n_perfect / res.n, res.n_locked / res.n]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], raw_vals, w,
                label=f"without ({baseline_label})", color="#9ecae1")
    b2 = ax.bar([i + w / 2 for i in x], cal_vals, w,
                label=f"with ({calib_label})", color="#1f77b4")
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.axhline(res.alpha, color="red", ls="--", lw=1.2,
               label=f"alpha = {res.alpha:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("value (0-1)")
    ax.set_title("With vs without calibration (test set)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, "with_vs_without.png")

    return saved


def _draw_boxes(ax, boxes, edgecolor, ls="-", lw=2.0, fill=False):
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


def save_example_overlays(calibrator, test_path, lam_hat, baseline_lambda,
                          base_losses, cal_losses, out_dir, n_examples,
                          baseline_label, calib_label):
    """Side-by-side WITHOUT vs WITH calibration on the images it helped most.

    Both panels show the same zoomed crop so the boxes are not superposed:
    left is the baseline operating point (`baseline_lambda`), right is
    lambda-hat. GT is green; the prediction set is blue on the left, orange
    (translucent fill) on the right. Images are ranked by loss drop.
    """
    ds = CalibrationDataset(test_path)
    n = min(len(ds), len(base_losses), len(cal_losses))
    improvement = [base_losses[i] - cal_losses[i] for i in range(n)]
    chosen = sorted(range(n), key=lambda i: improvement[i], reverse=True)[:n_examples]

    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)

    gt_patch = mpatches.Patch(edgecolor="lime", facecolor="none", label="ground truth")
    base_patch = mpatches.Patch(edgecolor="deepskyblue", facecolor="none",
                                label=f"prediction ({baseline_label})")
    cal_patch = mpatches.Patch(edgecolor="orange", facecolor="none",
                               label=f"prediction ({calib_label})")

    saved = []
    for rank, idx in enumerate(chosen):
        path, gt = ds[idx]
        raw = calibrator._predict_raw(path)
        base_set = calibrator._apply_expansion(raw, baseline_lambda)
        cal_set = calibrator._apply_expansion(raw, lam_hat)

        with Image.open(path) as im:
            img = im.convert("RGB")
        zx1, zy1, zx2, zy2 = _zoom_window([gt, base_set, cal_set],
                                          img.width, img.height)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
        for ax in (axL, axR):
            ax.imshow(img)
            ax.set_xlim(zx1, zx2)
            ax.set_ylim(zy2, zy1)   # inverted y: image coords run top-down
            ax.axis("off")

        _draw_boxes(axL, base_set, "deepskyblue", ls="--", lw=2.2)
        _draw_boxes(axL, gt, "lime", ls="-", lw=2.2)
        axL.set_title(f"WITHOUT calibration ({baseline_label})\n"
                      f"loss = {base_losses[idx]:.2f}", fontsize=11)
        axL.legend(handles=[gt_patch, base_patch], loc="upper right",
                   fontsize=8, framealpha=0.85)

        _draw_boxes(axR, cal_set, "orange", ls="-", lw=2.6, fill=True)
        _draw_boxes(axR, gt, "lime", ls="-", lw=2.2)
        axR.set_title(f"WITH calibration ({calib_label})\n"
                      f"loss = {cal_losses[idx]:.2f}", fontsize=11)
        axR.legend(handles=[gt_patch, cal_patch], loc="upper right",
                   fontsize=8, framealpha=0.85)

        fig.suptitle(Path(path).name, fontsize=10)
        fig.tight_layout()
        out = ex_dir / f"example_{rank:02d}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        saved.append(out)
    return saved


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(*, weights, calib_path, test_path,
                 risk_fn, expansion_fn, efficiency_fn,
                 alpha, confidence_threshold, lambda_range, output_dir,
                 baseline_lambda, baseline_label, calib_label,
                 loss_name, expansion_name, eff_name, eff_unit,
                 report_effective_threshold=False, extra_metrics=None,
                 batch_size=16, num_workers=4, n_examples=6,
                 make_plots=True, save_examples=True, lambda_hat_override=None):
    """Calibrate lambda-hat, evaluate it on test, compare to baseline, plot.

    All console output is mirrored to `output_dir/results.txt`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.txt"
    original_stdout = sys.stdout
    with open(results_path, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(original_stdout, fh)
        try:
            _run(weights=weights, calib_path=calib_path, test_path=test_path,
                 risk_fn=risk_fn, expansion_fn=expansion_fn,
                 efficiency_fn=efficiency_fn, alpha=alpha,
                 confidence_threshold=confidence_threshold,
                 lambda_range=lambda_range, output_dir=output_dir,
                 baseline_lambda=baseline_lambda, baseline_label=baseline_label,
                 calib_label=calib_label, loss_name=loss_name,
                 expansion_name=expansion_name, eff_name=eff_name,
                 eff_unit=eff_unit,
                 report_effective_threshold=report_effective_threshold,
                 extra_metrics=extra_metrics,
                 batch_size=batch_size, num_workers=num_workers,
                 n_examples=n_examples, make_plots=make_plots,
                 save_examples=save_examples,
                 lambda_hat_override=lambda_hat_override)
        finally:
            sys.stdout = original_stdout
    print(f"\n(full log written to {results_path})")


def _run(*, weights, calib_path, test_path, risk_fn, expansion_fn,
         efficiency_fn, alpha, confidence_threshold, lambda_range, output_dir,
         baseline_lambda, baseline_label, calib_label, loss_name,
         expansion_name, eff_name, eff_unit, report_effective_threshold,
         extra_metrics, batch_size, num_workers, n_examples, make_plots,
         save_examples, lambda_hat_override):
    rule("CONFIG")
    print(f"  weights              : {weights}")
    print(f"  calibration split    : {calib_path}")
    print(f"  test split           : {test_path}")
    print(f"  alpha (target risk)  : {alpha}")
    print(f"  confidence threshold : {confidence_threshold}")
    print(f"  lambda search range  : {lambda_range}")
    print(f"  loss                 : {loss_name}")
    print(f"  expansion            : {expansion_name}")
    print(f"  baseline (no cal.)   : lambda={baseline_lambda} ({baseline_label})")

    check_paths(weights, calib_path, test_path)
    print(f"  calibration images   : {count_lines(calib_path)}")
    print(f"  test images          : {count_lines(test_path)}")

    rule("LOADING MODEL")
    from conformal.prediction.yolo import YoloPredictor
    predictor = YoloPredictor(str(weights))
    print("  YOLO model loaded")

    calibrator = Calibrator(
        prediction_fn=predictor,
        expansion_fn=expansion_fn,
        risk_fn=risk_fn,
        alpha=alpha,
        confidence_threshold=confidence_threshold,
    )
    calib_loader = make_calibration_loader(
        calib_path, batch_size=batch_size, num_workers=num_workers)
    test_loader = make_calibration_loader(
        test_path, batch_size=batch_size, num_workers=num_workers)

    # ── Calibrate (or reuse a saved lambda-hat) ───────────────────────────────
    rule("CALIBRATION")
    if lambda_hat_override is not None:
        lam_hat = float(lambda_hat_override)
        print(f"  using saved lambda-hat = {lam_hat:.4f} (override set)")
    else:
        print("  running detector over calibration set + Brent root search ...")
        try:
            lam_hat = calibrator.calibrate(calib_loader, lambda_range=lambda_range)
        except RuntimeError as e:
            print(f"\n  CALIBRATION FAILED -- {e}")
            sys.exit(2)
        print(f"\n  lambda-hat (calibrated margin) = {lam_hat:.4f}")

    if report_effective_threshold:
        eff_thr = max(confidence_threshold, 1.0 - lam_hat)
        print(f"  -> effective confidence threshold = max({confidence_threshold}, "
              f"1 - {lam_hat:.4f}) = {eff_thr:.4f}")

    calib_eval = calibrator.evaluate(calib_loader, lam_hat)
    print(f"  calibration-set risk at lambda-hat = {calib_eval.risk:.4f}  "
          f"(CRC bound {calib_eval.crc_bound:.4f} <= alpha {alpha})")

    # ── Evaluate on the test set (at lambda-hat, with a risk-curve sweep) ──────
    rule("TEST EVALUATION")
    print("  running detector over test set at lambda-hat ...")
    lo, hi = lambda_range
    span = hi - lo
    lambdas = [lo + i * span / 20 for i in range(21)]
    res = calibrator.evaluate(
        test_loader, lam_hat, efficiency_fn=efficiency_fn,
        risk_curve_lambdas=lambdas, extra_metrics=extra_metrics)

    # Reporting lives on EvaluationResult; we just label the efficiency metric.
    print(res.summary(eff_name=eff_name, eff_unit=eff_unit))

    # ── Baseline (without calibration) ────────────────────────────────────────
    rule("WITH vs WITHOUT CALIBRATION (test set)")
    print(f"  re-running test set at the baseline (lambda={baseline_lambda}) ...")
    res_raw = calibrator.evaluate(test_loader, baseline_lambda,
                                  efficiency_fn=efficiency_fn,
                                  extra_metrics=extra_metrics)
    print()
    print(res.comparison(res_raw, res, eff_name=eff_name,
                         baseline_label=baseline_label, calib_label=calib_label))

    # ── Plots + overlays ──────────────────────────────────────────────────────
    if make_plots:
        rule("PLOTS")
        saved = save_plots(res, res_raw, lam_hat, calib_eval.risk, output_dir,
                           eff_name, eff_unit, baseline_label, calib_label)
        for path in saved:
            print(f"  saved {path}")

    if save_examples:
        rule("EXAMPLE OVERLAYS (where calibration helped most)")
        examples = save_example_overlays(
            calibrator, test_path, lam_hat, baseline_lambda,
            res_raw.per_image_losses, res.per_image_losses,
            output_dir, n_examples, baseline_label, calib_label)
        for path in examples:
            print(f"  saved {path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    rule()
    print(f"  {res.verdict}")
    rule()
