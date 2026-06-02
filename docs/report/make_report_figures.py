"""Generate the cross-experiment comparison figures for the conformal report.

Every single-experiment run already drops its own diagnostic PNGs next to its
`results.txt` (risk_curve, loss_histogram, efficiency_cost, calib_vs_test,
with_vs_without, examples/). Those are embedded directly in the report.

What the report additionally needs are *cross-experiment* comparisons — the
plots that put two or more runs on the same axes so the reader can read off the
trade-off the experiment was designed to expose. Those do not exist yet; this
script builds them by parsing the `results.txt` logs (the only persisted record
of each run) and re-plotting.

    cd <repo root>
    python docs/report/make_report_figures.py

Figures land in `docs/report/figures/`. Re-run after any new calibration to
refresh them. The parser is deliberately tolerant of the two slightly different
`results.txt` formats (the early pixel run uses "mean expanded area", the later
ones use "mean box area"); if a field is missing it is left as None and the
dependent panel is skipped rather than crashing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Shared palette so the same method keeps the same colour across every figure.
C_RAW = "#9ecae1"
C_CAL = "#1f77b4"
C_ALPHA = "#d62728"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


# ── results.txt parser ───────────────────────────────────────────────────────

@dataclass
class Run:
    """Everything the report figures need from one run's results.txt."""

    name: str
    label: str
    alpha: float | None = None
    lam_hat: float | None = None
    test_risk: float | None = None
    raw_risk: float | None = None            # baseline (no-cal) risk
    frac_covered: float | None = None
    frac_missed: float | None = None
    raw_frac_missed: float | None = None
    mean_area: float | None = None
    raw_area: float | None = None
    inflation: float | None = None
    risk_curve: list[tuple[float, float]] = field(default_factory=list)
    # SeqCRC-only fields.
    alpha_cnf: float | None = None
    alpha_loc: float | None = None
    lam_cnf: float | None = None
    lam_loc: float | None = None
    t_eff: float | None = None
    n_loc: int | None = None


def _f(pattern: str, text: str, group: int = 1) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(group).replace(",", "")) if m else None


def parse_run(name: str, label: str) -> Run | None:
    """Parse `outputs/<name>/results.txt` into a Run, or None if absent."""
    path = OUTPUTS / name / "results.txt"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return None
    t = path.read_text(encoding="utf-8", errors="replace")
    run = Run(name=name, label=label)

    run.alpha = _f(r"alpha \([^)]*\)\s*[:=]?\s*([\d.]+)", t)
    run.lam_hat = _f(r"lambda-hat \(calibrated margin\)\s*=\s*([\d.]+)", t)
    run.test_risk = _f(r"test risk\s+R\(lambda\)\s*:\s*([\d.]+)", t)
    run.inflation = _f(r"inflation ratio\s*:\s*([\d.]+)x", t)
    run.mean_area = _f(r"mean (?:box area|expanded area)\s*:\s*([\d,]+)\s*px", t)
    run.raw_area = _f(r"mean raw box area\s*:\s*([\d,]+)\s*px", t)

    cov = re.search(r"images fully covered\s*:\s*(\d+)/(\d+)", t)
    if cov:
        run.frac_covered = int(cov.group(1)) / int(cov.group(2))
    mis = re.search(r"images fully missed\s*:\s*(\d+)/(\d+)", t)
    if mis:
        run.frac_missed = int(mis.group(1)) / int(mis.group(2))

    # "mean risk   <without>   <with>" line in the WITH vs WITHOUT block.
    mr = re.search(r"mean risk\s+([\d.]+)\s+([\d.]+)", t)
    if mr:
        run.raw_risk = float(mr.group(1))
    # baseline fully-missed fraction from the comparison table.
    bm = re.search(r"images fully missed\s+(\d+)/(\d+)\s+(\d+)/(\d+)", t)
    if bm:
        run.raw_frac_missed = int(bm.group(1)) / int(bm.group(2))

    # Risk curve: lines like "     0.200 |   0.0775".
    run.risk_curve = [
        (float(a), float(b))
        for a, b in re.findall(r"^\s*([\d.]+)\s*\|\s*([\d.]+)", t, re.MULTILINE)
    ]

    # SeqCRC fields (present only in seqcrc runs).
    split = re.search(r"alpha_cnf=([\d.]+)\s*\+\s*alpha_loc=([\d.]+)", t)
    if split:
        run.alpha_cnf = float(split.group(1))
        run.alpha_loc = float(split.group(2))
    run.lam_cnf = _f(r"lambda_cnf\s*=\s*([\d.]+)", t)
    run.lam_loc = _f(r"lambda_loc\s*=\s*([\d.]+)", t)
    run.t_eff = _f(r"T_eff = max\([^)]*\)\s*=\s*([\d.]+)", t)
    nl = re.search(r"survivors \(n_loc\)\s*=\s*(\d+)", t)
    if nl:
        run.n_loc = int(nl.group(1))
    return run


def _save(fig, name: str) -> None:
    fig.tight_layout()
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(ROOT)}")


# ── Figure 1 — pixel-wise loss vs 75% coverage-indicator loss ─────────────────

def fig_loss_comparison(pixel: Run, coverage: Run) -> None:
    """Same expansion (multiplicative), two losses: cost of each definition.

    Both certify the same alpha; the question is how much margin (lambda-hat)
    and how much box inflation each loss demands.
    """
    metrics = ["lambda-hat", "test risk", "inflation (x)"]
    pvals = [pixel.lam_hat, pixel.test_risk, pixel.inflation]
    cvals = [coverage.lam_hat, coverage.test_risk, coverage.inflation]
    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], pvals, w,
                label="pixel-wise recall loss", color=PALETTE[0])
    b2 = ax.bar([i + w / 2 for i in x], cvals, w,
                label="75% coverage-indicator loss", color=PALETTE[1])
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.axhline(pixel.alpha, color=C_ALPHA, ls="--", lw=1.0,
               label=f"alpha = {pixel.alpha:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_title("Loss definition: pixel-wise vs 75% coverage-indicator\n"
                 "(multiplicative expansion, alpha = 0.09)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, "fig_loss_comparison.png")


# ── Figure 2 — expansion geometry: multiplicative vs asymmetric vs additive ───

def fig_expansion_efficiency(runs: list[Run]) -> None:
    """All reach the same risk; compare the box-area cost (efficiency)."""
    labels = [r.label for r in runs]
    inflation = [r.inflation for r in runs]
    area = [r.mean_area for r in runs]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    b = axL.bar(labels, inflation, color=PALETTE[:len(runs)])
    axL.bar_label(b, fmt="%.2fx", fontsize=9)
    axL.set_ylabel("inflation ratio  (expanded / raw area)")
    axL.set_title("Cost of certifying alpha = 0.09")
    axL.grid(alpha=0.3, axis="y")
    axL.tick_params(axis="x", rotation=15)

    b2 = axR.bar(labels, area, color=PALETTE[:len(runs)])
    axR.bar_label(b2, fmt="%.0f", fontsize=9)
    if runs[0].raw_area:
        axR.axhline(runs[0].raw_area, color="grey", ls="--", lw=1.0,
                    label=f"raw mean area = {runs[0].raw_area:.0f} px^2")
        axR.legend(fontsize=8)
    axR.set_ylabel("mean box area (px^2)")
    axR.set_title("Mean prediction-set size at lambda-hat")
    axR.grid(alpha=0.3, axis="y")
    axR.tick_params(axis="x", rotation=15)
    fig.suptitle("Expansion geometry under the same loss (75% coverage)",
                 fontsize=12)
    _save(fig, "fig_expansion_efficiency.png")


# ── Figure 3 — overlaid risk curves for the geometric expansions ──────────────

def fig_expansion_riskcurves(mult: Run, asym: Run, add: Run) -> None:
    """R(lambda) for each geometric expansion.

    Multiplicative and asymmetric share a dimensionless lambda axis (fraction of
    box side); additive lambda is in pixels, so it gets its own panel.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, color in ((mult, PALETTE[0]), (asym, PALETTE[2])):
        if run and run.risk_curve:
            xs, ys = zip(*run.risk_curve)
            axL.plot(xs, ys, "o-", ms=3, color=color, label=run.label)
            if run.lam_hat:
                axL.axvline(run.lam_hat, color=color, ls=":", lw=1.0)
    axL.axhline(mult.alpha, color=C_ALPHA, ls="--", lw=1.0,
                label=f"alpha = {mult.alpha:.2f}")
    axL.set_xlabel("lambda  (fraction of box side)")
    axL.set_ylabel("test risk R(lambda)")
    axL.set_title("Multiplicative vs asymmetric")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8)

    if add and add.risk_curve:
        xs, ys = zip(*add.risk_curve)
        axR.plot(xs, ys, "o-", ms=3, color=PALETTE[1], label=add.label)
        if add.lam_hat:
            axR.axvline(add.lam_hat, color=PALETTE[1], ls=":", lw=1.0,
                        label=f"lambda-hat = {add.lam_hat:.2f} px")
        axR.set_xlim(0, max(8, (add.lam_hat or 1) * 4))
    axR.axhline(add.alpha, color=C_ALPHA, ls="--", lw=1.0,
                label=f"alpha = {add.alpha:.2f}")
    axR.set_xlabel("lambda  (pixels per side)")
    axR.set_ylabel("test risk R(lambda)")
    axR.set_title("Additive (pixel margin)")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=8)
    fig.suptitle("Risk curves by expansion geometry (75% coverage loss)",
                 fontsize=12)
    _save(fig, "fig_expansion_riskcurves.png")


# ── Figure 4 — confidence-wise vs box-size expansion ──────────────────────────

def fig_confidence_vs_geometric(conf: Run, geom: Run) -> None:
    """Two ways to spend lambda: admit more boxes vs grow existing boxes.

    Both certify the same coverage alpha; they attack different failure modes,
    so we compare what each does to the fully-missed fraction.
    """
    cats = ["raw frac missed", "frac missed\n(calibrated)", "frac covered"]
    conf_vals = [conf.raw_frac_missed, conf.frac_missed, conf.frac_covered]
    geom_vals = [geom.raw_frac_missed, geom.frac_missed, geom.frac_covered]
    x = range(len(cats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], conf_vals, w,
                label="confidence-filter expansion", color=PALETTE[3])
    b2 = ax.bar([i + w / 2 for i in x], geom_vals, w,
                label="multiplicative (box-size) expansion", color=PALETTE[0])
    ax.bar_label(b1, fmt="%.2f", fontsize=8)
    ax.bar_label(b2, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("fraction of test images")
    ax.set_title("Confidence-wise vs box-size expansion\n"
                 "(75% coverage loss, alpha = 0.09)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    _save(fig, "fig_confidence_vs_geometric.png")


# ── Figure 5 — SeqCRC alpha-allocation sweep ──────────────────────────────────

def fig_seqcrc_allocation(runs: list[Run]) -> None:
    """How the union-bound split (alpha_cnf vs alpha_loc) moves each knob.

    x-axis = fraction of the global budget given to Phase 1 (confidence).
    """
    runs = [r for r in runs if r and r.alpha_cnf is not None]
    runs.sort(key=lambda r: r.alpha_cnf)
    frac = [r.alpha_cnf / (r.alpha_cnf + r.alpha_loc) for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = [
        ("T_eff (effective conf. threshold)", [r.t_eff for r in runs], axes[0, 0]),
        ("lambda_loc (additive margin, px)", [r.lam_loc for r in runs], axes[0, 1]),
        ("survivors n_loc (of 800)", [r.n_loc for r in runs], axes[1, 0]),
        ("end-to-end test risk", [r.test_risk for r in runs], axes[1, 1]),
    ]
    for title, ys, ax in series:
        ax.plot(frac, ys, "o-", color=C_CAL, ms=6)
        for xf, yv in zip(frac, ys):
            if yv is not None:
                ax.annotate(f"{yv:.3f}" if yv < 5 else f"{yv:.0f}",
                            (xf, yv), textcoords="offset points",
                            xytext=(0, 6), fontsize=8, ha="center")
        ax.set_xlabel("Phase-1 budget fraction  alpha_cnf / alpha")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if title == "end-to-end test risk" and runs[0].alpha:
            ax.axhline(runs[0].alpha, color=C_ALPHA, ls="--", lw=1.0,
                       label=f"alpha = {runs[0].alpha:.2f}")
            ax.legend(fontsize=8)
    fig.suptitle("SeqCRC: effect of the union-bound budget split "
                 "(alpha = alpha_cnf + alpha_loc = 0.09)", fontsize=12)
    _save(fig, "fig_seqcrc_allocation.png")


# ── Figure 6 — master summary across every method ─────────────────────────────

def fig_master_summary(runs: list[Run]) -> None:
    """Validity (risk <= alpha) and efficiency (inflation) for all methods."""
    runs = [r for r in runs if r and r.test_risk is not None]
    labels = [r.label for r in runs]
    risks = [r.test_risk for r in runs]
    raws = [r.raw_risk for r in runs]
    alpha = next((r.alpha for r in runs if r.alpha), 0.09)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(runs))
    w = 0.38
    b1 = axL.bar([i - w / 2 for i in x], raws, w, label="without calibration",
                 color=C_RAW)
    b2 = axL.bar([i + w / 2 for i in x], risks, w, label="at lambda-hat",
                 color=C_CAL)
    axL.bar_label(b2, fmt="%.3f", fontsize=8)
    axL.axhline(alpha, color=C_ALPHA, ls="--", lw=1.2, label=f"alpha = {alpha:.2f}")
    axL.set_xticks(list(x))
    axL.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axL.set_ylabel("test risk")
    axL.set_title("Validity: every method lands at or below alpha")
    axL.grid(alpha=0.3, axis="y")
    axL.legend(fontsize=8)

    # Efficiency scatter: inflation vs risk (only runs that report inflation).
    eff = [r for r in runs if r.inflation is not None]
    for i, r in enumerate(eff):
        axR.scatter(r.inflation, r.test_risk, s=90, color=PALETTE[i % len(PALETTE)],
                    zorder=3, label=r.label)
        axR.annotate(r.label, (r.inflation, r.test_risk),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    axR.axhline(alpha, color=C_ALPHA, ls="--", lw=1.0, label=f"alpha = {alpha:.2f}")
    axR.set_xlabel("inflation ratio  (lower = cheaper)")
    axR.set_ylabel("test risk  (lower = safer)")
    axR.set_title("Efficiency vs validity trade-off")
    axR.grid(alpha=0.3)
    axR.legend(fontsize=7, loc="upper right")
    _save(fig, "fig_master_summary.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Reading runs from {OUTPUTS}")

    # Single-knob CRC runs (all alpha = 0.09).
    pixel_mult = parse_run("crc_09", "Pixel . multiplicative")
    cov_mult = parse_run("crc_cov_09", "Coverage . multiplicative")
    cov_asym = parse_run("crc_asym_cov_09_06", "Coverage . asymmetric")
    cov_add = parse_run("crc_add_cov_09", "Coverage . additive")
    cov_conf = parse_run("crc_conf_09", "Coverage . confidence")

    # SeqCRC runs (alpha = 0.09, different union-bound splits).
    seq = [
        parse_run("seqcrc_09", "50/50"),
        parse_run("seqcrc_09_top1", "50/50 (top-1)"),
        parse_run("seqcrc_09_top1_2", "75/25 (top-1)"),
        parse_run("seqcrc_09_top1_3", "35/65 (top-1)"),
    ]
    seq_main = next(r for r in seq if r and r.name == "seqcrc_09")
    seq_main.label = "SeqCRC (two-phase)"

    print("Building figures:")
    if pixel_mult and cov_mult:
        fig_loss_comparison(pixel_mult, cov_mult)
    geom = [r for r in (cov_mult, cov_asym, cov_add) if r]
    if len(geom) == 3:
        fig_expansion_efficiency(geom)
        fig_expansion_riskcurves(cov_mult, cov_asym, cov_add)
    if cov_conf and cov_mult:
        fig_confidence_vs_geometric(cov_conf, cov_mult)
    fig_seqcrc_allocation(seq)

    summary_runs = [pixel_mult, cov_mult, cov_asym, cov_add, cov_conf, seq_main]
    fig_master_summary([r for r in summary_runs if r])
    print("Done.")


if __name__ == "__main__":
    main()
