"""Plotting / overlay helpers for the two-step SeqCRC driver.

Kept out of `calibrate_seqcrc.py` so the driver stays a thin CONFIG + flow
script. Two outputs:

  - `save_risk_curves`: the two empirical risk curves behind the calibrated
    parameters -- R_cnf vs lambda_cnf and R_loc(lambda_cnf_plus, .) vs
    lambda_loc -- each against its target alpha and calibrated lambda.
  - `save_overlays`: the calibrated Gamma_loc field (orange) vs ground truth
    (green) on the test frames, zoomed to the clips.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save PNGs, no display
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from PIL import Image

from conformal.dataset import CalibrationDataset
from conformal.seqcrc import confidence_risk, localization_risk
from conformal.seqcrc.config import SeqCRCConfig

from crc_common import _draw_boxes, _zoom_window


def save_risk_curves(
    predictions: list[torch.Tensor],
    ground_truth: list[torch.Tensor],
    lambda_cnf: float,
    lambda_loc: float,
    cfg: SeqCRCConfig,
    out_dir: Path,
) -> list[Path]:
    """Save the confidence and localization risk-curve PNGs; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # R_cnf(lambda_cnf) sweep over the confidence parameter space [0, 1].
    cnf_lams = [i / 20 for i in range(21)]
    cnf_risks = [confidence_risk(predictions, ground_truth, lam, cfg)
                 for lam in cnf_lams]
    saved.append(_curve(
        cnf_lams, cnf_risks, cfg.alpha_cnf, lambda_cnf,
        xlabel="lambda_cnf", title="Confidence risk R_cnf (test set)",
        out=out_dir / "risk_curve_cnf.png"))

    # R_loc(lambda_cnf_plus, lambda_loc) sweep over [0, lambda_bar_loc].
    span = cfg.lambda_bar_loc
    loc_lams = [span * i / 20 for i in range(21)]
    loc_risks = [localization_risk(predictions, ground_truth, lambda_cnf, lam, cfg)
                 for lam in loc_lams]
    saved.append(_curve(
        loc_lams, loc_risks, cfg.alpha_loc, lambda_loc,
        xlabel="lambda_loc", title="Localization risk R_loc (test set)",
        out=out_dir / "risk_curve_loc.png"))
    return saved


def _curve(lams, risks, alpha, lam_hat, *, xlabel, title, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lams, risks, marker="o", ms=3, color="#1f77b4", label="test risk")
    ax.axhline(alpha, color="red", ls="--", lw=1.2, label=f"alpha = {alpha:.3f}")
    ax.axvline(lam_hat, color="green", ls="--", lw=1.2,
               label=f"lambda-hat = {lam_hat:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("empirical risk")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_overlays(
    inferencer,
    test_path: Path,
    out_dir: Path,
    n_examples: int = 6,
) -> list[Path]:
    """Draw the calibrated Gamma_loc field vs ground truth on a few test frames."""
    ds = CalibrationDataset(test_path)
    # Prefer frames that actually have a clip, so the overlay shows coverage.
    indices = [i for i in range(len(ds)) if ds[i][1].numel() > 0][:n_examples]
    if not indices:
        indices = list(range(min(n_examples, len(ds))))

    ex_dir = out_dir / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    gt_patch = mpatches.Patch(edgecolor="lime", facecolor="none", label="ground truth")
    field_patch = mpatches.Patch(edgecolor="orange", facecolor="none",
                                 label="SeqCRC field (Gamma_loc)")

    saved: list[Path] = []
    for rank, idx in enumerate(indices):
        path, gt = ds[idx]
        field = inferencer(path)
        with Image.open(path) as im:
            img = im.convert("RGB")
        zx1, zy1, zx2, zy2 = _zoom_window([gt, field], img.width, img.height)

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.imshow(img)
        ax.set_xlim(zx1, zx2)
        ax.set_ylim(zy2, zy1)        # inverted y: image coords run top-down
        ax.axis("off")
        _draw_boxes(ax, field, "orange", ls="-", lw=2.6, fill=True)
        _draw_boxes(ax, gt, "lime", ls="-", lw=2.2)
        ax.set_title(Path(path).name, fontsize=10)
        ax.legend(handles=[gt_patch, field_patch], loc="upper right",
                  fontsize=8, framealpha=0.85)
        fig.tight_layout()
        out = ex_dir / f"example_{rank:02d}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        saved.append(out)
    return saved
