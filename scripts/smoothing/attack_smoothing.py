"""Empirical adversarial robustness: raw vs smoothed under l2-PGD (methodology §5).

For each test image we craft an l2-bounded PGD perturbation that tries to make
the clip vanish, then compare how much localization survives:

  - RAW   top-1 IoU(GT), clean vs adversarial
  - SMOOTH top-1 IoU(GT), clean vs adversarial

The headline number is the IoU *drop* under attack for each: smoothing passes the
test if its drop is smaller than the raw detector's (the median votes out the
attack's per-copy damage). Worst-attacked cases are saved as side-by-side
overlays.

This is the empirical complement to the certificate in
`conformal.smoothing.certificate`. It has no calibration dependency; if you set
LAMBDA_HAT (an additive pixel margin from a prior CRC run), it additionally
re-measures the 75%-coverage risk on the attacked set as a sanity check that the
conformal envelope still holds under perturbation.

    cd ~/croco_detection
    python scripts/smoothing/attack_smoothing.py

Output lands in outputs/smoothing_attack/. PGD on CPU is slow — keep MAX_IMAGES
small unless you have a GPU.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "calibration"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from conformal.dataset import CalibrationDataset
from conformal.expansion.additive import additive_expansion
from conformal.loss.coverage import image_coverage_indicator_loss
from conformal.prediction.yolo import YoloPredictor
from conformal.smoothing.attack import pgd_l2
from conformal.smoothing.predictor import (
    SmoothedTop1Predictor,
    collect_samples_tensor,
    load_image_chw01,
)

from crc_common import _Tee, check_paths, count_lines, rule


# ── CONFIG ───────────────────────────────────────────────────────────────────
WEIGHTS = ROOT / "models" / "best.pt"
TEST_SPLIT = ROOT / "data" / "splits" / "test.txt"

ATTACK_SIZE = 640            # square size images are resized to (divisible by 32)
EPSILON = 0.10               # total-l2 perturbation budget (keep <= cert radius ~0.134)
OUT_DIR = ROOT / "outputs" / f"smoothing_attack_eps{EPSILON}"
PGD_STEPS = 10

SIGMA = 0.08                 # smoothing noise scale
N_SAMPLES = 50
QUORUM = 0.5
CONF_THRESHOLD = 0.30
CONF_FLOOR = 0.05
SEED = 0

LAMBDA_HAT = None            # set to a calibrated additive margin to re-check CRC risk
N_OVERLAYS = 6
MAX_IMAGES = 20              # PGD is expensive; raise on a GPU


def _resize(img: torch.Tensor, size: int) -> torch.Tensor:
    """Resize `[3, H, W]` to `[3, size, size]` (bilinear)."""
    return F.interpolate(img.unsqueeze(0), size=(size, size),
                         mode="bilinear", align_corners=False)[0]


def _scale_gt(gt: torch.Tensor, w: int, h: int, size: int) -> torch.Tensor:
    """Scale GT pixel-xyxy from original `(w, h)` to a `size x size` frame."""
    if gt.numel() == 0:
        return gt
    out = gt.clone()
    out[:, [0, 2]] *= size / w
    out[:, [1, 3]] *= size / h
    return out


def _top1(preds: torch.Tensor) -> torch.Tensor:
    if preds.numel() == 0:
        return preds
    j = int(preds[:, 4].argmax())
    return preds[j:j + 1]


def _iou(a: torch.Tensor, gt: torch.Tensor) -> float:
    """IoU of a single box `[>=4]` vs the best GT; 0 if box or GT empty."""
    if a.numel() == 0 or gt.numel() == 0:
        return 0.0
    best = 0.0
    for g in gt.reshape(-1, gt.shape[-1]):
        x1 = max(float(a[0, 0]), float(g[0])); y1 = max(float(a[0, 1]), float(g[1]))
        x2 = min(float(a[0, 2]), float(g[2])); y2 = min(float(a[0, 3]), float(g[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = float(a[0, 2] - a[0, 0]) * float(a[0, 3] - a[0, 1])
        ga = float(g[2] - g[0]) * float(g[3] - g[1])
        union = aa + ga - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return best


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
    print(f"  attack size    : {ATTACK_SIZE}  | epsilon(l2) {EPSILON} | steps {PGD_STEPS}")
    print(f"  sigma / N      : {SIGMA} / {N_SAMPLES}")
    print(f"  max images     : {MAX_IMAGES}")
    check_paths(WEIGHTS, TEST_SPLIT)
    print(f"  test images    : {count_lines(TEST_SPLIT)}")

    rule("LOADING MODEL")
    base = YoloPredictor(str(WEIGHTS))
    det_module = base.model.model        # raw DetectionModel (nn.Module)
    print("  YOLO model + raw detection module ready")

    ds = CalibrationDataset(TEST_SPLIT)
    n = min(MAX_IMAGES, len(ds))

    raw_clean, raw_adv, sm_clean, sm_adv = [], [], [], []
    cov_clean, cov_adv = [], []          # coverage loss (only if LAMBDA_HAT set)
    per_image = []                       # (idx, raw_drop) for overlay ranking

    rule("ATTACKING")
    for i in range(n):
        path, gt = ds[i]
        img = load_image_chw01(path)
        _, h, w = img.shape
        img_r = _resize(img, ATTACK_SIZE)
        gt_r = _scale_gt(gt, w, h, ATTACK_SIZE)

        adv = pgd_l2(det_module, img_r.unsqueeze(0), epsilon=EPSILON,
                     steps=PGD_STEPS)[0].cpu()

        # Raw top-1 on clean vs adversarial.
        rc = _iou(_top1(base.predict_arrays(img_r.unsqueeze(0), CONF_THRESHOLD)[0]), gt_r)
        ra = _iou(_top1(base.predict_arrays(adv.unsqueeze(0), CONF_THRESHOLD)[0]), gt_r)
        # Smoothed top-1 on clean vs adversarial.
        sc_box = collect_samples_tensor(base, img_r, SIGMA, N_SAMPLES,
                                        conf_floor=CONF_FLOOR, quorum=QUORUM,
                                        conf_threshold=CONF_THRESHOLD).median
        sa_box = collect_samples_tensor(base, adv, SIGMA, N_SAMPLES,
                                        conf_floor=CONF_FLOOR, quorum=QUORUM,
                                        conf_threshold=CONF_THRESHOLD).median
        sc = _iou(sc_box, gt_r)
        sa = _iou(sa_box, gt_r)

        raw_clean.append(rc); raw_adv.append(ra)
        sm_clean.append(sc); sm_adv.append(sa)
        per_image.append((i, rc - ra, gt_r, img_r, adv, sc_box, sa_box))

        if LAMBDA_HAT is not None:
            cov_clean.append(image_coverage_indicator_loss(
                additive_expansion(sc_box, LAMBDA_HAT, CONF_THRESHOLD), gt_r))
            cov_adv.append(image_coverage_indicator_loss(
                additive_expansion(sa_box, LAMBDA_HAT, CONF_THRESHOLD), gt_r))

        print(f"  [{i + 1:>3}/{n}] raw IoU {rc:.2f}->{ra:.2f}   "
              f"smooth IoU {sc:.2f}->{sa:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rule("ROBUSTNESS SUMMARY")
    rc_m, ra_m = statistics.mean(raw_clean), statistics.mean(raw_adv)
    sc_m, sa_m = statistics.mean(sm_clean), statistics.mean(sm_adv)
    print(f"  {'':<18}{'clean':>10}{'adversarial':>14}{'drop':>10}")
    print("  " + "-" * 52)
    print(f"  {'raw top-1 IoU':<18}{rc_m:>10.3f}{ra_m:>14.3f}{rc_m - ra_m:>10.3f}")
    print(f"  {'smoothed IoU':<18}{sc_m:>10.3f}{sa_m:>14.3f}{sc_m - sa_m:>10.3f}")
    gap = (rc_m - ra_m) - (sc_m - sa_m)
    print(f"\n  robustness gain (raw drop - smoothed drop): {gap:+.3f} IoU")
    verdict = "smoothing IS more robust" if gap > 0 else "no robustness gain here"
    print(f"  --> {verdict} under l2={EPSILON} PGD.")

    if LAMBDA_HAT is not None and cov_adv:
        rule("CRC RISK UNDER ATTACK")
        print(f"  lambda-hat (additive margin) : {LAMBDA_HAT}")
        print(f"  coverage risk  clean         : {statistics.mean(cov_clean):.4f}")
        print(f"  coverage risk  adversarial   : {statistics.mean(cov_adv):.4f}")

    # ── Overlays of the worst raw-attacked cases ──────────────────────────────
    rule("OVERLAYS (worst raw-attacked cases)")
    worst = sorted(per_image, key=lambda r: r[1], reverse=True)[:N_OVERLAYS]
    saved = _save_overlays(worst)
    for p in saved:
        print(f"  saved {p}")
    rule()


def _draw(ax, boxes, color, lw=2.2, ls="-"):
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
            linewidth=lw, linestyle=ls, zorder=3))


def _save_overlays(worst) -> list[Path]:
    saved = []
    for rank, (idx, _drop, gt_r, img_r, adv, sc_box, sa_box) in enumerate(worst):
        clean_np = img_r.permute(1, 2, 0).numpy()
        adv_np = adv.permute(1, 2, 0).numpy()
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))
        axL.imshow(clean_np); axR.imshow(adv_np)
        for ax in (axL, axR):
            ax.axis("off")
        axL.set_title("clean", fontsize=10)
        axR.set_title("adversarial", fontsize=10)
        fig.suptitle(f"worst raw-attacked #{rank}", fontsize=10)
        fig.tight_layout()
        out = OUT_DIR / f"attack_{rank:02d}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        saved.append(out)
    return saved


if __name__ == "__main__":
    main()
