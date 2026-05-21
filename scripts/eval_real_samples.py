"""Run the 800-image model on the 18 real samples and display results visually.

Usage:
    python scripts/eval_real_samples.py          # default conf=0.25
    python scripts/eval_real_samples.py --conf 0.05

Outputs a matplotlib figure saved to experiments/sq_c30_m15_col/real_samples_eval_confXXX.png
and displayed on screen.
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS     = Path("experiments/sq_c30_m15_col/detect/size_800/weights/best.pt")
IMAGES_DIR  = Path("data/real_samples")
CONF_THRESH = args.conf
IMGSZ       = 640
OUT_PATH    = Path(f"experiments/sq_c30_m15_col/real_samples_eval_conf{int(CONF_THRESH*100):03d}.png")

# Ground-truth: which image indices actually contain a crocodile clip
CLIP_INDICES = {0, 16, 22, 55, 56}

# ── Load model ────────────────────────────────────────────────────────────────
model = YOLO(str(WEIGHTS))

# ── Collect images (sorted by the numeric suffix) ────────────────────────────
image_paths = sorted(
    IMAGES_DIR.glob("*.png"),
    key=lambda p: int(p.stem.split("_")[-1]),
)
print(f"Found {len(image_paths)} real samples")

# ── Run inference ─────────────────────────────────────────────────────────────
results = model.predict(
    source=[str(p) for p in image_paths],
    imgsz=IMGSZ,
    conf=CONF_THRESH,
    device="mps",
    verbose=False,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_cols = 6
n_rows = int(np.ceil(len(image_paths) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4.5))
axes = axes.flatten()

for ax, path, result in zip(axes, image_paths, results):
    img_idx = int(path.stem.split("_")[-1])
    has_gt  = img_idx in CLIP_INDICES

    img = np.array(Image.open(path))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.axis("off")

    boxes = result.boxes
    n_det = len(boxes)

    # Draw detected bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 4, 0),
            f"{conf:.2f}",
            color="lime",
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )

    # Build title with outcome label
    if has_gt and n_det > 0:
        outcome, title_color = "TP", "green"
    elif has_gt and n_det == 0:
        outcome, title_color = "FN", "red"
    elif not has_gt and n_det > 0:
        outcome, title_color = "FP", "orange"
    else:
        outcome, title_color = "TN", "gray"

    gt_marker = " [CLIP]" if has_gt else ""
    ax.set_title(
        f"img {img_idx}{gt_marker}  {outcome}  ({n_det} det.)",
        fontsize=9,
        color=title_color,
        fontweight="bold",
        pad=4,
    )

    # Colour the frame around the subplot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3)
        spine.set_edgecolor(title_color)

# Hide unused axes
for ax in axes[len(image_paths):]:
    ax.set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="green",  label="TP — clip present, detected"),
    mpatches.Patch(color="red",    label="FN — clip present, missed"),
    mpatches.Patch(color="orange", label="FP — no clip, false alarm"),
    mpatches.Patch(color="gray",   label="TN — no clip, correctly ignored"),
    mpatches.Patch(color="lime",   label=f"bounding box (conf threshold = {CONF_THRESH:.2f})"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=True,
    bbox_to_anchor=(0.5, 0.0),
)

fig.suptitle(
    "Model sq_c30_m15_col — 800 images — inference on real samples",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout(rect=[0, 0.07, 1, 1])

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")

# ── Print text summary ────────────────────────────────────────────────────────
print("\n{:<25} {:>5} {:>6}  {}".format("Image", "GT", "#det", "Outcome"))
print("-" * 55)
for path, result in zip(image_paths, results):
    img_idx = int(path.stem.split("_")[-1])
    has_gt  = img_idx in CLIP_INDICES
    n_det   = len(result.boxes)
    confs   = [f"{float(b.conf[0]):.2f}" for b in result.boxes]
    outcome = ("TP" if has_gt and n_det else
               "FN" if has_gt else
               "FP" if n_det else "TN")
    gt_str  = "clip" if has_gt else "none"
    print(f"{path.name:<25} {gt_str:>5} {n_det:>4}   {outcome}  {confs}")

plt.show()
