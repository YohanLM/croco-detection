"""Check for overfitting by comparing val metrics on the training split vs. the test split.

Runs for training sizes 800 and 2800.
For the 2800 training set we sample 800 images to keep the comparison fair and fast.

Usage:
    python scripts/eval_overfit_check.py
"""

import random
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT   = Path("experiments/sq_c30_m15_col")
SPLITS_DIR   = EXPERIMENT / "splits"
DETECT_DIR   = EXPERIMENT / "detect"
OUT_PATH     = EXPERIMENT / "overfit_check.png"
DEVICE       = "mps"
IMGSZ        = 640
TRAIN_SAMPLE = 800   # how many training images to evaluate (cap to keep it fast)
SEED         = 42
SIZES        = [800, 2800]

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_paths(txt: Path) -> list[str]:
    return [l.strip() for l in txt.read_text().splitlines() if l.strip()]


def make_temp_yaml(image_paths: list[str], classes: dict) -> Path:
    """Write a temporary dataset YAML pointing at a flat list of images."""
    tmp_txt = Path(tempfile.mktemp(suffix=".txt"))
    tmp_txt.write_text("\n".join(image_paths))
    tmp_yaml = Path(tempfile.mktemp(suffix=".yaml"))
    tmp_yaml.write_text(yaml.dump({
        "path": "/",
        "train": str(tmp_txt),
        "val":   str(tmp_txt),   # YOLO needs both keys; we only use val
        "names": classes,
    }))
    return tmp_yaml


def run_val(weights: Path, image_paths: list[str], classes: dict, label: str) -> dict:
    yaml_path = make_temp_yaml(image_paths, classes)
    model = YOLO(str(weights))
    m = model.val(
        data    = str(yaml_path),
        imgsz   = IMGSZ,
        device  = DEVICE,
        split   = "val",
        verbose = False,
    )
    result = {
        "label":        label,
        "mAP50":        float(m.box.map50),
        "mAP50_95":     float(m.box.map),
        "precision":    float(m.box.mp),
        "recall":       float(m.box.mr),
    }
    print(f"  {label:<30}  mAP@.5={result['mAP50']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

classes = {0: "crocodile_clip"}
test_paths = read_paths(SPLITS_DIR / "test.txt")

all_results = {}

for size in SIZES:
    weights  = DETECT_DIR / f"size_{size}" / "weights" / "best.pt"
    train_paths_full = read_paths(SPLITS_DIR / f"train_{size}.txt")

    rng = random.Random(SEED)
    train_sample = rng.sample(train_paths_full, min(TRAIN_SAMPLE, len(train_paths_full)))

    print(f"\n=== size={size} ===")
    train_result = run_val(weights, train_sample, classes, f"train sample ({len(train_sample)} imgs)")
    val_result   = run_val(weights, test_paths,   classes, f"val set     ({len(test_paths)} imgs)")
    all_results[size] = {"train": train_result, "val": val_result}

# ── Plot ──────────────────────────────────────────────────────────────────────

metrics = ["mAP50", "mAP50_95", "precision", "recall"]
labels  = ["mAP@.5", "mAP@.5:.95", "Precision", "Recall"]

fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), sharey=False)

x      = list(range(len(SIZES)))
width  = 0.35
colors = {"train": "#4C9BE8", "val": "#E8834C"}

for ax, metric, label in zip(axes, metrics, labels):
    train_vals = [all_results[s]["train"][metric] for s in SIZES]
    val_vals   = [all_results[s]["val"][metric]   for s in SIZES]

    bars_train = ax.bar([i - width/2 for i in x], train_vals, width, label="train", color=colors["train"])
    bars_val   = ax.bar([i + width/2 for i in x], val_vals,   width, label="val",   color=colors["val"])

    for bar in bars_train + bars_val:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title(label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"size={s}" for s in SIZES])
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Train vs. Val metrics — sq_c30_m15_col (best.pt)\n"
    f"Train evaluated on {TRAIN_SAMPLE}-image random sample",
    fontweight="bold",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PATH}")
plt.show()
