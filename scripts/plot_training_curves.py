"""Plot per-epoch training and validation losses for each subset size.

Reads results.csv produced by ultralytics inside each detect/size_* directory
and draws one figure per experiment with train and val loss curves.

Usage:
    python scripts/plot_training_curves.py                          # default experiment
    python scripts/plot_training_curves.py experiments/sq_c30_m15_col
    python scripts/plot_training_curves.py experiments/sq_c30_m15_col experiments/sq_c30_m15_col_nogeom
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN_LOSSES = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
VAL_LOSSES   = ["val/box_loss",   "val/cls_loss",   "val/dfl_loss"]
LOSS_LABELS  = ["Box loss", "Class loss", "DFL loss"]


def read_results_csv(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open()))
    out = {}
    for key in rows[0]:
        try:
            out[key.strip()] = [float(r[key]) for r in rows]
        except ValueError:
            pass
    return out


def plot_experiment(run_dir: Path) -> None:
    detect_dir = run_dir / "detect"
    csv_files = sorted(
        detect_dir.glob("size_*/results.csv"),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    if not csv_files:
        print(f"No results.csv files found under {detect_dir}")
        return

    sizes = [int(p.parent.name.split("_")[1]) for p in csv_files]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_losses = len(TRAIN_LOSSES)
    fig, axes = plt.subplots(n_losses, 2, figsize=(12, 4 * n_losses), sharey="row")

    for col, (losses, split) in enumerate([(TRAIN_LOSSES, "Train"), (VAL_LOSSES, "Val")]):
        for row, (loss_key, label) in enumerate(zip(losses, LOSS_LABELS)):
            ax = axes[row, col]
            for i, (csv_path, size) in enumerate(zip(csv_files, sizes)):
                data = read_results_csv(csv_path)
                if loss_key not in data:
                    continue
                epochs = list(range(1, len(data[loss_key]) + 1))
                ax.plot(epochs, data[loss_key],
                        color=colors[i % len(colors)],
                        label=f"n={size}", linewidth=1.8)
            ax.set_title(f"{split} — {label}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

    fig.suptitle(f"Training curves — {run_dir.name}", fontsize=13)
    fig.tight_layout()

    out_path = run_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    run_dirs = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [Path("experiments/sq_c30_m15_col")]
    for run_dir in run_dirs:
        print(f"\n=== {run_dir.name} ===")
        plot_experiment(run_dir)
