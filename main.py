"""Train YOLO on increasingly large subsets of the dataset and compare metrics.

Pluggable loader: swap `load_dataset` for any function with the
`dataset_loader` interface to run the same experiment on a different dataset.
"""

import json
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from dataset_pothole import load_kaggle_pothole
from dataset_synthetic import load_synthetic_rails

# Load KAGGLE_USERNAME and KAGGLE_KEY from .env so kagglehub can authenticate
load_dotenv()

# ── Pluggable dataset loader ──────────────────────────────────────────────────
# To test a different dataset, write a new loader in its own dataset_*.py file
# and replace this line. The rest of the script stays the same.
load_dataset = load_synthetic_rails

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/dataset")   # where images/ and labels/ will be created
WORK_DIR = Path("data/splits")    # train_N.txt, test.txt and YAML files go here
RESULTS_FILE = Path("results.json")

# ── Experiment config ─────────────────────────────────────────────────────────
SUBSET_SIZES = [100, 250, 500]  # number of training images to try
TEST_RATIO = 0.2    # 20 % of the dataset is held out as a fixed test set
EPOCHS = 25
IMGSZ = 640         # longest-side target; rect=True keeps the 570x100 aspect ratio
DEVICE = "mps"      # Apple Silicon GPU; use "cpu" or "0" (CUDA) on other machines
SEED = 42           # fixes shuffle so every run produces the same splits
WEIGHTS = "yolo11n.pt"  # nano variant — smallest and fastest YOLO11


def main():
    # ── 1. Prepare the dataset ────────────────────────────────────────────────
    # Downloads if needed, converts XML → YOLO .txt, symlinks images.
    info = load_dataset(DATA_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # ── 2. Build a fixed train/test split ─────────────────────────────────────
    # Only keep images that have a matching label file.
    images = sorted(
        p for p in info["images_dir"].iterdir()
        if (info["labels_dir"] / (p.stem + ".txt")).exists()
    )
    random.Random(SEED).shuffle(images)  # same order every run

    n_test = int(len(images) * TEST_RATIO)
    test_set = images[:n_test]           # fixed test set — never used for training
    train_pool = images[n_test:]         # everything else is the training pool
    print(f"Total: {len(images)} | Test: {len(test_set)} | Train pool: {len(train_pool)}")

    # Write the test set image paths to a file (ultralytics reads these)
    test_file = WORK_DIR / "test.txt"
    test_file.write_text("\n".join(str(p.resolve()) for p in test_set))

    # ── 3. Train once per subset size ─────────────────────────────────────────
    results = []
    for size in SUBSET_SIZES:
        if size > len(train_pool):
            print(f"Skipping size={size}: only {len(train_pool)} train images available")
            continue

        # Subsets are nested: train_20 contains the same 10 images as train_10
        # plus 10 new ones. This makes the comparison fair.
        train_file = WORK_DIR / f"train_{size}.txt"
        train_file.write_text("\n".join(str(p.resolve()) for p in train_pool[:size]))

        # YOLO expects a YAML file describing the dataset (paths + class names)
        yaml_path = WORK_DIR / f"dataset_{size}.yaml"
        yaml_path.write_text(yaml.dump({
            "path": str(WORK_DIR.resolve()),
            "train": train_file.name,
            "val": test_file.name,   # evaluate on the held-out test set
            "names": {i: c for i, c in enumerate(info["classes"])},
        }))

        print(f"\n=== Training with {size} images ===")
        model = YOLO(WEIGHTS)   # fresh pretrained weights for each run
        model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            device=DEVICE,
            name=f"size_{size}",  # run saved to runs/detect/size_N/
            exist_ok=True,        # overwrite previous run with the same name
            seed=SEED,
            rect=True,            # rectangular training: pads to 640x128 instead of 640x640
        )

        # Evaluate the trained model on the fixed test set
        metrics = model.val(data=str(yaml_path), device=DEVICE)
        results.append({
            "size": size,
            "mAP@.5":     float(metrics.box.map50),  # IoU threshold = 0.5
            "mAP@.5:.95": float(metrics.box.map),    # averaged over IoU 0.5–0.95
            "inference_ms": float(metrics.speed.get("inference", 0.0)),
        })

        # Save after each run so results aren't lost if the script is interrupted
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    # ── 4. Print summary table ────────────────────────────────────────────────
    print(f"\n{'Size':>6} {'mAP@.5':>10} {'mAP@.5:.95':>12} {'Inf (ms)':>10}")
    for r in results:
        print(f"{r['size']:>6} {r['mAP@.5']:>10.4f} {r['mAP@.5:.95']:>12.4f} {r['inference_ms']:>10.2f}")


if __name__ == "__main__":
    main()
