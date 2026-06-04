"""Check that CRC calib/test splits are disjoint from training data.

Run from the croco_detection project root on the GPU machine:

    cd /home/data/home/lemorhedec-y/croco_detection
    python scripts/evaluation/check_disjointness.py

Reads the training and val image lists from dataset_800.yaml and compares
them against data/splits/{calibration,test}.txt.  Reports any overlap and
exits with code 1 if any is found.
"""

import sys
from pathlib import Path

import yaml

CROCO_ROOT   = Path(__file__).resolve().parent.parent.parent
DATASET_YAML = Path(
    "/home/data/home/lemorhedec-y/object-detection/"
    "experiments/sq_c30_m15_col/splits/dataset_800.yaml"
)
CALIB_TXT = CROCO_ROOT / "data" / "splits" / "calibration.txt"
TEST_TXT  = CROCO_ROOT / "data" / "splits" / "test.txt"


def stems_from_txt(path: Path) -> set[str]:
    return {Path(ln.strip()).stem for ln in path.read_text().splitlines() if ln.strip()}


def stems_from_yaml_key(yaml_path: Path, key: str) -> set[str]:
    """Return image stems from a YOLO dataset YAML split key (train/val/test)."""
    cfg  = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg.get("path", "/"))
    val  = cfg.get(key)
    if not val:
        return set()
    split_path = Path(val)
    if not split_path.is_absolute():
        split_path = root / split_path
    if split_path.suffix == ".txt":
        return {Path(ln.strip()).stem for ln in split_path.read_text().splitlines() if ln.strip()}
    if split_path.is_dir():
        return {p.stem for p in split_path.rglob("*") if p.is_file()}
    return set()


if not DATASET_YAML.exists():
    sys.exit(f"ERROR: dataset YAML not found at {DATASET_YAML}\n"
             "Update the DATASET_YAML path in this script.")

calib_stems = stems_from_txt(CALIB_TXT)
test_stems  = stems_from_txt(TEST_TXT)
train_stems = stems_from_yaml_key(DATASET_YAML, "train")
val_stems   = stems_from_yaml_key(DATASET_YAML, "val")

print(f"CRC calibration : {len(calib_stems)} images")
print(f"CRC test        : {len(test_stems)} images")
print(f"Training set    : {len(train_stems)} images")
print(f"Validation set  : {len(val_stems)} images  (used to select best.pt)")
print()

checks = [
    ("calib  ∩  train", calib_stems & train_stems),
    ("calib  ∩  val  ", calib_stems & val_stems),
    ("test   ∩  train", test_stems  & train_stems),
    ("test   ∩  val  ", test_stems  & val_stems),
    ("calib  ∩  test ", calib_stems & test_stems),
]

fail = False
for label, overlap in checks:
    if overlap:
        examples = sorted(overlap)[:5]
        print(f"  OVERLAP  {label} : {len(overlap)} files  e.g. {examples}")
        fail = True
    else:
        print(f"  OK       {label} : no overlap")

print()
if fail:
    print("FAIL — overlapping splits invalidate the CRC guarantee.")
    sys.exit(1)
else:
    print("PASS — all splits are disjoint.  CRC results are valid.")
