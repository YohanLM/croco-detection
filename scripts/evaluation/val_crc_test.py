"""Evaluate best.pt on the CRC held-out test split (n=800).

Run from the croco_detection project root on the GPU machine:

    cd /home/data/home/lemorhedec-y/croco_detection
    python scripts/evaluation/val_crc_test.py

Prints the four numbers needed for the report table and saves diagnostic
plots to outputs/val_crc_test/.
"""

import tempfile
from pathlib import Path

import yaml

ROOT     = Path(__file__).resolve().parent.parent.parent
WEIGHTS  = ROOT / "models" / "best.pt"
TEST_TXT = ROOT / "data" / "splits" / "test.txt"
CONF     = 0.30
IMGSZ    = 640
DEVICE   = "cuda"
OUT_DIR  = ROOT / "outputs" / "val_crc_test"

# Build a minimal YAML so YOLO val can locate images + labels.
# path="/" makes the absolute paths in the .txt file resolve correctly.
data_cfg = {
    "path": "/",
    "train": str(TEST_TXT),   # required by Ultralytics even for val-only runs
    "val": str(TEST_TXT),
    "nc": 1,
    "names": {0: "croco"},
}
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
    yaml.dump(data_cfg, fh)
    tmp_yaml = Path(fh.name)

from ultralytics import YOLO  # noqa: E402

model   = YOLO(str(WEIGHTS))
metrics = model.val(
    data    = str(tmp_yaml),
    split   = "val",
    conf    = CONF,
    iou     = 0.50,
    imgsz   = IMGSZ,
    device  = DEVICE,
    project = str(OUT_DIR.parent),
    name    = OUT_DIR.name,
    plots   = True,
)
tmp_yaml.unlink(missing_ok=True)

print("\n" + "=" * 52)
print("YOLO base detector  —  CRC test split  (n=800)")
print(f"weights : {WEIGHTS}")
print(f"conf    : {CONF}   iou@.50")
print("=" * 52)
print(f"  Precision  : {metrics.box.mp:.4f}")
print(f"  Recall     : {metrics.box.mr:.4f}")
print(f"  mAP@.50    : {metrics.box.map50:.4f}")
print(f"  mAP@.50:.95: {metrics.box.map:.4f}")
print("=" * 52)
print(f"Plots saved to {OUT_DIR}")
