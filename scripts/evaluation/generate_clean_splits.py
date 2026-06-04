"""Regenerate calib/test splits free of model-selection bias.

Problem: the original test split equals the YOLO training validation split —
the 800 images on which best.pt was selected.  The CRC guarantee is still
mathematically valid (lambda-hat was calibrated on clean data), but the
reported test metrics are optimistic because the model was chosen for good
performance on those exact images.

Fix: combine both 800-image splits (all 1600 are disjoint from the training
set) and re-shuffle into a fresh 800/800 partition.  Each new split contains
~50% previously-clean images, which is far less biased than 100% model-selected
in the test set.

Run from the croco_detection project root:
    python scripts/evaluation/generate_clean_splits.py

Writes:
    data/splits/calibration.txt   (overwritten in-place)
    data/splits/test.txt          (overwritten in-place)
    data/splits/calibration_orig.txt  (backup)
    data/splits/test_orig.txt         (backup)
"""

import random
import shutil
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent.parent
CALIB_TXT = ROOT / "data" / "splits" / "calibration.txt"
TEST_TXT  = ROOT / "data" / "splits" / "test.txt"
SEED      = 42

# --- back up originals ---------------------------------------------------
shutil.copy(CALIB_TXT, CALIB_TXT.with_name("calibration_orig.txt"))
shutil.copy(TEST_TXT,  TEST_TXT.with_name("test_orig.txt"))
print("Originals backed up to calibration_orig.txt / test_orig.txt")

# --- load and shuffle ----------------------------------------------------
calib_lines = [ln.strip() for ln in CALIB_TXT.read_text().splitlines() if ln.strip()]
test_lines  = [ln.strip() for ln in TEST_TXT.read_text().splitlines()  if ln.strip()]

assert len(calib_lines) == 800, f"Expected 800 calib images, got {len(calib_lines)}"
assert len(test_lines)  == 800, f"Expected 800 test images,  got {len(test_lines)}"

all_images = calib_lines + test_lines        # 1600 total, all disjoint from train
rng = random.Random(SEED)
rng.shuffle(all_images)

new_calib = all_images[:800]
new_test  = all_images[800:]

# --- sanity checks -------------------------------------------------------
assert set(new_calib) & set(new_test) == set(), "new splits overlap — bug"
assert set(new_calib) | set(new_test) == set(all_images), "images lost — bug"

orig_val_stems = {Path(p).stem for p in test_lines}   # the model-selected images
n_val_in_calib = sum(Path(p).stem in orig_val_stems for p in new_calib)
n_val_in_test  = sum(Path(p).stem in orig_val_stems for p in new_test)

# --- write ---------------------------------------------------------------
CALIB_TXT.write_text("\n".join(new_calib) + "\n")
TEST_TXT.write_text("\n".join(new_test) + "\n")

print(f"\nNew calibration split: {n_val_in_calib}/800 from orig model-selection set")
print(f"New test split       : {n_val_in_test}/800 from orig model-selection set")
print(f"\nWritten to {CALIB_TXT}")
print(f"          {TEST_TXT}")
print("\nNext: re-run all CRC experiments (scripts/calibration/calibrate_*.py)")
