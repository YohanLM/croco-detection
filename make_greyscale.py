"""Convert a synthetic-rails dataset (or a flat folder of PNGs) to greyscale.

Strips the chroma signal so the model has to rely on shape and brightness
patterns rather than the red/green palette. The output is still saved as
a **3-channel PNG with R = G = B** — YOLO expects 3 channels (COCO-pretrained
weights would fail on a single-channel input), and matching channels is the
standard idiom for "monochrome data through an RGB network."

Two input layouts are supported automatically:
  - **YOLO dataset** — `<src>/images/*.png` and `<src>/labels/*.txt`.
    Mirrored into `<dst>/images/` (greyscale) and `<dst>/labels/` (verbatim).
  - **Flat folder** — any directory containing `*.png` directly (e.g. the
    `preview/` and `preview_square/` directories). All PNGs are converted
    into the destination directory.

The script is idempotent: if `<dst>` already contains a converted copy
matching the source layout, it just overwrites — no stateful caching, since
the conversion is so cheap.

Usage:
    python make_greyscale.py <src_dir> <dst_dir>

Examples:
    python make_greyscale.py preview preview_grey
    python make_greyscale.py preview_square preview_square_grey
    python make_greyscale.py data/dataset/experiment_15c_5s \\
                             data/dataset/experiment_15c_5s_grey
"""

import shutil
import sys
from pathlib import Path

from PIL import Image


def _convert_one(src_png: Path, dst_png: Path) -> None:
    """Read a colour PNG, save it as 3-channel greyscale.

    PIL's `.convert("L")` uses the ITU-R 601-2 luma transform
    (`L = R * 299/1000 + G * 587/1000 + B * 114/1000`). We then expand back
    to 3 channels so YOLO's 3-channel input layer is satisfied.
    """
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as im:
        grey = im.convert("L").convert("RGB")
        grey.save(dst_png)


def _convert_yolo_layout(src: Path, dst: Path) -> tuple[int, int]:
    """Mirror an `images/` + `labels/` directory pair from src into dst.

    Images are converted, labels are copied verbatim. Returns
    (n_images_converted, n_labels_copied).
    """
    images_src = src / "images"
    labels_src = src / "labels"
    images_dst = dst / "images"
    labels_dst = dst / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for png in sorted(images_src.glob("*.png")):
        _convert_one(png, images_dst / png.name)
        n_images += 1

    n_labels = 0
    for txt in sorted(labels_src.glob("*.txt")):
        shutil.copy2(txt, labels_dst / txt.name)
        n_labels += 1

    return n_images, n_labels


def _convert_flat(src: Path, dst: Path) -> int:
    """Convert all PNGs found directly inside `src` into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(src.glob("*.png")):
        _convert_one(png, dst / png.name)
        n += 1
    return n


def convert(src: Path, dst: Path) -> None:
    """Detect the input layout and convert accordingly."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise SystemExit(f"Source directory not found: {src}")

    images_subdir = src / "images"
    if images_subdir.is_dir():
        n_img, n_lab = _convert_yolo_layout(src, dst)
        print(f"YOLO layout: converted {n_img} images, copied {n_lab} labels")
        print(f"  -> {dst}")
        return

    # Flat folder of PNGs
    n = _convert_flat(src, dst)
    if n == 0:
        raise SystemExit(
            f"No PNGs found in {src} (and no images/ subdirectory either)."
        )
    print(f"Flat layout: converted {n} images")
    print(f"  -> {dst}")


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        print(__doc__.strip())
        raise SystemExit(2)
    convert(argv[1], argv[2])


if __name__ == "__main__":
    main(sys.argv)
