"""Kaggle pothole dataset.

Downloads the `andrewmvd/pothole-detection` dataset from Kaggle, symlinks the
images into a local folder, and converts the Pascal-VOC XML annotations to
YOLO-format `.txt` labels.

Exposes a loader matching the project's dataset interface:
    f(output_dir) -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import kagglehub


def load_kaggle_pothole(output_dir):
    output_dir = Path(output_dir)
    images_dst = output_dir / "images"
    labels_dst = output_dir / "labels"
    classes = ["pothole"]

    # Skip download if labels are already there from a previous run
    if labels_dst.exists() and any(labels_dst.iterdir()):
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    # Download the dataset from Kaggle (cached locally after the first call)
    src = Path(kagglehub.dataset_download("andrewmvd/pothole-detection"))
    images_src = src / "images"
    annotations_src = src / "annotations"

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    cls_to_idx = {c: i for i, c in enumerate(classes)}

    for img in sorted(images_src.iterdir()):
        if img.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        # Symlink instead of copying to avoid duplicating data on disk
        link = images_dst / img.name
        if not link.exists():
            link.symlink_to(img.resolve())

        # Convert Pascal-VOC XML annotation to YOLO .txt format
        xml = annotations_src / (img.stem + ".xml")
        if xml.exists():
            (labels_dst / (img.stem + ".txt")).write_text(_voc_xml_to_yolo(xml, cls_to_idx))

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}


def _voc_xml_to_yolo(xml_path, cls_to_idx):
    """Convert a Pascal-VOC XML file to a YOLO label string.

    YOLO format (one line per object):
        <class_idx> <cx> <cy> <width> <height>
    All values are normalised to [0, 1] relative to image size.
    """
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in cls_to_idx:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_to_idx[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return ("\n".join(lines) + "\n") if lines else ""
