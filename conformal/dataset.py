"""Ground-truth data loading for Conformal Risk Control.

This module is the GT side of the pipeline only.  It reads image paths and
their corresponding YOLO-format label files; it knows nothing about model
predictions or confidence scores.  Predictions (with confidence in column 4)
come from `conformal/prediction/yolo.py`.  The two meet inside the loss and
expansion functions, which expect `[P, 5]` predictions and `[G, 4]` GT boxes.

Two datasets, mirroring the two stages of the pipeline:

  - `CalibrationDataset`: yields `(image_path, gt_xyxy_pixels)` per sample.
    Used by the Calibrator and any test-time evaluation that needs GT.
    Reads each image's dimensions via PIL header (no full decode) to convert
    YOLO-normalized labels to pixel-xyxy at load time, so the rest of the
    pipeline stays format-agnostic.

  - `PredictionDataset`: yields just `image_path` per sample. Used at
    deployment when there is no GT — pairs with the calibrated `λ̂` and
    the expansion function to produce prediction sets only.

Each split file (`data/splits/{calib,test}.txt`) holds one absolute image
path per line. Labels live in a sibling `labels/` directory with the same
stem and a `.txt` extension, in YOLO format:

    class_id  cx  cy  w  h        # normalized to [0, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PathLike = Union[str, Path]


# ── Path & label helpers ─────────────────────────────────────────────────────

def _label_path_for(image_path: Path) -> Path:
    """Map `.../images/<stem>.<ext>` → `.../labels/<stem>.txt`.

    Replaces the first `images` path component with `labels`, so this works
    regardless of OS separators or where `images/` sits in the tree.
    """
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _parse_yolo_label(label_file: Path) -> torch.Tensor:
    """Parse a YOLO-format label file into `[N, 5]` (cls, cx, cy, w, h).

    Returns an empty `(0, 5)` tensor if the file is missing or empty.
    """
    if not label_file.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    rows = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append([float(x) for x in parts])
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _image_dims(image_path: Path) -> tuple[int, int]:
    """Return `(img_w, img_h)` by reading the image header — no decode."""
    with Image.open(image_path) as im:
        return im.size  # PIL returns (width, height)


def yolo_norm_to_xyxy(
    gt_norm: torch.Tensor, img_w: int, img_h: int
) -> torch.Tensor:
    """Convert YOLO-normalized `[N, 5]` to pixel-xyxy `[N, 4]`.

    Drops the class column — single-class detector, no use for it downstream.
    """
    if gt_norm.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx = gt_norm[:, 1] * img_w
    cy = gt_norm[:, 2] * img_h
    w = gt_norm[:, 3] * img_w
    h = gt_norm[:, 4] * img_h
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


# ── Datasets ─────────────────────────────────────────────────────────────────

def _read_split(split_file: PathLike) -> list[Path]:
    """One absolute image path per non-blank line."""
    lines = Path(split_file).read_text().splitlines()
    return [Path(line.strip()) for line in lines if line.strip()]


class CalibrationDataset(Dataset):
    """Yields `(image_path: str, gt_xyxy_pixels: Tensor[N, 4]), img_size: tuple[int, int])` per sample."""

    def __init__(self, split_file: PathLike) -> None:
        self.image_paths = _read_split(split_file)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        gt_norm = _parse_yolo_label(_label_path_for(img_path))
        img_w, img_h = _image_dims(img_path)
        gt_xyxy = yolo_norm_to_xyxy(gt_norm, img_w, img_h)
        return str(img_path), gt_xyxy, (img_w, img_h)


class PredictionDataset(Dataset):
    """Yields just `image_path: str, img_size: tuple[int, int])` per sample — no labels needed."""

    def __init__(self, split_file: PathLike) -> None:
        self.image_paths = _read_split(split_file)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, tuple[int, int]]:
        img_path = self.image_paths[idx]
        img_w, img_h = _image_dims(img_path)
        return str(img_path), (img_w, img_h)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _collate_with_gt(batch):
    """Keep variable-length GT tensors as a list — don't stack."""
    paths, gts,sizes = zip(*batch)
    return list(paths), list(gts), list(sizes)


def make_calibration_loader(
    split_file: PathLike,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Loader over `(image_path, gt_xyxy_pixels, img_size)` — for Calibrator / evaluation."""
    return DataLoader(
        CalibrationDataset(split_file),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_with_gt,
    )


def make_prediction_loader(
    split_file: PathLike,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Loader over image paths only — for deployment-time inference."""
    return DataLoader(
        PredictionDataset(split_file),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
