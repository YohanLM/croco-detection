"""Data loading for the image-level "clip / no-clip" binary classifier.

This is the classification counterpart to ``conformal/dataset.py``. Where the CRC
pipeline reads YOLO labels as *boxes* (for localisation), here we collapse each
image to a single binary target:

    label = +1  if the image contains a clip   (YOLO label file has ≥1 box)
    label = -1  if it does not                 (label file empty or missing)

The ``{+1, -1}`` convention is what the HKR / Kantorovich-Rubinstein losses expect
(see ``deel.torchlip`` and the MNIST 0-vs-8 notebook this module ports). At most one
clip ever appears per image, so "is there a clip" is a clean binary question and the
box geometry is irrelevant — only its *presence*.

Images are returned as ``[3, H, W]`` float tensors in ``[0, 1]``, RGB order — the same
convention as ``conformal/smoothing/predictor.py`` so a certified L2 radius produced by
the Lipschitz classifier lives in the same pixel units as the smoothing ``sigma``.

The label-reading helpers are reused from ``conformal.dataset`` rather than
re-implemented, so the two pipelines can never disagree about what counts as a clip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Reuse the GT-side helpers — single source of truth for label parsing / split files.
from conformal.dataset import _label_path_for, _parse_yolo_label, _read_split

PathLike = Union[str, Path]

# Target encoding (HKR / hinge convention).
CLIP_LABEL = 1.0
NO_CLIP_LABEL = -1.0


# ── Image loading ────────────────────────────────────────────────────────────

def _load_image_chw01(image_path: PathLike) -> torch.Tensor:
    """Load an image as a ``[3, H, W]`` float tensor in ``[0, 1]``, RGB order.

    Mirrors ``conformal/smoothing/predictor.py:load_image_chw01`` so the classifier
    sees pixels in exactly the same representation as the smoothing certificate.
    Kept local to avoid importing the (heavier) smoothing/calibrator chain.
    """
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()          # [3, H, W]


def label_for(image_path: PathLike) -> float:
    """``+1`` if the image's YOLO label file holds ≥1 box, else ``-1``."""
    boxes = _parse_yolo_label(_label_path_for(Path(image_path)))
    return CLIP_LABEL if boxes.shape[0] > 0 else NO_CLIP_LABEL


# ── Dataset ──────────────────────────────────────────────────────────────────

class ClipClassificationDataset(Dataset):
    """Yields ``(image[3, H, W] float[0, 1], label ∈ {+1, -1})`` per sample.

    Args:
        image_paths: image file paths. Each is expected to have a sibling
            ``labels/<stem>.txt`` (the ``images`` → ``labels`` swap done by
            ``conformal.dataset._label_path_for``); a missing/empty file means
            "no clip".
    """

    def __init__(self, image_paths: Sequence[PathLike]) -> None:
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        image = _load_image_chw01(path)
        label = torch.tensor(label_for(path), dtype=torch.float32)
        return image, label

    def label_array(self) -> torch.Tensor:
        """All labels as a ``[N]`` tensor — cheap (reads label files, not images)."""
        return torch.tensor([label_for(p) for p in self.image_paths], dtype=torch.float32)


# ── Loader ───────────────────────────────────────────────────────────────────

def make_loader(
    image_paths: Sequence[PathLike],
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """A ``DataLoader`` over ``ClipClassificationDataset``.

    All images in a split share one H×W (synthetic generator emits a fixed size and
    the YOLO splits are letterboxed upstream), so the default collate stacks them
    into a ``[B, 3, H, W]`` batch with no padding needed.
    """
    return DataLoader(
        ClipClassificationDataset(image_paths),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ── Balanced synthetic training data ──────────────────────────────────────────

def build_balanced_synthetic(
    output_dir: PathLike,
    n_samples: int,
    seed: int,
    p_clip: float = 0.5,
    name: str | None = None,
) -> list[Path]:
    """Generate a balanced synthetic split on disk and return its image paths.

    Delegates image synthesis to the project's square generator
    (``data_generation.dataset_synthetic_square.load_synthetic_rails_square``),
    passing a custom config dict so the clip prevalence is ``p_clip`` (default 0.5
    for a balanced classifier training set) rather than the low/realistic rates the
    detector configs use.

    The generator caches per-config (it skips regeneration when the label dir already
    holds ≥ ``n_samples`` files), so repeat calls with the same ``name`` are cheap.

    Args:
        output_dir: root under which a ``<name>/`` subdir is created.
        n_samples:  number of images to generate.
        seed:       deterministic seed (use different seeds for train vs val).
        p_clip:     fraction of images that contain a clip.
        name:       config subdir name; defaults to ``clf_seed<seed>``.

    Returns:
        Sorted list of generated image paths.
    """
    # Imported here (not at module top) so merely importing `lipschitz.data` does not
    # pull in the synthetic generator's numpy/PIL drawing stack.
    from data_generation.dataset_synthetic_square import load_synthetic_rails_square

    cfg_name = name or f"clf_seed{seed}"
    config = {
        "name": cfg_name,
        "p_clip": p_clip,
        "p_switch": 0.0,
        "p_motif": 0.05,
        "clip_tracks": ("upper", "lower"),
    }
    info = load_synthetic_rails_square(
        output_dir, config=config, n_samples=n_samples, seed=seed
    )
    return sorted(Path(info["images_dir"]).glob("*.png"))


def paths_from_split(split_file: PathLike) -> list[Path]:
    """Image paths from a split file (one absolute path per line).

    Thin wrapper over ``conformal.dataset._read_split`` so eval can point straight at
    ``data/splits/test.txt``.
    """
    return _read_split(split_file)


def class_balance(image_paths: Sequence[PathLike]) -> tuple[int, int]:
    """``(n_positive, n_negative)`` over a list of image paths."""
    labels = [label_for(p) for p in image_paths]
    n_pos = sum(1 for v in labels if v == CLIP_LABEL)
    return n_pos, len(labels) - n_pos
