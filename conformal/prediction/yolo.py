"""Ultralytics-YOLO `PredictionFunction` implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
from ultralytics import YOLO

from conformal.calibrator import PredictionFunction


ModelLike = Union[str, Path, YOLO]


class YoloPredictor:
    """Implements `PredictionFunction`.

    Holds an Ultralytics YOLO model; each call runs it on one image at the
    given confidence threshold and returns `[P, 5]` — pixel-xyxy boxes
    plus a per-box confidence score in column 4. The same instance can be
    reused at different confidence levels.

    Outputs are in PIXEL coordinates (NOT normalized) — Ultralytics
    `boxes.xyxy` is already pixel-space. The confidence column is carried
    through so downstream confidence-aware expansions can use it.
    """

    def __init__(self, model: ModelLike) -> None:
        self.model = YOLO(str(model)) if isinstance(model, (str, Path)) else model

    def __call__(
        self, image_path: str, confidence_threshold: float
    ) -> torch.Tensor:
        result = self.model(image_path, conf=confidence_threshold, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        xyxy = boxes.xyxy.detach().cpu().float()              # [P, 4] pixel coords
        conf = boxes.conf.detach().cpu().float().unsqueeze(1)  # [P, 1]
        return torch.cat([xyxy, conf], dim=1)                  # [P, 5]


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_predictor: PredictionFunction = YoloPredictor.__new__(YoloPredictor)
