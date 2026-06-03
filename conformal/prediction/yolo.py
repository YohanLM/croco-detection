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
        return self._result_to_tensor(result)

    def predict_batch(
        self, image_paths: list[str], confidence_threshold: float
    ) -> list[torch.Tensor]:
        results = self.model(image_paths, conf=confidence_threshold, verbose=False)
        return [self._result_to_tensor(r) for r in results]

    def predict_arrays(
        self, images: torch.Tensor, confidence_threshold: float
    ) -> list[torch.Tensor]:
        """Run the detector on a batch of **in-memory** images, not paths.

        `images` is a `[B, 3, H, W]` float tensor with values already
        normalized to `[0, 1]` in **RGB** channel order. Ultralytics' tensor
        source path treats such input as pre-normalized RGB, which sidesteps
        the BGR convention it applies to numpy arrays — so the same pixels a
        caller perturbed in-place are exactly what the model sees.

        This is the hook median smoothing needs: noise is injected at the
        pixel-tensor level (see `conformal.smoothing`), so the base detector
        must accept tensors, not just file paths. Output matches `__call__` /
        `predict_batch`: one `[P, 5]` pixel-xyxy + score tensor per image,
        filtered at `confidence_threshold`.
        """
        results = self.model(images, conf=confidence_threshold, verbose=False)
        return [self._result_to_tensor(r) for r in results]

    @staticmethod
    def _result_to_tensor(result) -> torch.Tensor:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        xyxy = boxes.xyxy.detach().cpu().float()              # [P, 4] pixel coords
        conf = boxes.conf.detach().cpu().float().unsqueeze(1)  # [P, 1]
        return torch.cat([xyxy, conf], dim=1)                  # [P, 5]


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_predictor: PredictionFunction = YoloPredictor.__new__(YoloPredictor)
