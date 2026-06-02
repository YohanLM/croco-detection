"""Sequential Conformal Risk Control (SeqCRC) — runtime composition helpers.

The two SeqCRC phases each reuse the single-knob `Calibrator` unchanged:

  - **Phase 1 (confidence)** calibrates the objectness threshold with a
    detection-miss loss (`conformal.loss.detection`) and the
    `confidence_filter_expansion` knob, yielding `lambda_cnf` -> an effective
    confidence threshold `T_eff = max(floor, 1 - lambda_cnf)`.
  - **Phase 2 (localization)** calibrates an additive pixel margin with the
    75 %-coverage-indicator loss, on the *surviving* frames only, with the
    detector run at `T_eff`, yielding `lambda_loc`.

The user-specified risk budget is partitioned by the Bonferroni union bound,
`alpha = alpha_cnf + alpha_loc`, so the global expected failure rate is
bounded by `alpha`.

This module provides the two pieces that the bare `Calibrator` does not:

  - `build_survivor_split`: materialize the Phase-2 calibration subset (the
    `n_loc` frames Phase 1 did not miss) as a split file, so Phase 2 reuses
    `make_calibration_loader` with no Calibrator change and gets the correct
    `n_loc` finite-sample correction for free.
  - `SeqCRCInferencer`: the composed runtime pipeline — filter at `T_eff`,
    then additively expand by `lambda_loc` — producing the certified field
    for a new query frame.

CAVEAT (deliberate, per the methodology's single-set design): Phase 2 is
conditioned on a survivor subset *selected using the same calibration data*,
so that subset is not strictly exchangeable. The guarantee leans on the union
bound over one set rather than a clean data split; the end-to-end empirical
risk on a held-out test set is the real validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

from conformal.calibrator import LossFunction, PredictionFunction
from conformal.dataset import CalibrationDataset
from conformal.expansion.additive import additive_expansion


PathLike = Union[str, Path]


def effective_threshold(confidence_floor: float, lambda_cnf: float) -> float:
    """Phase-1 effective objectness threshold `T_eff = max(floor, 1 - lambda)`.

    Mirrors `confidence_filter_expansion`'s cutoff so that running the
    detector at `T_eff` reproduces exactly the boxes that survive Phase 1.
    """
    return max(confidence_floor, 1.0 - lambda_cnf)


def build_survivor_split(
    predictor: PredictionFunction,
    split_file: PathLike,
    t_eff: float,
    detection_loss_fn: LossFunction,
    out_file: PathLike,
) -> tuple[int, int, Path]:
    """Write the Phase-2 survivor split and return `(n_total, n_loc, path)`.

    A frame survives iff it has at least one valid GT and the Phase-1
    detection-miss loss is `0` at `t_eff` (every target was hit, so a box
    exists for Phase 2 to expand). Frames with no GT, or with any
    unrecovered target, are excluded — the latter are already charged to
    Phase 1 and would otherwise lock the Phase-2 loss at 1.0.

    The detector is run once per frame at `t_eff`; these are precisely the
    boxes that clear the Phase-1 threshold.
    """
    dataset = CalibrationDataset(split_file)
    survivors: list[str] = []
    n_total = len(dataset)
    for i in range(n_total):
        path, gt = dataset[i]
        if gt.numel() == 0:
            continue
        preds = predictor(path, t_eff)
        if detection_loss_fn(preds, gt) == 0.0:
            survivors.append(path)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(survivors) + ("\n" if survivors else ""))
    return n_total, len(survivors), out_path


@dataclass
class SeqCRCInferencer:
    """Composed two-stage runtime pipeline for a calibrated SeqCRC model.

    Given the calibrated `t_eff` (Phase 1) and `lambda_loc` (Phase 2), each
    call filters a query frame's detections at the threshold and then grows
    every surviving box by `lambda_loc` pixels per side — the certified
    prediction field whose expected failure is bounded by `alpha`.
    """

    predictor: PredictionFunction
    t_eff: float
    lambda_loc: float

    def __call__(self, image_path: str) -> torch.Tensor:
        raw = self.predictor(image_path, self.t_eff)        # Phase-1 filter
        return additive_expansion(raw, self.lambda_loc, self.t_eff)  # Phase-2 expand
