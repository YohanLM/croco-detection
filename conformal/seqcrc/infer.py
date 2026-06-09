"""SeqCRC inference (spec Section 8) — the calibrated two-step runtime pipeline.

Given the calibrated `lambda_cnf_plus` (Step 1) and `lambda_loc_plus` (Step 2),
each query frame is processed as:

    1. predict + prefilter        (same `conf >= prefilter` floor as calibration)
    2. confidence filtering       Gamma_cnf at lambda_cnf_plus
    3. localization margin        expand by lambda_loc_plus

returning the conformally expanded boxes (single class, no class set). Inference
is cheap — pure set construction — negligible against the detector forward pass.
Under the SeqCRC guarantee the expected localization loss of this field is
bounded by `alpha_loc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from conformal.calibrator import PredictionFunction
from conformal.seqcrc.config import MarginMode, SeqCRCConfig
from conformal.seqcrc.sets import localization_set


@dataclass
class SeqCRCInferencer:
    """Composed runtime handle for a calibrated SeqCRC model.

    Construct directly from the calibrated parameters, or via
    `from_config(...)` to inherit `prefilter` / `margin_mode` from the same
    `SeqCRCConfig` used for calibration.
    """

    predictor: PredictionFunction
    lambda_cnf_plus: float
    lambda_loc_plus: float
    prefilter: float = 1e-3
    margin_mode: MarginMode = "additive"

    @classmethod
    def from_config(
        cls,
        predictor: PredictionFunction,
        lambda_cnf_plus: float,
        lambda_loc_plus: float,
        cfg: SeqCRCConfig,
    ) -> "SeqCRCInferencer":
        """Build an inferencer that mirrors `cfg`'s prefilter and margin mode."""
        return cls(
            predictor=predictor,
            lambda_cnf_plus=lambda_cnf_plus,
            lambda_loc_plus=lambda_loc_plus,
            prefilter=cfg.prefilter,
            margin_mode=cfg.margin_mode,
        )

    def __call__(self, image_path: str) -> torch.Tensor:
        """Return the certified `Gamma_loc` field for one query frame `[P, 5]`."""
        raw = self.predictor(image_path, self.prefilter)        # predict + prefilter
        return localization_set(                                # filter then expand
            raw,
            self.lambda_cnf_plus,
            self.lambda_loc_plus,
            self.prefilter,
            self.margin_mode,
        )
