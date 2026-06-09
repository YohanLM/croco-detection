"""Prediction-set construction (spec Section 2).

Two nested sets, both keyed off a single prefiltered `[P, 5]` detection tensor
(pixel `xyxy` + confidence in column 4):

  - `confidence_set` = `Gamma_cnf(x, lambda_cnf)` — keep boxes whose score is
    at least `1 - lambda_cnf`. Larger `lambda_cnf` => lower cutoff => more
    boxes. Implemented by reusing `confidence_filter_expansion`, whose cutoff
    `max(prefilter, 1 - lambda_cnf)` is exactly `Gamma_cnf` floored at the
    a-priori prefilter, so the same operator drives calibration and inference.

  - `localization_set` = `Gamma_loc(x, lambda_cnf, lambda_loc)` — every box of
    `Gamma_cnf` grown by `expand(., lambda_loc)`. Same cardinality as
    `Gamma_cnf`.
"""

from __future__ import annotations

import torch

from conformal.expansion.confidence_filter import confidence_filter_expansion
from conformal.seqcrc.config import MarginMode
from conformal.seqcrc.geometry import expand_boxes


def confidence_set(
    predictions: torch.Tensor, lambda_cnf: float, prefilter: float
) -> torch.Tensor:
    """`Gamma_cnf(x, lambda_cnf)` — boxes with score >= `1 - lambda_cnf`.

    `predictions` is the prefiltered `[P, 5]` set; `prefilter` is the a-priori
    confidence floor, threaded through so the effective cutoff is
    `max(prefilter, 1 - lambda_cnf)`.
    """
    return confidence_filter_expansion(predictions, lambda_cnf, prefilter)


def localization_set(
    predictions: torch.Tensor,
    lambda_cnf: float,
    lambda_loc: float,
    prefilter: float,
    mode: MarginMode,
) -> torch.Tensor:
    """`Gamma_loc(x, lambda_cnf, lambda_loc)` — `Gamma_cnf` expanded by margin."""
    kept = confidence_set(predictions, lambda_cnf, prefilter)
    return expand_boxes(kept, lambda_loc, mode)
