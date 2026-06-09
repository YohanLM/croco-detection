"""Sequential Conformal Risk Control (SeqCRC) for single-class object detection.

Implements Andeol et al. "Conformal Object Detection by Sequential Risk
Control" (arXiv:2505.24038), specialized to one object class, a false-negative
-rate confidence loss, and a pixel-superposition-threshold localization loss
(see `docs/seqcrc_single_class_spec.md`).

Module map (spec Section 11):
    config.py    : SeqCRCConfig, MarginMode (a-priori params + startup checks)
    geometry.py  : area, intersection_area, expand_boxes
    sets.py      : confidence_set (Gamma_cnf), localization_set (Gamma_loc)
    matching.py  : d_haus, match (asymmetric signed Hausdorff)
    losses.py    : l_cnf_image (FNR), l_loc_image (pixel-superposition)
    calibrate.py : calibrate + the two sub-searches + empirical risks
    infer.py     : SeqCRCInferencer
"""

from conformal.seqcrc.calibrate import (
    CalibrationResult,
    calibrate,
    calibrate_confidence,
    calibrate_localization,
    collect_predictions,
    confidence_risk,
    localization_risk,
)
from conformal.seqcrc.config import MarginMode, SeqCRCConfig
from conformal.seqcrc.geometry import area, expand_boxes, intersection_area
from conformal.seqcrc.infer import SeqCRCInferencer
from conformal.seqcrc.losses import l_cnf_image, l_loc_image
from conformal.seqcrc.matching import d_haus, match
from conformal.seqcrc.sets import confidence_set, localization_set

__all__ = [
    # config
    "SeqCRCConfig",
    "MarginMode",
    # geometry
    "area",
    "intersection_area",
    "expand_boxes",
    # sets
    "confidence_set",
    "localization_set",
    # matching
    "d_haus",
    "match",
    # losses
    "l_cnf_image",
    "l_loc_image",
    # calibration
    "calibrate",
    "calibrate_confidence",
    "calibrate_localization",
    "collect_predictions",
    "confidence_risk",
    "localization_risk",
    "CalibrationResult",
    # inference
    "SeqCRCInferencer",
]
