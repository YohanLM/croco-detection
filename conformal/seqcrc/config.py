"""SeqCRC configuration (spec Section 0, 9, 11.config).

One immutable `SeqCRCConfig` holds every a-priori choice of the procedure —
the two target error rates, the localization hyperparameters, the prefilter
floor, and the bisection budget. Nothing here may ever be tuned on the
calibration data; that is exactly what the finite-sample guarantee assumes.

The only data-dependent check is `validate(n)`, run at calibration start: the
SeqCRC guarantee (spec Section 9) requires

    alpha_loc >= alpha_cnf + B_loc / (n + 1) = alpha_cnf + 1 / (n + 1)

so the localization budget must dominate the confidence budget by at least one
calibration-sample's worth of slack. `validate` fails loudly otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MarginMode = Literal["additive", "multiplicative"]


@dataclass(frozen=True)
class SeqCRCConfig:
    """Immutable bundle of the a-priori SeqCRC parameters.

    Attributes:
        alpha_cnf: target confidence (FNR) error rate at the confidence step.
        alpha_loc: target localization error rate — the operative end-to-end
            target. Must satisfy `alpha_loc >= alpha_cnf + 1/(n+1)`.
        tau_pix: pixel-superposition threshold in `(0, 1]` — a true box counts
            as "covered" once at least this fraction of its area is overlapped
            by its matched, expanded prediction.
        margin_mode: localization expansion geometry, "additive" (uniform pixel
            margin) or "multiplicative" (margin scales with box size).
        lambda_bar_loc: upper bound of the localization parameter space
            `Lambda_loc = [0, lambda_bar_loc]`. For "additive" this is a pixel
            count above any clip dimension; for "multiplicative" a unitless
            scale (e.g. 2.0 doubles each side).
        prefilter: confidence floor applied identically at calibration and
            inference (spec Section 7 pre-step). Boxes below it are dropped.
        bisection_steps: number of binary-search steps for `lambda_loc_plus`
            (spec Section 7.3); ~20-30 gives sub-pixel precision.
    """

    alpha_cnf: float
    alpha_loc: float
    tau_pix: float
    margin_mode: MarginMode = "additive"
    lambda_bar_loc: float = 100.0
    prefilter: float = 1e-3
    bisection_steps: int = 25

    def validate(self, n: int) -> None:
        """Assert the spec's required parameter relationship for `n` samples.

        Raises:
            ValueError: if `alpha_loc < alpha_cnf + 1/(n+1)` (guarantee void),
                or if any parameter is outside its admissible range.
        """
        if not 0.0 < self.tau_pix <= 1.0:
            raise ValueError(f"tau_pix must be in (0, 1], got {self.tau_pix}")
        if self.margin_mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"margin_mode must be 'additive' or 'multiplicative', "
                f"got {self.margin_mode!r}"
            )
        if self.lambda_bar_loc <= 0:
            raise ValueError(f"lambda_bar_loc must be > 0, got {self.lambda_bar_loc}")
        required = self.alpha_cnf + 1.0 / (n + 1)
        if self.alpha_loc < required:
            raise ValueError(
                "SeqCRC guarantee void: need "
                f"alpha_loc >= alpha_cnf + 1/(n+1) = {required:.4f}, "
                f"but alpha_loc = {self.alpha_loc:.4f} "
                f"(alpha_cnf = {self.alpha_cnf:.4f}, n = {n}). "
                "Raise alpha_loc or lower alpha_cnf."
            )
