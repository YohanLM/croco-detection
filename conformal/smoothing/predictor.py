"""Median-smoothing predictor — a drop-in `PredictionFunction` for the CRC stack.

At inference we replace `f(x)` with the smoothed predictor `g(x, sigma, N)`:
draw `N` Gaussian-noised copies of the image, run the base detector on each,
keep the top-1 box per copy, and return the **coordinate-wise median** of those
boxes. The median votes out spurious noise-triggered detections, yielding a
stable box (methodology Eq. 2).

To the `Calibrator` this wrapper is indistinguishable from a plain
`YoloPredictor`: it satisfies the `PredictionFunction` contract
(`conformal/calibrator.py:45-61`) — `[P, 5]` float32, pixel-xyxy + score,
empties as `zeros(0, 5)` — and exposes `predict_batch`, so dropping it into the
existing pipeline later is a one-line predictor swap.

The key design move is `collect_samples`: it runs the `N`-sample Monte-Carlo
pass **once** and returns the full record (every copy's box + a detection mask),
not just the median. `SmoothedTop1Predictor` returns only `.median`; the
evaluation metrics (`conformal.smoothing.metrics`) and the robustness
certificate (`conformal.smoothing.certificate`) read the rest. The dispersion of
that sample cloud is precisely what flags *unstable* predictions — boxes that
would move (or vanish) under a small input change — so the median (the estimate)
and the instability signal come from one computation.

No-vote rule (detection quorum): some noisy copies may detect nothing. If fewer
than a `quorum` fraction of the `N` copies produce a box, the smoothed
prediction is empty — the median "vote" is no-detection. Otherwise the median is
taken over the detecting copies only.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from PIL import Image

from conformal.calibrator import PredictionFunction
from conformal.smoothing.noise import NoiseFunction, gaussian_noise


# ── Base-predictor capability the wrapper needs ──────────────────────────────

class ArrayPredictor(Protocol):
    """A detector that can run on in-memory `[B, 3, H, W]` `[0, 1]` RGB images.

    `YoloPredictor` satisfies this via `predict_arrays`. The smoothing wrapper
    needs it because noise is injected at the pixel-tensor level, before the
    detector — file paths are not enough.
    """

    def predict_arrays(
        self, images: torch.Tensor, confidence_threshold: float
    ) -> list[torch.Tensor]: ...


# ── The per-image Monte-Carlo record ─────────────────────────────────────────

@dataclass
class SmoothingSamples:
    """The full `N`-sample record behind one smoothed prediction.

    Everything the wrapper and every evaluation metric needs is derived from
    this — collected in a single Monte-Carlo pass per image.

    Attributes:
        coords: `[N, 4]` top-1 box per noisy copy, pixel xyxy. Rows for copies
            that detected nothing are `NaN`.
        scores: `[N]` top-1 confidence per copy. `NaN` where no detection.
        detected: `[N]` bool — which copies produced a box.
        median: the smoothed prediction, `[1, 5]` (xyxy + median score) or
            `[0, 5]` if the quorum/confidence rule rejected it.
        sigma: the noise scale used (carried for the certificate).
    """

    coords: torch.Tensor
    scores: torch.Tensor
    detected: torch.Tensor
    median: torch.Tensor
    sigma: float

    @property
    def n(self) -> int:
        return int(self.detected.shape[0])

    @property
    def n_detected(self) -> int:
        return int(self.detected.sum().item())

    @property
    def detected_coords(self) -> torch.Tensor:
        """`[m, 4]` boxes from the detecting copies only (drops the NaN rows)."""
        return self.coords[self.detected]

    @property
    def detected_scores(self) -> torch.Tensor:
        """`[m]` scores from the detecting copies only."""
        return self.scores[self.detected]


# ── Image loading ────────────────────────────────────────────────────────────

def load_image_chw01(image_path: str) -> torch.Tensor:
    """Load an image as a `[3, H, W]` float tensor in `[0, 1]`, RGB order.

    This is the format the noise functions and `predict_arrays` operate on, so
    the exact pixels we perturb are the pixels the detector sees.
    """
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()         # [3, H, W]


def _top1(preds: torch.Tensor) -> torch.Tensor | None:
    """The single highest-confidence box `[5]` (col 4 = score); None if empty."""
    if preds.numel() == 0:
        return None
    idx = int(torch.argmax(preds[:, 4]).item())
    return preds[idx]


# ── The single Monte-Carlo pass everything is built on ───────────────────────

def collect_samples(
    base: ArrayPredictor,
    image_path: str,
    sigma: float,
    n_samples: int,
    *,
    noise_fn: NoiseFunction = gaussian_noise,
    conf_floor: float = 0.05,
    quorum: float = 0.5,
    conf_threshold: float = 0.30,
    generator: torch.Generator | None = None,
) -> SmoothingSamples:
    """Run the `N`-sample smoothing pass for one image; return the full record.

    Steps (methodology §2.1): load the image, make `N` noisy copies at scale
    `sigma`, run the base detector on all copies in **one** forward pass at the
    permissive `conf_floor`, keep the top-1 box per copy, then apply the median
    + quorum rule to get the smoothed box.

    Args:
        base: any `ArrayPredictor` (e.g. `YoloPredictor`).
        sigma: noise scale in normalized `[0, 1]` pixel units (see `noise.py`).
        n_samples: number of Monte-Carlo copies `N`.
        noise_fn: which perturbation to apply (default Gaussian — the certified one).
        conf_floor: low threshold the base runs at, so weak-but-real detections
            can still vote into the median. The final `conf_threshold` is applied
            to the median score afterwards.
        quorum: minimum fraction of copies that must detect a box; below it the
            smoothed prediction is empty (`[0, 5]`).
        conf_threshold: the operating threshold; the median box is dropped if its
            median score falls below it.
        generator: optional RNG for reproducible noise.

    Returns:
        A `SmoothingSamples` with the per-copy boxes, the detection mask, and the
        final `[1, 5]` / `[0, 5]` smoothed prediction.
    """
    return collect_samples_tensor(
        base, load_image_chw01(image_path), sigma, n_samples,
        noise_fn=noise_fn, conf_floor=conf_floor, quorum=quorum,
        conf_threshold=conf_threshold, generator=generator,
    )


def collect_samples_tensor(
    base: ArrayPredictor,
    image: torch.Tensor,
    sigma: float,
    n_samples: int,
    *,
    noise_fn: NoiseFunction = gaussian_noise,
    conf_floor: float = 0.05,
    quorum: float = 0.5,
    conf_threshold: float = 0.30,
    generator: torch.Generator | None = None,
) -> SmoothingSamples:
    """`collect_samples` for an already-loaded `[3, H, W]` `[0, 1]` RGB image.

    Used when the pixels are produced in memory rather than read from disk — e.g.
    an adversarial example from `conformal.smoothing.attack`. Identical Monte-Carlo
    logic; only the image source differs.
    """
    batch = image.unsqueeze(0).expand(n_samples, -1, -1, -1)  # [N, 3, H, W] (view)
    noisy = noise_fn(batch.clone(), sigma, generator)       # clone: noise writes in place
    per_copy = base.predict_arrays(noisy, conf_floor)       # list of N [P, 5]

    coords = torch.full((n_samples, 4), float("nan"))
    scores = torch.full((n_samples,), float("nan"))
    detected = torch.zeros(n_samples, dtype=torch.bool)
    for i, preds in enumerate(per_copy):
        box = _top1(preds)
        if box is not None:
            coords[i] = box[:4]
            scores[i] = box[4]
            detected[i] = True

    median = _smoothed_median(coords, scores, detected, quorum, conf_threshold)
    return SmoothingSamples(
        coords=coords, scores=scores, detected=detected, median=median, sigma=sigma
    )


def _smoothed_median(
    coords: torch.Tensor,
    scores: torch.Tensor,
    detected: torch.Tensor,
    quorum: float,
    conf_threshold: float,
) -> torch.Tensor:
    """Apply the quorum + coordinate-wise median + confidence rule.

    `torch.median` returns an actual order statistic (the lower-middle element
    for even counts, no averaging) — the right choice for a median-smoothing
    estimate, and what the certificate's order-statistic bounds assume.
    """
    empty = torch.zeros((0, 5), dtype=torch.float32)
    n = detected.shape[0]
    n_det = int(detected.sum().item())
    if n == 0 or n_det == 0 or (n_det / n) < quorum:
        return empty

    det_coords = coords[detected]                      # [m, 4]
    det_scores = scores[detected]                      # [m]
    med_coords = det_coords.median(dim=0).values       # [4]
    med_score = det_scores.median()                    # scalar
    if float(med_score) < conf_threshold:
        return empty
    return torch.cat([med_coords, med_score.reshape(1)]).reshape(1, 5).float()


# ── The drop-in PredictionFunction ───────────────────────────────────────────

class SmoothedTop1Predictor:
    """Implements `PredictionFunction` via coordinate-wise median smoothing.

    Wraps any `ArrayPredictor` base and exposes the same `__call__` /
    `predict_batch` surface the `Calibrator` already uses, so it is a drop-in
    replacement for `YoloPredictor` in the existing pipeline.

    Args:
        base: the wrapped detector (must support `predict_arrays`).
        n_samples: number of Monte-Carlo noisy copies `N`.
        noise_scale: the operating `sigma` (normalized `[0, 1]` units).
        noise_fn: perturbation to apply (default Gaussian — the certified one).
        quorum: detection-quorum fraction for the no-vote rule.
        conf_floor: permissive threshold the base runs at per copy.
        seed: optional base seed; noise is made deterministic **per image** (via
            a stable hash of the path) so `predict_batch` and per-image
            `__call__` produce identical results regardless of call order.
    """

    def __init__(
        self,
        base: ArrayPredictor,
        n_samples: int,
        noise_scale: float,
        *,
        noise_fn: NoiseFunction = gaussian_noise,
        quorum: float = 0.5,
        conf_floor: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.base = base
        self.n_samples = n_samples
        self.sigma = noise_scale
        self.noise_fn = noise_fn
        self.quorum = quorum
        self.conf_floor = conf_floor
        self.seed = seed

    def _generator_for(self, image_path: str) -> torch.Generator | None:
        """A per-image RNG seeded from `(seed, path)` — stable, order-independent."""
        if self.seed is None:
            return None
        salt = zlib.crc32(image_path.encode("utf-8")) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed((self.seed * 1_000_003 + salt) & 0x7FFF_FFFF_FFFF_FFFF)
        return g

    def samples_for(
        self, image_path: str, confidence_threshold: float
    ) -> SmoothingSamples:
        """The full `SmoothingSamples` for one image — used by the eval metrics."""
        return collect_samples(
            self.base, image_path, self.sigma, self.n_samples,
            noise_fn=self.noise_fn, conf_floor=self.conf_floor,
            quorum=self.quorum, conf_threshold=confidence_threshold,
            generator=self._generator_for(image_path),
        )

    def __call__(
        self, image_path: str, confidence_threshold: float
    ) -> torch.Tensor:
        return self.samples_for(image_path, confidence_threshold).median

    def certify(
        self,
        image_path: str,
        confidence_threshold: float,
        *,
        epsilon: float = 0.1,
        tol_px: float = 2.0,
        conf: float = 0.0,
    ):
        """The smoothed box **plus its per-output robustness certificate**.

        Unlike `__call__` (which returns just the box, to keep the `Calibrator`
        contract pure), this returns a `CertifiedPrediction`: the robust decision
        bundled with GT-free guarantees — the certified edge band at `epsilon`,
        the detection-existence radius, and the localization radius for tolerance
        `tol_px`. Use this at deployment when each output must ship certified.
        """
        # Local import: certificate.py imports this module, so importing it at
        # top level would form a cycle.
        from conformal.smoothing.certificate import certify_samples
        return certify_samples(
            self.samples_for(image_path, confidence_threshold),
            epsilon=epsilon, tol_px=tol_px, conf=conf,
        )

    def certify_batch(
        self,
        image_paths: list[str],
        confidence_threshold: float,
        *,
        epsilon: float = 0.1,
        tol_px: float = 2.0,
        conf: float = 0.0,
    ) -> list:
        """`certify` over a list of images — a `CertifiedPrediction` each."""
        return [
            self.certify(p, confidence_threshold,
                         epsilon=epsilon, tol_px=tol_px, conf=conf)
            for p in image_paths
        ]

    def predict_batch(
        self, image_paths: list[str], confidence_threshold: float
    ) -> list[torch.Tensor]:
        """One smoothing pass per image; each batches its `N` copies internally.

        Per-image batching (N copies in one forward pass) is what keeps the MC
        cost manageable. Images are processed independently so frames of
        differing resolutions are handled correctly.
        """
        return [self(p, confidence_threshold) for p in image_paths]


# ── Protocol-conformance assertion ───────────────────────────────────────────

if TYPE_CHECKING:
    _check_predictor: PredictionFunction = SmoothedTop1Predictor.__new__(
        SmoothedTop1Predictor
    )
