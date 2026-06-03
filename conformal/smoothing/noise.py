"""Pluggable input-noise functions for randomized (median) smoothing.

A `NoiseFunction` perturbs a batch of images so the base detector can be run on
many noisy copies of the same frame; the coordinate-wise median of the resulting
top-1 boxes is the smoothed prediction (see `conformal.smoothing.predictor`).

Convention — **strict contract** (mirrors the predictor's pixel contract):

  - Images are `[B, 3, H, W]` float tensors with values in `[0, 1]`, RGB order.
    This is the format `YoloPredictor.predict_arrays` expects.
  - `sigma` is the noise scale in **normalized pixel-value units**, i.e. on the
    same `[0, 1]` scale as the pixels. `sigma = 0.05` perturbs each channel by
    ~5 % of full range (≈ 12.75 on the usual 0–255 scale). Operationally useful
    values are roughly `0.0`–`0.25`; `0.0` is a no-op (identity).
  - The returned tensor has the same shape/dtype and is re-clipped to `[0, 1]`
    so it stays a valid image.

The Gaussian noise is the one the certificate theory in
`conformal.smoothing.certificate` is derived for (the smoothed median is the
50th percentile of `f(x + N(0, sigma^2 I))`). The uniform / impulse variants are
provided as alternative robustness knobs for empirical comparison only — they do
**not** carry the Gaussian certificate.

Reproducibility: every function accepts an optional `generator` (a
`torch.Generator`) so a sweep can fix the seed and get identical noise across
sigma values, isolating the effect of sigma from Monte-Carlo randomness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch


class NoiseFunction(Protocol):
    """Perturb a batch of `[0, 1]` RGB images: `(images, sigma, generator) -> images`.

    Output MUST share the input shape and dtype and stay clipped to `[0, 1]`.
    `sigma = 0.0` MUST be the identity so a sweep can include the clean baseline.
    """

    def __call__(
        self,
        images: torch.Tensor,
        sigma: float,
        generator: torch.Generator | None = ...,
    ) -> torch.Tensor: ...


def gaussian_noise(
    images: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Additive i.i.d. Gaussian noise `N(0, sigma^2)`, clipped to `[0, 1]`.

    The canonical smoothing noise: the smoothed coordinate-wise median is the
    median of `f(x + delta)` with `delta ~ N(0, sigma^2 I)`, which is what the
    median-smoothing certificate in `certificate.py` certifies.
    """
    if sigma == 0.0:
        return images
    noise = torch.randn(
        images.shape, dtype=images.dtype, device=images.device, generator=generator
    )
    return torch.clamp(images + sigma * noise, 0.0, 1.0)


def uniform_noise(
    images: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Additive uniform noise on `[-a, a]` matched to std `sigma` (`a = sigma*sqrt(3)`).

    Alternative robustness knob — NOT covered by the Gaussian certificate. The
    half-width `a = sqrt(3) * sigma` makes its standard deviation equal to the
    Gaussian's at the same `sigma`, so the two are comparable in dispersion.
    """
    if sigma == 0.0:
        return images
    a = sigma * (3.0 ** 0.5)
    u = torch.rand(
        images.shape, dtype=images.dtype, device=images.device, generator=generator
    )
    noise = (2.0 * u - 1.0) * a
    return torch.clamp(images + noise, 0.0, 1.0)


def impulse_noise(
    images: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Salt-and-pepper impulse noise: a `sigma` fraction of pixels forced to 0/1.

    Here `sigma` is reinterpreted as the **corruption probability** per pixel
    (clamped to `[0, 1]`); half the corrupted pixels go to black, half to white.
    Models dead/hot sensor pixels or compression speckle. NOT covered by the
    Gaussian certificate — empirical comparison only.
    """
    if sigma == 0.0:
        return images
    p = min(max(sigma, 0.0), 1.0)
    out = images.clone()
    probs = torch.rand(
        images.shape, dtype=images.dtype, device=images.device, generator=generator
    )
    pepper = probs < (p / 2.0)
    salt = probs > (1.0 - p / 2.0)
    out[pepper] = 0.0
    out[salt] = 1.0
    return out


# ── Protocol-conformance assertions ──────────────────────────────────────────

if TYPE_CHECKING:
    _check_gaussian: NoiseFunction = gaussian_noise
    _check_uniform: NoiseFunction = uniform_noise
    _check_impulse: NoiseFunction = impulse_noise
