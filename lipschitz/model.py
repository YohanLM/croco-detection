"""The 1-Lipschitz convolutional "clip / no-clip" classifier.

This ports the fully-connected HKR classifier from the MNIST 0-vs-8 notebook
(``robustness/wasserstein_classification_MNIST08.ipynb``) to our 640×640 LiDAR-style
rail images, swapping the dense stack for ``deel.torchlip`` 1-Lipschitz **conv** layers.

Why 1-Lipschitz. Every layer is constrained so the whole network ``f : R^d → R`` has
Lipschitz constant ``L = 1`` w.r.t. the input L2 norm. The decision is ``sign(f(x))``,
so the signed output is a *certified margin*: if ``y·f(x) = m > 0`` then **no** L2
perturbation of norm ``< m`` can flip the prediction (``metrics.certified_radius``).
That is a deterministic, Monte-Carlo-free analogue of the randomized-smoothing
certificate in ``conformal/smoothing``.

To stay globally 1-Lipschitz the composition uses only 1-Lipschitz building blocks:

  * ``SpectralConv2d``      — spectrally-normalised conv (σ_max = 1)
  * ``GroupSort2`` / ``MaxMin`` — gradient-norm-preserving activation
  * ``ScaledL2NormPool2d``  — norm-preserving downsampling (the scaled pools, unlike
                              plain avg-pool, keep the Lipschitz constant at 1)
  * ``FrobeniusLinear``     — 1-Lipschitz linear head

Layer class names have drifted slightly across ``deel-torchlip`` releases, so
``_first_available`` resolves each role against whatever the installed version
exposes, preferring the most norm-preserving option. Every fallback listed is itself
1-Lipschitz, so the global guarantee holds regardless of which is picked.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# Default architecture — 5 downsampling stages take 640 → 20 spatial, then an
# adaptive pool to HEAD_POOL before the linear head.
DEFAULT_WIDTHS = (32, 64, 128, 128, 128)
DEFAULT_HEAD_POOL = (4, 4)


# ── Layer-role resolution (tolerant to deel-torchlip version drift) ───────────

def _first_available(module, names: Sequence[str]):
    """Return the first attribute in ``names`` that ``module`` actually exposes.

    Used to pick a concrete ``torchlip`` class for each role (activation / pool)
    without hard-coding a name that a given release may have renamed.
    """
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise AttributeError(
        f"deel-torchlip exposes none of {list(names)} — check the installed version."
    )


def build_lip_classifier(
    in_ch: int = 3,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    head_pool: Sequence[int] = DEFAULT_HEAD_POOL,
):
    """Build the 1-Lipschitz CNN as a ``torchlip.Sequential``.

    Architecture: for each width ``c`` a block of
    ``SpectralConv2d(·→c, k=3, pad=1) → GroupSort2 → ScaledL2NormPool2d(2)`` (each
    block halves the spatial size), then an adaptive 1-Lipschitz pool to
    ``head_pool``, ``Flatten``, and a ``FrobeniusLinear(·→1)`` head.

    With the defaults and a 640×640 input the spatial size goes 640→320→160→80→40→20,
    the adaptive pool brings it to 4×4, and the head sees ``128·4·4 = 2048`` features.

    Args:
        in_ch:     input channels (3 for RGB).
        widths:    output channels per conv block; ``len(widths)`` = #downsamplings.
        head_pool: spatial size fed to the linear head after the adaptive pool.

    Returns:
        A ``torchlip.Sequential`` with scalar output. Lipschitz constant ≈ 1 w.r.t.
        the input L2 norm.
    """
    from deel import torchlip

    Activation = _first_available(torchlip, ("GroupSort2", "MaxMin", "FullSort"))
    Pool = _first_available(torchlip, ("ScaledL2NormPool2d", "ScaledAvgPool2d"))
    AdaptivePool = _first_available(
        torchlip, ("ScaledAdaptiveL2NormPool2d", "ScaledAdaptiveAvgPool2d")
    )

    layers: list[nn.Module] = []
    prev = in_ch
    for c in widths:
        layers.append(torchlip.SpectralConv2d(prev, c, kernel_size=3, padding=1))
        layers.append(Activation())
        layers.append(Pool(kernel_size=2))
        prev = c

    layers.append(AdaptivePool(tuple(head_pool)))
    layers.append(nn.Flatten())
    head_features = prev * int(math.prod(head_pool))
    layers.append(torchlip.FrobeniusLinear(head_features, 1))

    return torchlip.Sequential(*layers)


def build_config(
    in_ch: int = 3,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    head_pool: Sequence[int] = DEFAULT_HEAD_POOL,
    img_size: tuple[int, int] | None = None,
) -> dict:
    """The JSON-serialisable kwargs needed to rebuild the model at eval time."""
    return {
        "in_ch": in_ch,
        "widths": list(widths),
        "head_pool": list(head_pool),
        "img_size": list(img_size) if img_size is not None else None,
    }


def model_from_config(config: dict):
    """Rebuild an (uninitialised) model from a ``build_config`` dict."""
    return build_lip_classifier(
        in_ch=config["in_ch"],
        widths=config["widths"],
        head_pool=config["head_pool"],
    )


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, path) -> None:
    """Save the (still-parametrised) torchlip state dict."""
    torch.save(model.state_dict(), str(path))


def load_checkpoint(config: dict, path, device, sample_input: torch.Tensor | None = None):
    """Rebuild from ``config``, run one forward to materialise the spectral
    parametrisations, then load weights from ``path``.

    ``torchlip`` layers build their spectral/Frobenius parametrisation lazily on the
    first forward pass, so a dummy forward is required before ``load_state_dict`` will
    line up. ``sample_input`` defaults to a zero image at ``config["img_size"]``.
    """
    model = model_from_config(config).to(device)
    if sample_input is None:
        h, w = config["img_size"]
        sample_input = torch.zeros(1, config["in_ch"], h, w, device=device)
    model.eval()
    with torch.no_grad():
        model(sample_input.to(device))
    model.load_state_dict(torch.load(str(path), map_location=device))
    return model


def export_vanilla(model, sample_input: torch.Tensor):
    """Convert to a plain-``torch`` model for fast inference (notebook §4.2).

    ``vanilla_export`` folds the spectral/Frobenius parametrisations into ordinary
    ``nn.Conv2d`` / ``nn.Linear`` weights. It modifies in place, so callers that need
    to keep the trainable model should export a copy. One forward pass must have run
    first to initialise the parametrisations.
    """
    with torch.no_grad():
        model(sample_input)
    return model.vanilla_export()
