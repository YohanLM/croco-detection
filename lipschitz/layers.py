"""The layer vocabulary — one common set of roles, three interchangeable backends.

This is the file to read to see *what building blocks are available* and which
concrete class each one maps to in each library. A network here is just a **layer
spec** (a list, see ``SPEC_HELP``); a *backend* turns that same spec into either a
1-Lipschitz network (``deel-torchlip`` or ``orthogonium``) or an ordinary CNN
(``vanilla``), so the very same architecture can be trained three ways and compared.

The common roles
----------------
======================  ====================================================
role (spec keyword)     what it is
======================  ====================================================
``conv``                k×k convolution, default k=3 s=1 p=1 (size-preserving)
``act``                 gradient-norm-preserving activation (GroupSort/MaxMin)
``pool``                k× spatial downsample
``adaptive_pool``       force the spatial size to a fixed (h, w)
``flatten``             [B,C,H,W] -> [B, C·H·W]
``linear``              fully-connected layer
======================  ====================================================

How each backend fills those roles
----------------------------------
* ``torchlip``    — ``SpectralConv2d`` / ``GroupSort2`` / scaled-L2 pools /
                    ``FrobeniusLinear``, wrapped in ``torchlip.Sequential`` (so
                    ``vanilla_export`` works). Globally 1-Lipschitz.
* ``orthogonium`` — ``AdaptiveOrthoConv2d`` / ``MaxMin`` / ``OrthoLinear``.
                    Orthogonium concentrates on *orthogonal* conv/linear and
                    activations; for pooling we use the dependency-free
                    ``ScaledAvgPool2d`` below (provably 1-Lipschitz). Globally
                    1-Lipschitz. **Import paths drift between orthogonium
                    releases** — every role is resolved against a list of
                    candidates (``_import_first``); if a name has moved, add it
                    to the relevant ``*_CANDIDATES`` list and re-run
                    ``describe_backend('orthogonium')`` to confirm.
* ``vanilla``     — plain ``nn.Conv2d`` / ``nn.ReLU`` / ``nn.MaxPool2d`` /
                    ``nn.Linear``. **Not** Lipschitz-constrained — the baseline
                    you compare the certified networks against.

Pooling note (why the scaled pools)
------------------------------------
Plain non-overlapping average pooling over a k×k window has L2 operator norm
``1/k`` (each output is the mean of ``k²`` inputs). Multiplying the output by
``k`` makes it *exactly* 1-Lipschitz. ``ScaledAvgPool2d`` / ``ScaledAdaptiveAvgPool2d``
below do precisely that, so the Lipschitz guarantee is preserved without depending
on any library's pooling implementation.
"""

from __future__ import annotations

import importlib
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dependency-free 1-Lipschitz pooling ───────────────────────────────────────

class ScaledAvgPool2d(nn.Module):
    """k×k non-overlapping average pool, scaled by ``k`` so it is 1-Lipschitz (L2)."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        return k * F.avg_pool2d(x, kernel_size=k, stride=k)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, scale={self.kernel_size}"


class ScaledAdaptiveAvgPool2d(nn.Module):
    """Adaptive avg pool to ``(h, w)``, scaled by ``sqrt(n)`` (n = inputs/output)
    so it is 1-Lipschitz (L2) whenever the pooling windows don't overlap."""

    def __init__(self, output_size: Sequence[int]) -> None:
        super().__init__()
        self.output_size = (int(output_size[0]), int(output_size[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        oh, ow = self.output_size
        n = (h * w) / (oh * ow)
        return math.sqrt(max(n, 1.0)) * F.adaptive_avg_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


# ── orthogonium role resolution (tolerant to version drift) ───────────────────
# Each entry is "module.path:AttributeName"; the first importable one wins.
ORTHO_CONV_CANDIDATES = (
    "orthogonium.layers:AdaptiveOrthoConv2d",
    "orthogonium.layers:OrthoConv2d",
    "orthogonium.layers.conv.AOC:AdaptiveOrthoConv2d",
)
ORTHO_LINEAR_CANDIDATES = (
    "orthogonium.layers:OrthoLinear",
    "orthogonium.layers:UnitNormLinear",
    "orthogonium.layers.linear:OrthoLinear",
)
ORTHO_ACT_CANDIDATES = (
    "orthogonium.layers.custom_activations:MaxMin",
    "orthogonium.layers:MaxMin",
    "orthogonium.layers.custom_activations:GroupSort2",
)


def _import_first(candidates: Sequence[str]):
    """Return the first importable ``"module:attr"`` in ``candidates`` (else raise)."""
    tried = []
    for spec in candidates:
        mod_name, _, attr = spec.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - any import failure -> try next
            tried.append(f"{spec} ({type(exc).__name__})")
            continue
        cls = getattr(mod, attr, None)
        if cls is not None:
            return cls
        tried.append(f"{spec} (no attr {attr})")
    raise ImportError(
        "None of the candidate classes could be imported:\n  "
        + "\n  ".join(tried)
        + "\nEdit the *_CANDIDATES lists in lipschitz/layers.py for your "
          "installed version."
    )


def _torchlip_attr(names: Sequence[str]):
    """First attribute that the installed ``deel.torchlip`` actually exposes."""
    from deel import torchlip

    for name in names:
        cls = getattr(torchlip, name, None)
        if cls is not None:
            return cls
    raise AttributeError(f"deel-torchlip exposes none of {list(names)}.")


# ── Backends ──────────────────────────────────────────────────────────────────
# A backend resolves the six roles to concrete constructors. Each constructor
# takes plain ints/tuples so the spec builder (model.build_from_spec) stays
# backend-agnostic. Construction is lazy: importing this module never imports
# torchlip or orthogonium.

class _Backend:
    """Base class; subclasses fill ``conv/linear/act/pool/adaptive_pool`` and
    declare whether the result is Lipschitz-certified."""

    name = "base"
    lipschitz = False

    def conv(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        raise NotImplementedError

    def linear(self, in_f, out_f):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def pool(self, kernel_size):
        raise NotImplementedError

    def adaptive_pool(self, output_size):
        raise NotImplementedError

    def flatten(self):
        return nn.Flatten()

    def sequential(self, modules):
        return nn.Sequential(*modules)

    def resolved(self) -> dict[str, str]:
        """``role -> concrete class name`` actually in use (for ``describe``)."""
        raise NotImplementedError


class TorchlipBackend(_Backend):
    name = "torchlip"
    lipschitz = True

    def conv(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        Conv = _torchlip_attr(("SpectralConv2d",))
        return Conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def linear(self, in_f, out_f):
        Linear = _torchlip_attr(("FrobeniusLinear", "SpectralLinear"))
        return Linear(in_f, out_f)

    def act(self):
        Act = _torchlip_attr(("GroupSort2", "MaxMin", "FullSort"))
        return Act()

    def pool(self, kernel_size):
        # torchlip ships norm-preserving pools; fall back to our scaled pool.
        try:
            Pool = _torchlip_attr(("ScaledL2NormPool2d", "ScaledAvgPool2d"))
            return Pool(kernel_size=kernel_size)
        except AttributeError:
            return ScaledAvgPool2d(kernel_size)

    def adaptive_pool(self, output_size):
        # torchlip's scaled adaptive L2 pool only supports global (1,1) output.
        # For any other size use our scaled avg pool (1-Lipschitz for
        # non-overlapping windows), so arbitrary head sizes stay valid.
        if tuple(output_size) == (1, 1):
            try:
                Pool = _torchlip_attr(
                    ("ScaledAdaptiveL2NormPool2d", "ScaledAdaptiveAvgPool2d")
                )
                return Pool((1, 1))
            except AttributeError:
                pass
        return ScaledAdaptiveAvgPool2d(output_size)

    def sequential(self, modules):
        from deel import torchlip

        # torchlip.Sequential is required for vanilla_export / Lipschitz tracking.
        return torchlip.Sequential(*modules)

    def resolved(self) -> dict[str, str]:
        def nm(fn):
            try:
                return type(fn()).__name__
            except Exception as exc:  # noqa: BLE001
                return f"<error: {type(exc).__name__}>"
        return {
            "conv": type(self.conv(1, 1)).__name__,
            "act": nm(self.act),
            "pool": type(self.pool(2)).__name__,
            "adaptive_pool": type(self.adaptive_pool((1, 1))).__name__,
            "linear": type(self.linear(1, 1)).__name__,
        }


class OrthogoniumBackend(_Backend):
    name = "orthogonium"
    lipschitz = True

    def conv(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        Conv = _import_first(ORTHO_CONV_CANDIDATES)
        return Conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def linear(self, in_f, out_f):
        Linear = _import_first(ORTHO_LINEAR_CANDIDATES)
        return Linear(in_f, out_f)

    def act(self):
        Act = _import_first(ORTHO_ACT_CANDIDATES)
        return Act()

    def pool(self, kernel_size):
        # Orthogonium has no canonical norm-preserving pool across versions;
        # the scaled avg pool is provably 1-Lipschitz and keeps the guarantee.
        return ScaledAvgPool2d(kernel_size)

    def adaptive_pool(self, output_size):
        return ScaledAdaptiveAvgPool2d(output_size)

    def sequential(self, modules):
        # Orthogonium layers are ordinary nn.Modules; plain Sequential is fine.
        return nn.Sequential(*modules)

    def resolved(self) -> dict[str, str]:
        return {
            "conv": type(self.conv(1, 1)).__name__,
            "act": type(self.act()).__name__,
            "pool": type(self.pool(2)).__name__,
            "adaptive_pool": type(self.adaptive_pool((1, 1))).__name__,
            "linear": type(self.linear(1, 1)).__name__,
        }


class VanillaBackend(_Backend):
    """Ordinary, *unconstrained* CNN — the comparison baseline.

    Same spec, no Lipschitz machinery: real conv/ReLU/maxpool/linear. Certified
    radii are meaningless here (the network is not 1-Lipschitz); use it for the
    clean-accuracy / parameter-count comparison only.
    """

    name = "vanilla"
    lipschitz = False

    def conv(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def linear(self, in_f, out_f):
        return nn.Linear(in_f, out_f)

    def act(self):
        return nn.ReLU(inplace=True)

    def pool(self, kernel_size):
        return nn.MaxPool2d(kernel_size=kernel_size)

    def adaptive_pool(self, output_size):
        return nn.AdaptiveAvgPool2d(tuple(output_size))

    def resolved(self) -> dict[str, str]:
        return {
            "conv": "Conv2d",
            "act": "ReLU",
            "pool": "MaxPool2d",
            "adaptive_pool": "AdaptiveAvgPool2d",
            "linear": "Linear",
        }


BACKENDS = {
    "torchlip": TorchlipBackend,
    "orthogonium": OrthogoniumBackend,
    "vanilla": VanillaBackend,
}


def get_backend(name: str) -> _Backend:
    """Instantiate a backend by name (``torchlip`` / ``orthogonium`` / ``vanilla``)."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKENDS)}")
    return BACKENDS[name]()


def describe_backend(name: str) -> str:
    """Human-readable mapping of each role to the concrete class for ``name``.

    Resolving the classes actually imports the library, so this doubles as a quick
    'is my install wired up correctly?' check — run it once on the GPU machine.
    """
    backend = get_backend(name)
    try:
        mapping = backend.resolved()
        body = "\n".join(f"    {role:<14}-> {cls}" for role, cls in mapping.items())
    except Exception as exc:  # noqa: BLE001
        body = f"    <could not resolve: {type(exc).__name__}: {exc}>"
    tag = "1-Lipschitz" if backend.lipschitz else "NOT Lipschitz (baseline)"
    return f"backend '{name}' [{tag}]\n{body}"


# ── Spec mini-language documentation ──────────────────────────────────────────
SPEC_HELP = """\
A layer spec is a list. Each item is either a bare role string or a tuple:

    "act"                       activation
    "flatten"                   flatten before the linear head
    ("conv", out_ch)            3x3 stride-1 conv, output channels = out_ch
    ("conv", out_ch, opts)      opts is a dict, e.g. {"kernel_size":3,"stride":2,"padding":1}
    ("pool", k)                 downsample by k (k x k)
    ("adaptive_pool", (h, w))   force spatial size to (h, w)
    ("linear", out_f)           fully-connected layer, output features = out_f

Channels and linear in-features are inferred automatically from the running shape,
so you only ever write *output* sizes. A typical presence classifier:

    SPEC = [
        ("conv", 32), "act", ("pool", 2),
        ("conv", 64), "act", ("pool", 2),
        ("conv", 128), "act", ("pool", 2),
        ("adaptive_pool", (4, 4)),
        "flatten",
        ("linear", 1),          # scalar output: sign(output) = clip / no-clip
    ]
"""
