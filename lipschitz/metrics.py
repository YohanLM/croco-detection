"""Metrics for the 1-Lipschitz clip classifier — accuracy and *certified* robustness.

The headline quantity this framework buys us is the **certified L2 radius**. For a
1-Lipschitz network ``f`` with decision ``sign(f(x))`` and true label ``y ∈ {+1,-1}``:

    margin   m(x) = y · f(x)
    radius   r(x) = max(m(x), 0)

If ``m(x) > 0`` the point is correctly classified and *no* input perturbation ``δ``
with ``‖δ‖₂ < m(x)`` can change ``sign(f(x+δ))`` — because ``|f(x+δ) - f(x)| ≤ ‖δ‖₂``
by the Lipschitz bound (``L = 1``). So the margin **is** the certified radius, in the
same ``[0, 1]`` pixel units as the smoothing ``sigma``. No Monte-Carlo needed.

``certified_accuracy_curve`` reports, per radius ``r``, the fraction of the set that is
*both* correct and certified to that radius (``m(x) ≥ r``). At ``r = 0`` it reduces to
clean accuracy; it is monotonically non-increasing in ``r``.
"""

from __future__ import annotations

import torch


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).float()


def margins(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Signed margin ``y · f(x)`` per sample, ``[N]``. Positive ⇔ correct."""
    return _flat(outputs) * _flat(targets)


def binary_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where ``sign(f(x)) == sign(y)``.

    A sample exactly on the boundary (``f(x) == 0``) counts as incorrect, matching
    ``margin ≥ r`` with ``r > 0`` having zero certified radius there.
    """
    return float((margins(outputs, targets) > 0).float().mean())


def certified_radius(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample certified L2 radius ``max(y·f(x), 0)``, ``[N]`` (0 if misclassified)."""
    return torch.clamp(margins(outputs, targets), min=0.0)


def certified_accuracy_curve(
    outputs: torch.Tensor, targets: torch.Tensor, radii: torch.Tensor
) -> torch.Tensor:
    """Certified accuracy at each radius in ``radii``.

    Returns ``[len(radii)]`` with entry ``k`` = fraction of samples whose margin is
    ``≥ radii[k]`` (i.e. correct *and* certified to that radius).
    """
    m = margins(outputs, targets)                       # [N]
    radii = radii.reshape(-1).to(m)                     # [R]
    return (m[None, :] >= radii[:, None]).float().mean(dim=1)  # [R]


def confusion_counts(
    outputs: torch.Tensor, targets: torch.Tensor
) -> dict[str, int]:
    """Image-level TP / FP / FN / TN at the natural threshold ``f(x) = 0``.

    Positive = "clip present" (``y = +1``). Lets the classifier be compared head-to-head
    with the detector's image-level TP/FP/FN/TN counts.
    """
    out = _flat(outputs)
    tgt = _flat(targets)
    pred_pos = out > 0
    true_pos = tgt > 0
    return {
        "TP": int((pred_pos & true_pos).sum()),
        "FP": int((pred_pos & ~true_pos).sum()),
        "FN": int((~pred_pos & true_pos).sum()),
        "TN": int((~pred_pos & ~true_pos).sum()),
    }
