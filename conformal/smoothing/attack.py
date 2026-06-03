"""l2-bounded PGD attack for the empirical robustness check (methodology §5).

Median smoothing comes with a *certificate* (`certificate.py`); this module is
the *empirical* counterpart — it actually attacks the orthophotos and lets the
evaluation script confirm that the smoothed detector degrades gracefully where
the raw detector collapses, and that the conformal risk still holds on perturbed
data.

The attack is an untargeted "make the object vanish" PGD: it descends a
detection-suppression surrogate (the maximum class confidence over all anchors)
inside an l2 ball around the clean image. Suppressing detection is exactly the
failure smoothing's detection-quorum is meant to resist, so it is the right
stress test.

Gradients flow through the raw detection module (`DetectionModel`, i.e. the
`nn.Module` behind a `YoloPredictor`), whose eval-mode forward returns decoded
predictions `[B, 4+nc, A]` (xywh + per-class probabilities) — differentiable,
unlike the post-NMS `Results`. We avoid a full label-assignment loss by attacking
the confidence channel directly.

Budget convention: `epsilon` is the **total l2 norm** of the perturbation over
the whole image tensor (all pixels and channels), in normalized `[0, 1]` units.
This is a different, coarser scale than the per-pixel `sigma` of the smoothing
noise; relate them via `epsilon_total ~= sigma * sqrt(3 * H * W)` if a like-for-
like comparison is wanted. The input `H, W` must be divisible by 32 (resize
upstream — the evaluation script handles this).
"""

from __future__ import annotations

import torch
from torch import nn


def _max_confidence(det_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Surrogate: the maximum class confidence over all anchors (a scalar/batch).

    `det_module(x)` in eval mode returns decoded predictions; the class scores
    occupy channels `4:` and are already sigmoid-activated, so their max is the
    detector's "best guess this frame contains an object" — the quantity an
    attacker drives down to make the box disappear.
    """
    out = det_module(x)
    preds = out[0] if isinstance(out, (tuple, list)) else out  # [B, 4+nc, A]
    return preds[:, 4:, :].amax(dim=(1, 2))                     # [B]


def pgd_l2(
    det_module: nn.Module,
    image: torch.Tensor,
    *,
    epsilon: float,
    steps: int = 20,
    step_size: float | None = None,
) -> torch.Tensor:
    """Return an l2-bounded adversarial copy of `image` (suppress detection).

    Args:
        det_module: the raw `DetectionModel` (`YoloPredictor.model.model`).
        image: clean image `[1, 3, H, W]` in `[0, 1]`, `H, W` divisible by 32.
        epsilon: total l2 perturbation budget over the whole tensor.
        steps: PGD iterations.
        step_size: per-step l2 size; defaults to `2.5 * epsilon / steps` (the
            standard PGD heuristic ensuring the ball can be traversed).

    Returns:
        Perturbed image `[1, 3, H, W]` in `[0, 1]`, detached, on the input's
        original device.
    """
    device = next(det_module.parameters()).device
    x0 = image.to(device).detach()
    step_size = step_size if step_size is not None else 2.5 * epsilon / max(steps, 1)

    # Start at a small random point inside the ball (standard PGD init).
    delta = torch.randn_like(x0)
    delta = _project_l2(delta, epsilon)
    x = torch.clamp(x0 + delta, 0.0, 1.0).detach()

    was_training = det_module.training
    det_module.eval()
    for _ in range(steps):
        x.requires_grad_(True)
        conf = _max_confidence(det_module, x).sum()
        grad, = torch.autograd.grad(conf, x)
        # Descend the confidence (suppress detection): step along -grad.
        g_norm = grad.flatten().norm() + 1e-12
        x = x.detach() - step_size * grad / g_norm
        # Project back into the l2 ball around x0, then onto the valid image box.
        delta = _project_l2(x - x0, epsilon)
        x = torch.clamp(x0 + delta, 0.0, 1.0).detach()
    if was_training:
        det_module.train()
    return x.detach()


def _project_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project a perturbation onto the total-l2 ball of radius `epsilon`."""
    norm = delta.flatten().norm()
    if float(norm) <= epsilon or float(norm) == 0.0:
        return delta
    return delta * (epsilon / norm)
