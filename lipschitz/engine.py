"""Training and evaluation loops for the 1-Lipschitz clip classifier.

A faithful port of section 3 of the MNIST 0-vs-8 notebook: HKR loss + Adam, logging
the Kantorovich-Rubinstein term, the hinge term, and accuracy each epoch on both
train and validation. The only structural change is bookkeeping — we keep the
best-validation-accuracy weights and return a ``history`` for plotting.

The ``HKRLoss(alpha, min_margin)`` trades off the KR term (which *maximises* the
margin / Wasserstein-1 separation) against the hinge term (which enforces a minimum
margin), exactly as in the reference notebook. A larger ``alpha`` (→1) weights the
hinge more; ``min_margin`` sets the target margin.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from lipschitz.metrics import binary_accuracy, certified_accuracy_curve, confusion_counts


@dataclass
class History:
    """Per-epoch training/validation metrics, for plotting and the results file."""

    train_loss: list[float] = field(default_factory=list)
    train_kr: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_kr: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _evaluate_outputs(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader; return flat ``(outputs, targets)`` on CPU."""
    model.eval()
    outs, tgts = [], []
    with torch.no_grad():
        for data, target in loader:
            out = model(data.to(device)).detach().cpu().reshape(-1)
            outs.append(out)
            tgts.append(target.reshape(-1))
    return torch.cat(outs), torch.cat(tgts)


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 30,
    alpha: float = 0.98,
    min_margin: float = 1.0,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[object, History]:
    """Train with HKR loss; return ``(best_model, history)``.

    ``best_model`` is ``model`` with the best-validation-accuracy weights loaded back
    in (``model`` is also mutated in place to those weights).
    """
    from deel.torchlip import HKRLoss, KRLoss

    model = model.to(device)
    hkr_loss = HKRLoss(alpha=alpha, min_margin=min_margin)
    kr_loss = KRLoss()
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    history = History()
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        m_kr = m_acc = run_loss = 0.0
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = hkr_loss(output, target)
            loss.backward()
            optimizer.step()

            run_loss += float(loss)
            m_kr += float(kr_loss(output, target))
            m_acc += binary_accuracy(output, target)
            n_batches += 1

        history.train_loss.append(run_loss / n_batches)
        history.train_kr.append(m_kr / n_batches)
        history.train_acc.append(m_acc / n_batches)

        val_out, val_tgt = _evaluate_outputs(model, val_loader, device)
        history.val_loss.append(float(hkr_loss(val_out, val_tgt)))
        history.val_kr.append(float(kr_loss(val_out, val_tgt)))
        v_acc = binary_accuracy(val_out, val_tgt)
        history.val_acc.append(v_acc)

        if v_acc >= best_acc:
            best_acc = v_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {history.train_loss[-1]:.4f} - KR: {history.train_kr[-1]:.4f} - "
                f"acc: {history.train_acc[-1]:.4f} - "
                f"val_loss: {history.val_loss[-1]:.4f} - val_KR: {history.val_kr[-1]:.4f} - "
                f"val_acc: {v_acc:.4f}"
            )

    model.load_state_dict(best_state)
    if verbose:
        print(f"Best val accuracy: {best_acc:.4f}")
    return model, history


def evaluate(
    model,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    radii: torch.Tensor | None = None,
) -> dict:
    """Full evaluation: accuracy, confusion counts, and a certified-accuracy curve.

    Args:
        radii: L2 radii (``[0, 1]`` pixel units) at which to report certified accuracy.
            Defaults to ``linspace(0, 1, 21)``.

    Returns a dict with ``outputs``/``targets`` (flat tensors), ``accuracy``, the four
    confusion counts, ``radii`` and ``certified_accuracy`` (aligned lists).
    """
    if radii is None:
        radii = torch.linspace(0.0, 1.0, 21)
    out, tgt = _evaluate_outputs(model, loader, device)
    curve = certified_accuracy_curve(out, tgt, radii)
    return {
        "outputs": out,
        "targets": tgt,
        "accuracy": binary_accuracy(out, tgt),
        **confusion_counts(out, tgt),
        "radii": radii.tolist(),
        "certified_accuracy": curve.tolist(),
    }
