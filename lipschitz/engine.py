"""Training and evaluation loops for the clip presence classifier.

The loss is **not** baked in here — you pass a ``loss_fn`` (and, optionally, extra
per-batch metrics to log). That keeps the choice of objective yours: HKR, plain
hinge / KR, BCE, anything that maps ``(output[B,1], target[B]) -> scalar``. The loop
only owns bookkeeping: it records loss + accuracy (plus any extra metrics) each
epoch on train and val, and keeps the best-validation-accuracy weights.

Targets follow the ``{+1, -1}`` convention from ``lipschitz.data`` (clip / no-clip),
which is what the Wasserstein-style losses expect; accuracy is ``sign(output)``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch
from torch.utils.data import DataLoader

from lipschitz.metrics import binary_accuracy, certified_accuracy_curve, confusion_counts

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass
class History:
    """Per-epoch metrics. ``train``/``val`` are dicts keyed by metric name.

    Always contains ``"loss"`` and ``"acc"``; any names in ``extra_metrics`` passed
    to ``train`` are added as further keys. Each value is a list (one per epoch).
    """

    train: dict[str, list[float]] = field(default_factory=dict)
    val: dict[str, list[float]] = field(default_factory=dict)

    def _append(self, side: str, values: Mapping[str, float]) -> None:
        target = self.train if side == "train" else self.val
        for k, v in values.items():
            target.setdefault(k, []).append(float(v))


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
    loss_fn: LossFn,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    extra_metrics: Mapping[str, MetricFn] | None = None,
    verbose: bool = True,
) -> tuple[object, History]:
    """Train with the supplied ``loss_fn``; return ``(best_model, history)``.

    Args:
        loss_fn:       your objective, ``(output[B,1], target[B]) -> scalar tensor``.
        extra_metrics: optional ``name -> fn(output, target) -> float`` logged each
                       epoch alongside loss/accuracy (e.g. the KR / margin term).
        epochs, lr:    Adam optimisation hyper-parameters.

    ``best_model`` is ``model`` with the best-val-accuracy weights reloaded (``model``
    is also mutated in place to those weights).
    """
    extra_metrics = dict(extra_metrics or {})
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    history = History()
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        run = {"loss": 0.0, "acc": 0.0, **{k: 0.0 for k in extra_metrics}}
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            run["loss"] += float(loss)
            run["acc"] += binary_accuracy(output, target)
            for name, fn in extra_metrics.items():
                run[name] += float(fn(output, target))
            n_batches += 1

        history._append("train", {k: v / n_batches for k, v in run.items()})

        val_out, val_tgt = _evaluate_outputs(model, val_loader, device)
        val_row = {
            "loss": float(loss_fn(val_out, val_tgt)),
            "acc": binary_accuracy(val_out, val_tgt),
        }
        for name, fn in extra_metrics.items():
            val_row[name] = float(fn(val_out, val_tgt))
        history._append("val", val_row)

        v_acc = val_row["acc"]
        if v_acc >= best_acc:
            best_acc = v_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            extra = " ".join(f"{k}: {history.train[k][-1]:.4f}" for k in extra_metrics)
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {history.train['loss'][-1]:.4f} - acc: {history.train['acc'][-1]:.4f} - "
                f"{extra}{' - ' if extra else ''}"
                f"val_loss: {val_row['loss']:.4f} - val_acc: {v_acc:.4f}"
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
    confusion counts, ``radii`` and ``certified_accuracy`` (aligned lists). Certified
    figures are only meaningful for a 1-Lipschitz backend.
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
