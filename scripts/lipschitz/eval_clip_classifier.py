"""Evaluate a trained 1-Lipschitz clip classifier on the YOLO test split.

Loads a run produced by ``train_clip_classifier.py`` and evaluates it on
``data/splits/test.txt`` (the same 800-image split the detector uses), reporting
image-level accuracy + confusion and the certified-accuracy-vs-radius curve. The
certified radius is in ``[0, 1]`` pixel-L2 units — directly comparable to the
``sigma`` in ``conformal/smoothing``.

Run from the project root on the GPU machine:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/eval_clip_classifier.py \
        --run outputs/lipschitz/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from lipschitz import data as ldata
from lipschitz.engine import evaluate
from lipschitz.metrics import certified_radius
from lipschitz.model import load_checkpoint


def _plot_cert_curve(radii, cert_acc, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(radii, cert_acc, marker="o", ms=3)
    ax.set_xlabel("certified L2 radius (normalised pixel units)")
    ax.set_ylabel("certified accuracy")
    ax.set_title("Certified accuracy vs radius")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "outputs" / "lipschitz" / "run1")
    p.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "test.txt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    config = json.loads((args.run / "config.json").read_text())
    model = load_checkpoint(config, args.run / "best.pt", args.device)

    paths = ldata.paths_from_split(args.split)
    n_pos, n_neg = ldata.class_balance(paths)
    loader = ldata.make_loader(paths, args.batch_size, shuffle=False)

    res = evaluate(model, loader, device=args.device)
    radii = certified_radius(res["outputs"], res["targets"])
    correct = radii[radii > 0]
    mean_r = float(correct.mean()) if correct.numel() else 0.0
    median_r = float(correct.median()) if correct.numel() else 0.0

    _plot_cert_curve(res["radii"], res["certified_accuracy"], args.run / "cert_curve.png")

    lines = [
        "1-Lipschitz HKR clip classifier — evaluation",
        f"split            : {args.split}  ({len(paths)} imgs, {n_pos} clip / {n_neg} none)",
        f"accuracy         : {res['accuracy']:.4f}",
        f"confusion (clip=+): TP={res['TP']} FP={res['FP']} FN={res['FN']} TN={res['TN']}",
        f"mean cert radius : {mean_r:.4f}   (over correctly-classified images)",
        f"median cert radius: {median_r:.4f}",
        "",
        "certified accuracy vs L2 radius (normalised pixel units):",
    ]
    for r, ca in zip(res["radii"], res["certified_accuracy"]):
        lines.append(f"  r={r:.3f}  cert_acc={ca:.4f}")
    text = "\n".join(lines) + "\n"

    (args.run / "eval_results.txt").write_text(text)
    print(text)
    print(f"Wrote {args.run / 'eval_results.txt'} and cert_curve.png")


if __name__ == "__main__":
    main()
