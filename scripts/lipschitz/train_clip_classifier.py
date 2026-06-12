"""Train the 1-Lipschitz HKR "clip / no-clip" classifier.

Ports the MNIST 0-vs-8 notebook to the croco-clip domain. Generates a balanced
synthetic train/val set (clip in ~50 % of images), trains a 1-Lipschitz CNN with the
HKR loss, and writes weights + curves + a results file to the run directory.

Run from the project root on the GPU machine:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/train_clip_classifier.py \
        --epochs 30 --out outputs/lipschitz/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from lipschitz import data as ldata
from lipschitz.engine import train
from lipschitz.model import build_config, build_lip_classifier, export_vanilla, save_checkpoint


def _img_size(image_path: Path) -> tuple[int, int]:
    """``(H, W)`` of an image, by header read."""
    with Image.open(image_path) as im:
        w, h = im.size
    return h, w


def _plot_curves(history, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.train["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(epochs, history.train["loss"], label="train")
    ax[0].plot(epochs, history.val["loss"], label="val")
    ax[0].set_title("HKR loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(epochs, history.train["acc"], label="train")
    ax[1].plot(epochs, history.val["acc"], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(epochs, history.train["KR"], label="train")
    ax[2].plot(epochs, history.val["KR"], label="val")
    ax[2].set_title("KR (margin)"); ax[2].set_xlabel("epoch"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "lipschitz" / "run1")
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "dataset" / "lipschitz",
                   help="where the balanced synthetic splits are generated")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=800)
    p.add_argument("--p-clip", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.98)
    p.add_argument("--min-margin", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Balanced synthetic data — train and val use disjoint seeds.
    print("Generating synthetic data ...")
    train_paths = ldata.build_balanced_synthetic(
        args.data_root, args.n_train, seed=args.seed, p_clip=args.p_clip, name="clf_train"
    )
    val_paths = ldata.build_balanced_synthetic(
        args.data_root, args.n_val, seed=args.seed + 1, p_clip=args.p_clip, name="clf_val"
    )
    tr_pos, tr_neg = ldata.class_balance(train_paths)
    va_pos, va_neg = ldata.class_balance(val_paths)
    print(f"train: {len(train_paths)} ({tr_pos} clip / {tr_neg} none)")
    print(f"val:   {len(val_paths)} ({va_pos} clip / {va_neg} none)")

    h, w = _img_size(train_paths[0])
    print(f"image size: {h}x{w}")

    train_loader = ldata.make_loader(train_paths, args.batch_size, shuffle=True)
    val_loader = ldata.make_loader(val_paths, args.batch_size, shuffle=False)

    # 2. Build + train the 1-Lipschitz model.
    config = build_config(img_size=(h, w))
    model = build_lip_classifier(
        in_ch=config["in_ch"], widths=config["widths"], head_pool=config["head_pool"]
    )

    # Loss + logged margin metric. To experiment with architectures and a custom
    # loss instead, use scripts/lipschitz/experiment.py — this script is the fixed
    # HKR baseline trainer that also persists checkpoints.
    from deel.torchlip import HKRLoss, KRLoss
    hkr_loss = HKRLoss(alpha=args.alpha, min_margin=args.min_margin)
    kr_loss = KRLoss()

    model, history = train(
        model, train_loader, val_loader, hkr_loss,
        epochs=args.epochs, lr=args.lr, device=args.device,
        extra_metrics={"KR": lambda out, tgt: float(kr_loss(out, tgt))},
    )

    # 3. Persist: config, best weights, vanilla export, curves, results.
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    save_checkpoint(model, args.out / "best.pt")

    sample = torch.zeros(1, config["in_ch"], h, w, device=args.device)
    vanilla = export_vanilla(model, sample)
    save_checkpoint(vanilla, args.out / "vanilla.pt")

    _plot_curves(history, args.out / "curves.png")

    best_val = max(history.val["acc"])
    (args.out / "results.txt").write_text(
        "1-Lipschitz HKR clip classifier — training\n"
        f"image size       : {h}x{w}\n"
        f"train / val size : {len(train_paths)} / {len(val_paths)}\n"
        f"train balance    : {tr_pos} clip / {tr_neg} none\n"
        f"val balance      : {va_pos} clip / {va_neg} none\n"
        f"epochs           : {args.epochs}\n"
        f"alpha / margin   : {args.alpha} / {args.min_margin}\n"
        f"lr / batch       : {args.lr} / {args.batch_size}\n"
        f"best val acc     : {best_val:.4f}\n"
        f"final val KR     : {history.val['KR'][-1]:.4f}\n"
    )
    print(f"Done. Artifacts in {args.out}")


if __name__ == "__main__":
    main()
