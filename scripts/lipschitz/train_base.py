"""Train the BASE 1-Lipschitz clip classifier — the simple starting point to improve on.

ONE small network, trained on the exact same 800 images YOLO was trained against
(``data/splits/train.txt``) and evaluated on the same held-out test set
(``data/splits/test.txt``). This is deliberately the simplest thing that works — a
3-conv-block 1-Lipschitz CNN with an HKR loss — so there is a clean baseline to
iterate from in scripts/lipschitz/experiment.py.

Run on the compute engine (from the project root, venv active):

    CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/train_base.py

That's it. Override paths/epochs only if you need to:

    python scripts/lipschitz/train_base.py --epochs 40 --out outputs/lipschitz/base
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
from lipschitz.engine import evaluate, train
from lipschitz.metrics import certified_radius
from lipschitz.model import build_config, build_from_spec, format_summary, save_checkpoint


# ── THE BASE NETWORK — keep it simple; this is what you grow from ─────────────
# 3 conv blocks, each: 3x3 SpectralConv -> GroupSort -> /2 pool. Channels stay
# small (16/32/64). 640 -> 320 -> 160 -> 80, an 8x8 adaptive pool, then a single
# 1-Lipschitz linear to a scalar (sign = clip / no-clip).
BASE_SPEC = [
    ("conv", 16), "act", ("pool", 2),     # 640 -> 320
    ("conv", 32), "act", ("pool", 2),     # 320 -> 160
    ("conv", 64), "act", ("pool", 2),     # 160 ->  80
    ("adaptive_pool", (8, 8)),            #  80 ->   8
    "flatten",                            # 64 * 8 * 8 = 4096 features
    ("linear", 1),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=ROOT / "data" / "splits" / "train.txt",
                   help="image-path split YOLO trained on (default: data/splits/train.txt)")
    p.add_argument("--val", type=Path, default=ROOT / "data" / "splits" / "test.txt",
                   help="held-out split for validation/eval (default: data/splits/test.txt)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "lipschitz" / "base")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.98, help="HKR hinge/KR tradeoff")
    p.add_argument("--min-margin", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── Data: the same splits as YOLO / conformal ─────────────────────────────
    train_paths = ldata.paths_from_split(args.train)
    val_paths = ldata.paths_from_split(args.val)
    tr_pos, tr_neg = ldata.class_balance(train_paths)
    va_pos, va_neg = ldata.class_balance(val_paths)
    print(f"train: {len(train_paths)} imgs ({tr_pos} clip / {tr_neg} none)  <- {args.train}")
    print(f"val:   {len(val_paths)} imgs ({va_pos} clip / {va_neg} none)  <- {args.val}")
    if tr_pos == 0 or tr_neg == 0:
        print("WARNING: training split is single-class — check the split file.")

    with Image.open(train_paths[0]) as im:
        in_hw = (im.size[1], im.size[0])  # (H, W)
    print(f"image size: {in_hw[0]}x{in_hw[1]}")

    train_loader = ldata.make_loader(train_paths, args.batch_size, shuffle=True)
    val_loader = ldata.make_loader(val_paths, args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, summary = build_from_spec(
        BASE_SPEC, backend="torchlip", in_ch=3, in_hw=in_hw, return_summary=True
    )
    print(format_summary(summary, "torchlip"))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss (HKR) + train ────────────────────────────────────────────────────
    from deel.torchlip import HKRLoss, KRLoss
    hkr_loss = HKRLoss(alpha=args.alpha, min_margin=args.min_margin)
    kr_loss = KRLoss()
    model, history = train(
        model, train_loader, val_loader, hkr_loss,
        epochs=args.epochs, lr=args.lr, device=args.device,
        extra_metrics={"KR": lambda out, tgt: float(kr_loss(out, tgt))},
    )

    # ── Persist config + weights (reloadable by eval_clip_classifier.py) ──────
    config = build_config(img_size=in_hw, spec=BASE_SPEC, backend="torchlip")
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    save_checkpoint(model, args.out / "best.pt")

    # ── Final metrics on the held-out split ───────────────────────────────────
    res = evaluate(model, val_loader, device=args.device)
    radii = certified_radius(res["outputs"], res["targets"])
    correct = radii[radii > 0]
    mean_r = float(correct.mean()) if correct.numel() else 0.0

    lines = [
        "BASE 1-Lipschitz clip classifier",
        f"train / val      : {args.train.name} ({len(train_paths)}) / {args.val.name} ({len(val_paths)})",
        f"train balance    : {tr_pos} clip / {tr_neg} none",
        f"epochs           : {args.epochs}   (best val acc {max(history.val['acc']):.4f})",
        f"accuracy (val)   : {res['accuracy']:.4f}",
        f"confusion (clip=+): TP={res['TP']} FP={res['FP']} FN={res['FN']} TN={res['TN']}",
        f"mean cert radius : {mean_r:.4f}   (L2, normalised px, over correct images)",
    ]
    text = "\n".join(lines) + "\n"
    (args.out / "results.txt").write_text(text)
    print("\n" + text)
    print(f"Artifacts in {args.out}  (config.json, best.pt, results.txt)")


if __name__ == "__main__":
    main()
