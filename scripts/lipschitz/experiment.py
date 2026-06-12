"""Lipschitz classifier workbench — edit a layer sequence, pick a backend, see metrics.

This is the one file you work in to answer "how good is a 1-Lipschitz net at the
clip / no-clip question, and how does it compare to an ordinary CNN?". Everything you
are meant to change lives in the four banner blocks below:

    1. ARCHITECTURE  — the layer sequence (SPEC). Rearrange freely.
    2. BACKEND       — torchlip | orthogonium | vanilla (or --compare to run all).
    3. LOSS          — *** write your loss here ***. Left intentionally to you.
    4. TRAIN METRICS — optional extra per-epoch metrics to log.

Then run it on the GPU machine:

    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/lipschitz/experiment.py \
        --n-train 2000 --n-val 400 --epochs 15 --backend torchlip

    # same architecture, all three backends, side-by-side table:
    python scripts/lipschitz/experiment.py --compare --n-train 2000 --epochs 15

Nothing here trains for long by default — the point is a quick, honest read on the
architecture, not a final model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from lipschitz import data as ldata
from lipschitz.engine import evaluate, train
from lipschitz.layers import describe_backend
from lipschitz.metrics import certified_radius
from lipschitz.model import build_from_spec, format_summary


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1. ARCHITECTURE — edit this layer sequence.                              ║
# ║    Items: ("conv", out_ch) | "act" | ("pool", k) |                       ║
# ║           ("adaptive_pool", (h, w)) | "flatten" | ("linear", out)        ║
# ║    Channels and the linear head's in-features are inferred for you, so   ║
# ║    you only ever write output sizes. (full help: lipschitz.layers.SPEC_HELP)
# ╚══════════════════════════════════════════════════════════════════════════╝
SPEC = [
    ("conv", 32),  "act", ("pool", 2),     # 640 -> 320
    ("conv", 64),  "act", ("pool", 2),     # 320 -> 160
    ("conv", 128), "act", ("pool", 2),     # 160 ->  80
    ("conv", 128), "act", ("pool", 2),     #  80 ->  40
    ("conv", 128), "act", ("pool", 2),     #  40 ->  20
    ("adaptive_pool", (4, 4)),
    "flatten",
    ("linear", 1),                         # scalar: sign(output) = clip / no-clip
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3. LOSS  ***  WRITE YOUR LOSS HERE  ***                                  ║
# ║    Return a callable (output[B,1], target[B] in {+1,-1}) -> scalar.      ║
# ║    `backend` is passed in case you want a different objective for the    ║
# ║    unconstrained baseline. The commented lines are ready-made options    ║
# ║    from deel-torchlip — uncomment one, or write your own.                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def make_loss(backend: str):
    # Default: HKR (Wasserstein KR term + hinge), the deel-torchlip objective for
    # 1-Lipschitz binary classification. Swap it for one of the alternatives below,
    # or write your own — any (output[B,1], target[B] in {+1,-1}) -> scalar works.
    from deel.torchlip import HKRLoss
    return HKRLoss(alpha=0.98, min_margin=1.0)          # Wasserstein + hinge
    #
    # from deel.torchlip import KRLoss
    # return KRLoss()                                   # pure margin maximisation
    #
    # def hinge(output, target, margin=1.0):            # plain 1-Lipschitz hinge
    #     return torch.clamp(margin - output.reshape(-1) * target.reshape(-1), min=0).mean()
    # return hinge


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4. TRAIN METRICS — optional extra per-epoch metrics, logged with loss/acc.║
# ║    Each is name -> fn(output, target) -> float. Empty {} is fine.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def make_extra_metrics(backend: str) -> dict:
    metrics: dict = {}
    # from deel.torchlip import KRLoss
    # kr = KRLoss()
    # metrics["KR"] = lambda out, tgt: float(kr(out, tgt))   # mean margin / Wasserstein-1
    return metrics


# ── run one (backend) ─────────────────────────────────────────────────────────

def run_one(backend: str, loaders, in_hw, args) -> dict:
    train_loader, val_loader, test_loader = loaders

    model, summary = build_from_spec(
        SPEC, backend=backend, in_ch=3, in_hw=in_hw, return_summary=True
    )
    print(format_summary(summary, backend))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    parameters    : {n_params:,}")

    loss_fn = make_loss(backend)
    extra = make_extra_metrics(backend)

    model, _hist = train(
        model, train_loader, val_loader, loss_fn,
        epochs=args.epochs, lr=args.lr, device=args.device,
        extra_metrics=extra, verbose=True,
    )

    res = evaluate(model, test_loader, device=args.device)
    radii = certified_radius(res["outputs"], res["targets"])
    correct = radii[radii > 0]
    mean_r = float(correct.mean()) if correct.numel() else 0.0
    return {
        "backend": backend,
        "params": n_params,
        "accuracy": res["accuracy"],
        "TP": res["TP"], "FP": res["FP"], "FN": res["FN"], "TN": res["TN"],
        "mean_cert_radius": mean_r,
        "lipschitz": backend != "vanilla",
    }


def print_results(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("METRICS AT EXIT  (evaluated on the held-out test split)")
    print("=" * 78)
    head = f"{'backend':<13}{'params':>11}  {'acc':>6}  {'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}  {'cert.r':>7}"
    print(head)
    print("-" * len(head))
    for r in rows:
        cert = f"{r['mean_cert_radius']:.4f}" if r["lipschitz"] else "  n/a "
        print(
            f"{r['backend']:<13}{r['params']:>11,}  {r['accuracy']:>6.4f}  "
            f"{r['TP']:>4}{r['FP']:>4}{r['FN']:>4}{r['TN']:>4}  {cert:>7}"
        )
    print("-" * len(head))
    print("acc = clean accuracy; cert.r = mean certified L2 radius over correct images")
    print("(normalised pixel units, comparable to conformal/smoothing sigma; only")
    print("meaningful for 1-Lipschitz backends).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="torchlip", choices=["torchlip", "orthogonium", "vanilla"])
    p.add_argument("--compare", action="store_true",
                   help="run the SAME architecture through all three backends and tabulate")
    p.add_argument("--data-root", type=Path, default=ROOT / "data" / "dataset" / "lipschitz")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--p-clip", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--describe", action="store_true",
                   help="just print which concrete class each role resolves to, then exit")
    args = p.parse_args()

    backends = ["torchlip", "orthogonium", "vanilla"] if args.compare else [args.backend]

    if args.describe:
        for b in backends:
            print(describe_backend(b)); print()
        return

    # ── 2. DATA — balanced synthetic train/val/test on disjoint seeds ──────────
    print("Generating synthetic data ...")
    train_paths = ldata.build_balanced_synthetic(
        args.data_root, args.n_train, seed=args.seed, p_clip=args.p_clip, name="clf_train")
    val_paths = ldata.build_balanced_synthetic(
        args.data_root, args.n_val, seed=args.seed + 1, p_clip=args.p_clip, name="clf_val")
    test_paths = ldata.build_balanced_synthetic(
        args.data_root, args.n_test, seed=args.seed + 2, p_clip=args.p_clip, name="clf_test")

    for tag, paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
        pos, neg = ldata.class_balance(paths)
        print(f"  {tag:<5}: {len(paths):>5} imgs ({pos} clip / {neg} none)")

    from PIL import Image
    with Image.open(train_paths[0]) as im:
        in_hw = (im.size[1], im.size[0])  # (H, W)
    print(f"  image size: {in_hw[0]}x{in_hw[1]}")

    loaders = (
        ldata.make_loader(train_paths, args.batch_size, shuffle=True),
        ldata.make_loader(val_paths, args.batch_size, shuffle=False),
        ldata.make_loader(test_paths, args.batch_size, shuffle=False),
    )

    rows = []
    for b in backends:
        print("\n" + "#" * 78)
        print(f"# backend: {b}")
        print("#" * 78)
        rows.append(run_one(b, loaders, in_hw, args))

    print_results(rows)


if __name__ == "__main__":
    main()
