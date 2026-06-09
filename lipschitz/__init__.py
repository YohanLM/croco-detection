"""1-Lipschitz HKR classifier for the image-level "is there a clip?" question.

A port of deel-torchlip's Wasserstein/HKR binary classifier
(``robustness/wasserstein_classification_MNIST08.ipynb``) to the croco-clip domain,
remodelled as a presence-only binary task. The network is built 1-Lipschitz so its
signed output doubles as a deterministic certified L2 robustness radius — an
image-level counterpart to ``conformal/smoothing``.

Sub-modules:
  * ``data``    — ClipClassificationDataset, loaders, balanced synthetic builder
  * ``model``   — build_lip_classifier + checkpoint / vanilla-export helpers
  * ``metrics`` — accuracy, certified radius, certified-accuracy curve
  * ``engine``  — HKR train loop + evaluate
"""

from lipschitz.data import (
    ClipClassificationDataset,
    build_balanced_synthetic,
    class_balance,
    make_loader,
    paths_from_split,
)
from lipschitz.engine import History, evaluate, train
from lipschitz.metrics import (
    binary_accuracy,
    certified_accuracy_curve,
    certified_radius,
    confusion_counts,
)
from lipschitz.model import (
    build_config,
    build_lip_classifier,
    export_vanilla,
    load_checkpoint,
    model_from_config,
    save_checkpoint,
)

__all__ = [
    "ClipClassificationDataset",
    "make_loader",
    "build_balanced_synthetic",
    "paths_from_split",
    "class_balance",
    "build_lip_classifier",
    "build_config",
    "model_from_config",
    "save_checkpoint",
    "load_checkpoint",
    "export_vanilla",
    "binary_accuracy",
    "certified_radius",
    "certified_accuracy_curve",
    "confusion_counts",
    "train",
    "evaluate",
    "History",
]
