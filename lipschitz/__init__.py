"""1-Lipschitz HKR classifier for the image-level "is there a clip?" question.

A port of deel-torchlip's Wasserstein/HKR binary classifier
(``robustness/wasserstein_classification_MNIST08.ipynb``) to the croco-clip domain,
remodelled as a presence-only binary task. The network is built 1-Lipschitz so its
signed output doubles as a deterministic certified L2 robustness radius — an
image-level counterpart to ``conformal/smoothing``.

Sub-modules:
  * ``data``    — ClipClassificationDataset, loaders, balanced synthetic builder
  * ``layers``  — the common layer vocabulary + the torchlip/orthogonium/vanilla
                  backends (read this first to see what's available)
  * ``model``   — build_from_spec (edit a layer sequence), build_lip_classifier,
                  checkpoint / vanilla-export helpers
  * ``metrics`` — accuracy, certified radius, certified-accuracy curve
  * ``engine``  — loss-agnostic train loop (you pass loss_fn) + evaluate

The fast path: copy ``model.DEFAULT_SPEC``, rearrange the layers, and call
``model.build_from_spec(spec, backend="torchlip" | "orthogonium" | "vanilla")``.
``layers.describe_backend(name)`` prints which concrete class each role maps to.
"""

from lipschitz.data import (
    ClipClassificationDataset,
    build_balanced_synthetic,
    class_balance,
    make_loader,
    paths_from_split,
)
from lipschitz.engine import History, evaluate, train
from lipschitz.layers import (
    BACKENDS,
    SPEC_HELP,
    describe_backend,
    get_backend,
)
from lipschitz.metrics import (
    binary_accuracy,
    certified_accuracy_curve,
    certified_radius,
    confusion_counts,
)
from lipschitz.model import (
    DEFAULT_SPEC,
    build_config,
    build_from_spec,
    build_lip_classifier,
    export_vanilla,
    format_summary,
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
    "BACKENDS",
    "SPEC_HELP",
    "get_backend",
    "describe_backend",
    "DEFAULT_SPEC",
    "build_from_spec",
    "format_summary",
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
