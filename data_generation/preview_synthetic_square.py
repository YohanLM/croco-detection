"""Generate sample square synthetic images for visual inspection.

Produces a handful of variations:
  - Clip-in-upper-track examples with bbox drawn
  - Clip-in-lower-track examples with bbox drawn
  - No-clip examples showing the surrounding chaos in isolation
  - Switch + clip examples

Outputs go to `preview_square/`. Run: `python preview_synthetic_square.py`.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset_synthetic_square import _make_image_square


SPECS = [
    ("clip_upper_1",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("upper",)}),
    ("clip_upper_2",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("upper",)}),
    ("clip_lower_1",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("lower",)}),
    ("clip_lower_2",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("lower",)}),
    ("no_clip_1",      {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}),
    ("no_clip_2",      {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}),
    ("no_clip_3",      {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}),
    ("switch_clip",    {"p_clip": 1.0, "p_switch": 1.0, "clip_tracks": ("upper", "lower")}),
]


def _save(pil, name, out_dir, bbox=None):
    if bbox is not None:
        draw = ImageDraw.Draw(pil)
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
    out_path = out_dir / f"{name}.png"
    pil.save(out_path)
    print(f"Wrote {out_path}  ({pil.size[0]}x{pil.size[1]})")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "previews" / "preview_square"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=123)

    for name, cfg in SPECS:
        img, bbox = _make_image_square(rng, cfg)
        _save(Image.fromarray(img), name, out_dir, bbox)


if __name__ == "__main__":
    main()
