"""Generate sample synthetic images for visual inspection.

Produces:
  - Two clip-in-upper-track examples (bbox drawn in red)
  - Two clip-in-lower-track examples (bbox drawn in red)
  - Two empty-rails examples
  - Two switch-point examples (forced to contain a switch; one with clip, one without)

Outputs go to `preview/`. Run with: `python preview_synthetic.py`.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset_synthetic import _make_image


# Each entry: (filename, config dict to drive _make_image)
SPECS = [
    ("clip_upper_1",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("upper",)}),
    ("clip_upper_2",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("upper",)}),
    ("clip_lower_1",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("lower",)}),
    ("clip_lower_2",   {"p_clip": 1.0, "p_switch": 0.0, "clip_tracks": ("lower",)}),
    ("no_clip_1",      {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}),
    ("no_clip_2",      {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}),
    ("switch_no_clip", {"p_clip": 0.0, "p_switch": 1.0, "clip_tracks": ("upper", "lower")}),
    ("switch_with_clip", {"p_clip": 1.0, "p_switch": 1.0, "clip_tracks": ("upper", "lower")}),
]


def main():
    out_dir = Path("preview")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=123)

    for name, cfg in SPECS:
        img, bbox = _make_image(rng, cfg)
        pil = Image.fromarray(img)

        # Overlay the bounding box so we can confirm the label geometry
        if bbox is not None:
            draw = ImageDraw.Draw(pil)
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)

        out_path = out_dir / f"{name}.png"
        pil.save(out_path)
        print(f"Wrote {out_path}  ({pil.size[0]}x{pil.size[1]})")


if __name__ == "__main__":
    main()
