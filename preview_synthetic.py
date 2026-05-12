"""Generate 5 sample synthetic images for visual inspection.

Two of the images will contain a crocodile clip (with its bounding box drawn
in red so you can verify the label is correct), three will be empty rails.

Outputs go to `preview/`. Run with: `python preview_synthetic.py`.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset_synthetic import _make_image


# 2 with clip (p_clip=1.0 forces it on), 3 without (p_clip=0.0 forces it off)
SPECS = [
    ("with_clip_1", 1.0),
    ("with_clip_2", 1.0),
    ("no_clip_1", 0.0),
    ("no_clip_2", 0.0),
    ("no_clip_3", 0.0),
]


def main():
    out_dir = Path("preview")
    out_dir.mkdir(exist_ok=True)

    img_size = 640
    # Shared RNG so successive images don't all look the same
    rng = np.random.default_rng(seed=123)

    for name, p_clip in SPECS:
        img, bbox = _make_image(rng, img_size, p_clip)
        pil = Image.fromarray(img)

        # Overlay the bounding box on clipped images so we can confirm the
        # label geometry matches what we see in the image
        if bbox is not None:
            draw = ImageDraw.Draw(pil)
            x0, y0, x1, y1 = bbox
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        out_path = out_dir / f"{name}.png"
        pil.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
