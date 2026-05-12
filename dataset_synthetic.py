"""Synthetic rails dataset.

Procedurally generates images of two parallel "rail" lines on a noisy gray
background. With probability `p_object`, a coloured rectangle ("obstacle") is
placed somewhere along the rails — that rectangle is the bounding box YOLO
must learn to detect.

Exposes a loader matching the project's dataset interface:
    f(output_dir) -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_synthetic_rails(output_dir, n_samples=600, img_size=256, p_object=0.8, seed=42):
    output_dir = Path(output_dir)
    images_dst = output_dir / "images"
    labels_dst = output_dir / "labels"
    classes = ["object"]

    # Skip generation if we already have enough samples on disk
    if labels_dst.exists() and len(list(labels_dst.glob("*.txt"))) >= n_samples:
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        img, bbox = _make_rail_image(rng, img_size, p_object)
        name = f"rail_{i:05d}"
        Image.fromarray(img).save(images_dst / f"{name}.png")

        # YOLO label file — empty means "background image, no object"
        if bbox is None:
            (labels_dst / f"{name}.txt").write_text("")
        else:
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2 / img_size
            cy = (y0 + y1) / 2 / img_size
            bw = (x1 - x0) / img_size
            bh = (y1 - y0) / img_size
            (labels_dst / f"{name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}


def _make_rail_image(rng, size, p_object):
    """Generate one synthetic rail image. Returns (uint8 RGB array, bbox or None)."""
    # 1. Noisy light-gray background
    bg = 230 + rng.normal(0, 12, size=(size, size, 3))
    img_arr = np.clip(bg, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil)

    # 2. Pick rail geometry: small tilt from vertical, random horizontal position
    angle_rad = np.deg2rad(rng.uniform(-15, 15))
    x_center = rng.integers(size // 3, 2 * size // 3)
    y_center = size // 2
    spacing = size // 10  # half-distance between the two rails

    # Unit vector along the rails (downward), and the perpendicular
    dirx, diry = np.sin(angle_rad), np.cos(angle_rad)
    perpx, perpy = np.cos(angle_rad), -np.sin(angle_rad)

    # 3. Draw two parallel rails as long dark lines
    L = size * 2  # long enough to cross the whole image
    for sign in (-1, 1):
        ox = sign * spacing * perpx
        oy = sign * spacing * perpy
        p1 = (x_center + ox - L * dirx, y_center + oy - L * diry)
        p2 = (x_center + ox + L * dirx, y_center + oy + L * diry)
        draw.line([p1, p2], fill=(60, 60, 60), width=3)

    # 4. Maybe place an object on the rails
    bbox = None
    if rng.random() < p_object:
        t = rng.uniform(-0.3, 0.3) * size  # distance along rails from centre
        cx = int(x_center + t * dirx)
        cy = int(y_center + t * diry)

        half = int(rng.integers(12, 28))
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(size, cx + half)
        y1 = min(size, cy + half)

        color = tuple(int(c) for c in rng.integers(50, 200, size=3))
        draw.rectangle([x0, y0, x1, y1], fill=color)
        bbox = (x0, y0, x1, y1)

    return np.array(pil), bbox
