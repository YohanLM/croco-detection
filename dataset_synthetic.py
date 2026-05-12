"""Synthetic rails dataset.

Procedurally generates orthogonal LiDAR-style top-down rail images that mimic
the real point-cloud projections in `real_samples/`. See `image_description.md`
for the visual specification this generator targets.

With probability `p_clip` a green "crocodile clip" is placed between the two
rails — that rectangle is the bounding box YOLO must learn to detect.

Implementation choice: numpy + PIL. We only generate up to ~2000 images and the
math is simple per-pixel array work, so torch's GPU machinery would add weight
without speeding anything meaningful up.

Public loader (matches the project's dataset interface):
    f(output_dir) -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Colour palette
# --------------------------------------------------------------------------- #
# Each colour represents a height band in the source LiDAR data, NOT a real
# material colour. See image_description.md for the mapping.
# Stored as float32 so we can add noise without integer overflow before clipping.

WHITE = np.array([255, 255, 255], dtype=np.uint8)         # No-data (no LiDAR return)
BALLAST_RGB = np.array([155, 90, 45], dtype=np.float32)   # Ground level
RAIL_RGB = np.array([105, 45, 30], dtype=np.float32)      # Just above ground (rail head)
SLEEPER_RGB = np.array([75, 40, 25], dtype=np.float32)    # Darker brown stripes
GREEN_RGB = np.array([65, 140, 55], dtype=np.float32)     # Elevated vegetation/debris
CLIP_BODY_RGB = np.array([145, 50, 35], dtype=np.float32) # Clip body — red, at rail height
CLIP_DOT_RGB = np.array([60, 160, 55], dtype=np.float32)  # Clip contact points — green, higher


# --------------------------------------------------------------------------- #
# Layer builders — each one mutates the image array in place
# --------------------------------------------------------------------------- #

def _ballast_texture(rng, size):
    """Build the orange-brown granular ballast that fills the whole image.

    Combines two scales of noise so the texture doesn't look like uniform static:
      * fine per-pixel chromatic noise — the "grain" you see up close
      * coarse block noise — broad regions are slightly brighter or darker,
        mimicking how LiDAR point density varies across the scan
    """
    h = w = size
    fine = rng.normal(0, 22, size=(h, w, 3))

    # Low-frequency variation: generate noise on a coarse grid then upsample.
    block = 16
    coarse = rng.normal(0, 14, size=(h // block + 1, w // block + 1, 3))
    coarse = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)[:h, :w]

    img = BALLAST_RGB + fine + coarse
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_sleepers(rng, img, y_top_rail, y_bot_rail):
    """Draw dark transverse sleeper stripes across the rail band.

    Sleepers are blended (not overwritten) with the ballast underneath so they
    look embedded in the gravel rather than painted on top. They extend slightly
    above the top rail and below the bottom rail to match real samples.
    """
    h, w = img.shape[:2]

    # Sleepers extend a touch beyond the rail pair
    band_top = max(0, y_top_rail - int(h * 0.01))
    band_bot = min(h, y_bot_rail + int(h * 0.01))

    # Spacing every 3.5–5% of width, with a random phase so the pattern doesn't
    # start at the same x in every image
    spacing = int(w * rng.uniform(0.035, 0.05))
    sleeper_w = max(2, int(w * 0.012))
    x = rng.integers(0, spacing)

    while x < w:
        x0 = max(0, x - sleeper_w // 2)
        x1 = min(w, x + sleeper_w // 2)
        region = img[band_top:band_bot, x0:x1].astype(np.float32)
        new_pixels = SLEEPER_RGB + rng.normal(0, 10, size=region.shape)
        # 60/40 blend: sleeper dominates but ballast shows through
        blended = 0.6 * new_pixels + 0.4 * region
        img[band_top:band_bot, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        x += spacing


def _draw_rails(rng, img, y_top_rail, y_bot_rail):
    """Draw two parallel horizontal dark-red rails on top of the sleepers."""
    h, w = img.shape[:2]
    rail_thickness = max(2, int(h * 0.012))

    for y in (y_top_rail, y_bot_rail):
        y0 = max(0, y - rail_thickness // 2)
        y1 = min(h, y + rail_thickness // 2 + 1)
        # Per-pixel noise so the rail isn't a perfectly uniform line
        noise = rng.normal(0, 12, size=(y1 - y0, w, 3))
        img[y0:y1] = np.clip(RAIL_RGB + noise, 0, 255).astype(np.uint8)


def _add_green_noise(rng, img, y_top_rail, y_bot_rail):
    """Scatter small green blobs as hard negatives.

    The model must learn that "green pixel between rails = clip" but
    "green pixel outside rails = ignore." We achieve this by placing most blobs
    outside the inter-rail zone (vegetation alongside the track) and a few
    inside it (tiny debris that should NOT be confused with the clip).
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    n_blobs = rng.integers(25, 70)

    for _ in range(n_blobs):
        if rng.random() < 0.85:
            # Outside the rails: either above top rail or below bottom rail
            if rng.random() < 0.5:
                cy = rng.integers(0, max(1, y_top_rail - 2))
            else:
                cy = rng.integers(min(h - 1, y_bot_rail + 3), h)
        else:
            # Tiny blob inside the inter-rail zone (hard negative)
            if y_bot_rail - y_top_rail > 4:
                cy = rng.integers(y_top_rail + 2, y_bot_rail - 1)
            else:
                continue

        cx = rng.integers(0, w)
        radius = rng.integers(1, 4)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
        n_px = int(mask.sum())
        if n_px:
            colours = GREEN_RGB + rng.normal(0, 20, size=(n_px, 3))
            img[mask] = np.clip(colours, 0, 255).astype(np.uint8)


def _add_crocodile_clip(rng, img, y_top_rail, y_bot_rail):
    """Place a clip between the two rails. Returns bbox (x0, y0, x1, y1).

    The clip is modelled as a small elongated rectangle running perpendicular to
    the rails (taller than wide, since the rails are horizontal). Its body is red
    (at rail height in the LiDAR height map), with a random proportion of green
    dots scattered toward the centre — the contact points / spring that protrude
    slightly higher. Position between the rails is the primary discriminating
    feature; the colour signature is secondary.
    """
    h, w = img.shape[:2]
    inter_h = y_bot_rail - y_top_rail

    if inter_h <= 4:
        return None

    # Position: within the inter-rail zone, in the middle 70% horizontally
    cy = rng.integers(y_top_rail + max(1, inter_h // 5),
                      y_bot_rail - max(1, inter_h // 5))
    cx = rng.integers(int(w * 0.15), int(w * 0.85))

    # Dimensions: narrow (3–7 px wide), tall enough to span most of the gap
    clip_w = int(rng.integers(3, 8))
    clip_h = int(rng.integers(max(4, int(inter_h * 0.5)),
                              max(5, int(inter_h * 0.9))))

    x0 = max(0, cx - clip_w // 2)
    x1 = min(w, x0 + clip_w)
    y0 = max(0, cy - clip_h // 2)
    y1 = min(h, y0 + clip_h)

    bh = y1 - y0
    bw = x1 - x0

    # --- Clip body: red, at rail height ---
    body_noise = rng.normal(0, 15, size=(bh, bw, 3))
    img[y0:y1, x0:x1] = np.clip(CLIP_BODY_RGB + body_noise, 0, 255).astype(np.uint8)

    # --- Green dots: scattered in the centre portion of the body ---
    # A random fraction (20–70%) of pixels in the inner centre band become green,
    # mimicking the contact points / spring that stick up higher than the body.
    green_fraction = rng.uniform(0.2, 0.7)
    # Inner band: middle third horizontally and middle half vertically
    iy0 = y0 + bh // 4
    iy1 = y1 - bh // 4
    ix0 = x0 + bw // 4
    ix1 = x1 - bw // 4
    if iy1 > iy0 and ix1 > ix0:
        n_inner = (iy1 - iy0) * (ix1 - ix0)
        green_mask = rng.random(n_inner) < green_fraction
        inner_region = img[iy0:iy1, ix0:ix1].reshape(-1, 3).copy()
        dot_noise = rng.normal(0, 15, size=(n_inner, 3))
        inner_region[green_mask] = np.clip(
            CLIP_DOT_RGB + dot_noise[green_mask], 0, 255
        ).astype(np.uint8)
        img[iy0:iy1, ix0:ix1] = inner_region.reshape(iy1 - iy0, ix1 - ix0, 3)

    # --- Optional thin wire extending upward ---
    # NOT included in the bounding box — the detector should fire on the clip body
    if rng.random() < 0.6:
        wire_top = max(0, y0 - int(rng.integers(20, 80)))
        wire_x = cx + int(rng.integers(-1, 2))
        wx0 = max(0, wire_x - 1)
        wx1 = min(w, wire_x + 1)
        if y0 > wire_top and wx1 > wx0:
            wn = rng.normal(0, 12, size=(y0 - wire_top, wx1 - wx0, 3))
            img[wire_top:y0, wx0:wx1] = np.clip(
                CLIP_BODY_RGB + wn, 0, 255
            ).astype(np.uint8)

    return (x0, y0, x1, y1)


def _apply_scan_mask(rng, img, y_top_rail, y_bot_rail):
    """Punch irregular white "no-data" patches into the top/bottom edges.

    Real LiDAR scans have sparse data at the edges of the swath. We replicate
    that by replacing pixels with white in irregular blob patterns concentrated
    near the top and bottom of the image. The rail band itself is protected so
    the track structure and any clip remain fully visible.
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]

    # Protect a margin around the rail band — no white holes here
    protected_top = max(0, y_top_rail - int(h * 0.04))
    protected_bot = min(h, y_bot_rail + int(h * 0.04))

    final_mask = np.zeros((h, w), dtype=bool)  # True = becomes white
    n_holes = rng.integers(30, 80)

    for _ in range(n_holes):
        # Bias holes toward the top or bottom of the image
        if rng.random() < 0.5:
            cy = int(rng.integers(0, max(1, protected_top)))
        else:
            cy = int(rng.integers(min(h - 1, protected_bot), h))
        cx = int(rng.integers(0, w))
        radius = int(rng.integers(8, 40))
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
        # Speckled removal so the white area looks ragged, not perfectly round
        speckle = rng.random((h, w)) < 0.7
        final_mask |= circle & speckle

    # Hard guard: never touch the protected rail band
    final_mask[protected_top:protected_bot] = False
    img[final_mask] = WHITE


# --------------------------------------------------------------------------- #
# Compose one image
# --------------------------------------------------------------------------- #

def _make_image(rng, size, p_clip):
    """Generate one synthetic LiDAR-style rail image.

    Layer order (bottom to top):
      1. Ballast texture (the whole image)
      2. Sleepers (dark transverse stripes around the rail band)
      3. Rails (two horizontal lines)
      4. Green vegetation noise (hard negatives, mostly outside the rails)
      5. Optional crocodile clip (between the rails)
      6. White no-data patches (top/bottom edges only)

    Returns (uint8 RGB array, bbox tuple or None).
    """
    # Step 1: ballast everywhere
    img = _ballast_texture(rng, size)

    # Pick the rail geometry up front so every other layer can reference it.
    # Track is centred vertically with a small random jitter, and the two
    # rails are 10–15% of image height apart (standard gauge from overhead).
    y_center = size // 2 + int(rng.integers(-size // 25, size // 25 + 1))
    rail_gap = int(size * rng.uniform(0.10, 0.15))
    y_top_rail = y_center - rail_gap // 2
    y_bot_rail = y_center + rail_gap // 2

    # Steps 2–4: sleepers, then rails on top, then green noise around them
    _draw_sleepers(rng, img, y_top_rail, y_bot_rail)
    _draw_rails(rng, img, y_top_rail, y_bot_rail)
    _add_green_noise(rng, img, y_top_rail, y_bot_rail)

    # Step 5: optional crocodile clip
    bbox = None
    if rng.random() < p_clip:
        bbox = _add_crocodile_clip(rng, img, y_top_rail, y_bot_rail)

    # Step 6: white no-data edges (applied last so it visually overrides ballast/noise)
    _apply_scan_mask(rng, img, y_top_rail, y_bot_rail)

    return img, bbox


# --------------------------------------------------------------------------- #
# Public loader — writes a YOLO-style dataset to disk
# --------------------------------------------------------------------------- #

def load_synthetic_rails(output_dir, n_samples=700, img_size=640, p_clip=0.5, seed=42):
    """Generate (or reuse) a synthetic rails dataset on disk in YOLO format.

    Layout produced under `output_dir`:
        images/rail_00000.png ...
        labels/rail_00000.txt ...   # "0 cx cy w h" normalized, empty = no object

    If the labels directory already has at least `n_samples` files we skip
    regeneration and just return the paths — makes re-runs cheap.

    Args:
        output_dir: root directory for images/ and labels/.
        n_samples: total number of images to generate.
        img_size: square image side length in pixels.
        p_clip: probability that any given image contains a crocodile clip.
        seed: deterministic seed so runs are reproducible.
    """
    output_dir = Path(output_dir)
    images_dst = output_dir / "images"
    labels_dst = output_dir / "labels"
    classes = ["crocodile_clip"]

    # Cache: if enough labels already exist, treat the dataset as ready.
    if labels_dst.exists() and len(list(labels_dst.glob("*.txt"))) >= n_samples:
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        img, bbox = _make_image(rng, img_size, p_clip)
        name = f"rail_{i:05d}"
        Image.fromarray(img).save(images_dst / f"{name}.png")

        # YOLO label: empty file = "background, no object"
        if bbox is None:
            (labels_dst / f"{name}.txt").write_text("")
        else:
            x0, y0, x1, y1 = bbox
            # YOLO format: class cx cy w h, all normalized to image size
            cx = (x0 + x1) / 2 / img_size
            cy = (y0 + y1) / 2 / img_size
            bw = (x1 - x0) / img_size
            bh = (y1 - y0) / img_size
            (labels_dst / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}
