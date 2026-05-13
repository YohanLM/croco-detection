"""Synthetic rails dataset — square images with surrounding variation.

Generates 640×640 LiDAR-style images where the railway band is placed somewhere
within the vertical middle third, and the rest of the image is filled with the
variation seen in the raw `real_samples/` exports:

  - White no-data zones (the dominant background)
  - Brown ballast strips of varying thickness around the track, with ragged
    edges where the ballast meets the white
  - Large and small green vegetation patches (irregular blobs)
  - Red rock/debris patches (smaller)
  - Long thin green lines and elongated green motifs (poles, fences, strips)

The actual railway band itself — rails, sleepers, motifs, clip — is delegated
to the existing `dataset_synthetic.py` helpers, which already accept arbitrary
image widths and rail y-coordinates. Only the geometry picker, ballast
texture, and outer surroundings are new here.

Public loader matches the project's dataset interface:
    f(output_dir, config=..., n_samples=..., seed=...)
        -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

from pathlib import Path

import numpy as np
from PIL import Image

import data_generation.dataset_synthetic as ds


# --------------------------------------------------------------------------- #
# Image dimensions
# --------------------------------------------------------------------------- #
SQ_SIZE = 640
RAIL_BAND_H = 91  # rails span band_y0+2 .. band_y0+91 (incl. thickness of rail 4)

# Rail top y-positions WITHIN the band (relative offsets)
RAIL_TOPS_REL = (2, 27, 64, 89)


# --------------------------------------------------------------------------- #
# Extra palette (on top of dataset_synthetic.py)
# --------------------------------------------------------------------------- #
WHITE_RGB     = np.array([255, 255, 255], dtype=np.uint8)
GREEN_DARK    = np.array([45, 110, 40],  dtype=np.float32)   # darker vegetation
GREEN_BRIGHT  = np.array([95, 195, 80],  dtype=np.float32)   # brighter vegetation
RED_PATCH_RGB = np.array([170, 60, 25],  dtype=np.float32)   # rocky red blobs


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
# Mirror the configs in dataset_synthetic.py — same semantics for clip / switch
# rates, but applied to the square layout. The defaults are tuned for a
# realistic mix of surroundings.
CONFIGS = {
    "test_sparse":          {"p_clip": 0.10, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "test_dense":           {"p_clip": 0.50, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "train_two_tracks":     {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "train_with_switches":  {"p_clip": 0.30, "p_switch": 0.20, "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "experiment_15c_5s":    {"p_clip": 0.15, "p_switch": 0.05, "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "c30_m15":              {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.15, "clip_tracks": ("upper", "lower")},
}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _pick_band_y0(rng, size):
    """Top y of the rail band, kept inside the middle third.

    The band centre lies between size/3 and 2*size/3, so the rail band never
    sits above the upper third-mark or below the lower third-mark.
    """
    centre_min = size // 3
    centre_max = 2 * size // 3
    centre = int(rng.integers(centre_min, centre_max + 1))
    return centre - RAIL_BAND_H // 2


def _pick_rails_at(rng, band_y0):
    """Rail (y0, y1) pairs anchored at band_y0 — same jitter rules as the
    non-square dataset, just offset.
    """
    nominal_t = int(rng.choice([2, 3]))
    shift = int(rng.integers(-2, 3))
    rails = []
    for top in RAIL_TOPS_REL:
        jitter = int(rng.integers(-2, 3))
        y0 = band_y0 + top + shift + jitter
        rt = nominal_t + (1 if rng.random() < 0.25 else 0)
        rt = max(2, min(3, rt))
        rails.append((y0, y0 + rt))
    return rails, nominal_t


# --------------------------------------------------------------------------- #
# Square ballast texture (same look as the rectangular version)
# --------------------------------------------------------------------------- #

def _ballast_full(rng, size):
    """Same dotted dark-reddish-brown texture as dataset_synthetic, square."""
    h = w = size
    darkening = np.abs(rng.normal(0, 0.10, size=(h, w, 1))).astype(np.float32)
    brightness = np.clip(1.0 - darkening, 0.15, 1.0)
    tint = rng.normal(0, 3, size=(h, w, 3)).astype(np.float32)
    img = ds.BACKGROUND_RGB * brightness + tint
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# White outside the ballast strip (with irregular edges)
# --------------------------------------------------------------------------- #

def _white_outside_ballast(rng, img, band_y0, band_y1):
    """Paint white over rows above and below the rail band, leaving a
    randomly-thick ballast strip with a wavy boundary.

    The rail band itself (band_y0..band_y1) is always preserved.
    """
    h, w = img.shape[:2]
    # Ballast strip is biased toward thick — the band is rarely the only
    # ballast in the image. Roughly: average extent ~150 px, so ~70 % of
    # the image stays ballast on average.
    above_extent = int(rng.integers(40, 280))
    below_extent = int(rng.integers(40, 280))

    # Per-column boundary y. Each column gets a wavy offset on top of a smoothed
    # low-frequency wave so the edge looks ragged-but-coherent, not pure noise.
    def wavy_extent(base):
        if base == 0:
            return np.zeros(w, dtype=np.int32)
        low = rng.normal(0, 15, size=(w // 18 + 2,))
        low = np.repeat(low, 18)[:w]
        fine = rng.normal(0, 7, size=w)
        return np.clip(base + low + fine, 0, base * 2 + 20).astype(np.int32)

    above = wavy_extent(above_extent)
    below = wavy_extent(below_extent)

    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    top_thresh = (band_y0 - above)[None, :]
    bot_thresh = (band_y1 + below)[None, :]
    white_mask = (yy < top_thresh) | (yy >= bot_thresh)

    # Edge speckle — punch a few extra background dots inside the white margin
    # and a few white dots inside the ballast margin near the boundary, so the
    # boundary doesn't look like a hard line.
    boundary_y_top = band_y0 - above
    boundary_y_bot = band_y1 + below
    for x in range(w):
        # Speckle just above the boundary on the white side
        for y in range(max(0, boundary_y_top[x] - 5), boundary_y_top[x]):
            if 0 <= y < h and rng.random() < 0.25:
                white_mask[y, x] = False
        # Speckle just below
        for y in range(boundary_y_bot[x], min(h, boundary_y_bot[x] + 5)):
            if 0 <= y < h and rng.random() < 0.25:
                white_mask[y, x] = False
        # White flecks inside the ballast near the boundary
        for y in range(boundary_y_top[x], min(h, boundary_y_top[x] + 4)):
            if 0 <= y < h and rng.random() < 0.15:
                white_mask[y, x] = True
        for y in range(max(0, boundary_y_bot[x] - 4), boundary_y_bot[x]):
            if 0 <= y < h and rng.random() < 0.15:
                white_mask[y, x] = True

    img[white_mask] = WHITE_RGB


# --------------------------------------------------------------------------- #
# Vegetation patches (large blobs + small clumps)
# --------------------------------------------------------------------------- #

def _draw_irregular_blob(rng, img, cx, cy, radius, colour,
                          axis_ratio=1.0, angle=0.0, density=0.85,
                          chroma_jitter=22):
    """Soft-edged elliptical blob with per-pixel chromatic jitter and dropout."""
    h, w = img.shape[:2]
    if radius < 1:
        return
    # Compute bounding box
    rb = int(radius * max(1.0, 1.0 / axis_ratio)) + 2
    y_lo, y_hi = max(0, cy - rb), min(h, cy + rb + 1)
    x_lo, x_hi = max(0, cx - rb), min(w, cx + rb + 1)
    if y_hi <= y_lo or x_hi <= x_lo:
        return
    yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    dx = xx - cx
    dy = yy - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    # Soft radial fall-off
    dist = np.sqrt((rx) ** 2 + (ry / max(0.01, axis_ratio)) ** 2)
    soft = np.exp(-(dist / radius) ** 2)
    # Stochastic mask: probability = soft * density
    prob = soft * density
    keep = rng.random(prob.shape) < prob
    n = int(keep.sum())
    if n == 0:
        return
    colours = colour + rng.normal(0, chroma_jitter, size=(n, 3))
    region = img[y_lo:y_hi, x_lo:x_hi]
    region[keep] = np.clip(colours, 0, 255).astype(np.uint8)
    img[y_lo:y_hi, x_lo:x_hi] = region


def _add_vegetation(rng, img, band_y0, band_y1):
    """Scatter green vegetation across the image.

    All patches keep clear of the rail band (with a small margin) so they
    can't bleed into the centre and obscure the railway. Above the band the
    patches are also capped in size — large vegetation only appears below
    the band, where it's more realistic (track-side undergrowth).

    Mix:
      * 0–2 medium blobs *below the band only* (radius 15–35 px)
      * 4–10 small blobs anywhere outside the band (radius 5–14 px)
      * 15–40 tiny clumps anywhere outside the band (radius 2–6 px)
    """
    h, w = img.shape[:2]
    margin = 8     # keep patch centres at least this far from the band
    band_top = band_y0 - margin
    band_bot = band_y1 + margin

    def pick_y_above():
        if band_top <= 0:
            return None
        return int(rng.integers(0, band_top))

    def pick_y_below():
        if band_bot >= h:
            return None
        return int(rng.integers(band_bot, h))

    def pick_y_outside():
        above_room = max(0, band_top)
        below_room = max(0, h - band_bot)
        if above_room == 0 and below_room == 0:
            return None
        if rng.random() * (above_room + below_room) < above_room:
            return pick_y_above()
        return pick_y_below()

    # Medium blobs — restricted to below the band so they never appear above the tracks
    for _ in range(int(rng.integers(0, 3))):
        cy = pick_y_below()
        if cy is None:
            continue
        cx = int(rng.integers(0, w))
        radius = int(rng.integers(15, 35))
        axis_ratio = float(rng.uniform(0.4, 2.5))
        angle = float(rng.uniform(0, np.pi))
        colour = GREEN_DARK if rng.random() < 0.5 else GREEN_BRIGHT
        _draw_irregular_blob(rng, img, cx, cy, radius, colour,
                              axis_ratio=axis_ratio, angle=angle, density=0.75)

    # Small blobs — anywhere outside the band
    for _ in range(int(rng.integers(4, 11))):
        cy = pick_y_outside()
        if cy is None:
            continue
        cx = int(rng.integers(0, w))
        radius = int(rng.integers(5, 15))
        axis_ratio = float(rng.uniform(0.4, 2.0))
        angle = float(rng.uniform(0, np.pi))
        colour = GREEN_DARK if rng.random() < 0.5 else GREEN_BRIGHT
        _draw_irregular_blob(rng, img, cx, cy, radius, colour,
                              axis_ratio=axis_ratio, angle=angle, density=0.85)

    # Tiny clumps
    for _ in range(int(rng.integers(15, 41))):
        cy = pick_y_outside()
        if cy is None:
            continue
        cx = int(rng.integers(0, w))
        radius = int(rng.integers(2, 7))
        colour = GREEN_DARK if rng.random() < 0.4 else GREEN_BRIGHT
        _draw_irregular_blob(rng, img, cx, cy, radius, colour,
                              axis_ratio=1.0, angle=0, density=0.85)


# --------------------------------------------------------------------------- #
# Red patches (rocks / debris)
# --------------------------------------------------------------------------- #

def _add_red_patches(rng, img, band_y0, band_y1):
    """Scatter small red patches, mostly in the ballast strip.

    Patches stay clear of the rail band so they don't visually overlap
    with the rails and clip.
    """
    h, w = img.shape[:2]
    margin = 6
    band_top = band_y0 - margin
    band_bot = band_y1 + margin

    n = int(rng.integers(3, 18))
    for _ in range(n):
        above_room = band_top
        below_room = h - band_bot
        if above_room <= 0 and below_room <= 0:
            continue
        if rng.random() < 0.5 and band_top > 0:
            cy = int(rng.integers(max(0, band_y0 - 80), band_top))
        elif band_bot < h:
            cy = int(rng.integers(band_bot, min(h, band_y1 + 80)))
        else:
            cy = int(rng.integers(0, band_top))
        cx = int(rng.integers(0, w))
        radius = int(rng.integers(2, 10))
        axis_ratio = float(rng.uniform(0.5, 2.0))
        angle = float(rng.uniform(0, np.pi))
        _draw_irregular_blob(rng, img, cx, cy, radius, RED_PATCH_RGB,
                              axis_ratio=axis_ratio, angle=angle, density=0.8)


# --------------------------------------------------------------------------- #
# Long thin green features (poles, fences, vegetation strips)
# --------------------------------------------------------------------------- #

def _add_long_green_features(rng, img, band_y0, band_y1):
    """Long thin horizontal green lines / elongated motifs.

    Always horizontal (matching real-sample vegetation strips). Their y is
    chosen outside the rail band so they never visually overlap the rails.
    """
    h, w = img.shape[:2]
    margin = 8
    band_top = band_y0 - margin
    band_bot = band_y1 + margin

    n = int(rng.integers(0, 4))
    for _ in range(n):
        thickness = int(rng.integers(2, 6))
        colour = GREEN_DARK if rng.random() < 0.5 else GREEN_BRIGHT

        # Pick a horizontal y strictly outside the rail band
        above_room = band_top
        below_room = h - band_bot
        if above_room <= 0 and below_room <= 0:
            continue
        if rng.random() * (above_room + below_room) < above_room and band_top > 0:
            cy = int(rng.integers(0, band_top))
        elif band_bot < h:
            cy = int(rng.integers(band_bot, h))
        else:
            continue

        x_left = int(rng.integers(0, w // 3))
        x_right = int(rng.integers(2 * w // 3, w))
        for x in range(x_left, x_right):
            # Slight vertical wobble keeps the stripe from being a perfect line
            wobble = int(round(rng.normal(0, 0.7)))
            yc = cy + wobble
            y0 = max(0, yc - thickness // 2)
            y1 = min(h, y0 + thickness)
            if y1 > y0:
                seg = colour + rng.normal(0, 15, size=(y1 - y0, 3))
                alpha = rng.uniform(0.4, 1.0, size=(y1 - y0, 1))
                bg = img[y0:y1, x].astype(np.float32)
                img[y0:y1, x] = np.clip(
                    alpha * seg + (1 - alpha) * bg, 0, 255
                ).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Compose one square image
# --------------------------------------------------------------------------- #

def _make_image_square(rng, config):
    """Generate one square synthetic image.

    Layer order:
      1. Full-image ballast texture
      2. White outside the rail band's local ballast strip (with ragged edges)
      3. Surrounding features: red patches, vegetation, long green features
      4. The rail band content (rails, sleepers, motifs, inter-rail features,
         optional switch and clip) — drawn on top of everything, so the band
         is always cleanly visible.

    Returns (uint8 HxWx3 RGB array, bbox tuple or None).
    """
    size = SQ_SIZE
    band_y0 = _pick_band_y0(rng, size)
    band_y1 = band_y0 + RAIL_BAND_H

    # 1. Ballast everywhere
    img = _ballast_full(rng, size)

    # 2. Replace rows outside the ballast strip with white (ragged edge)
    _white_outside_ballast(rng, img, band_y0, band_y1)

    # 3. Surrounding features (drawn under the rail band so they don't
    # obscure rails / clip)
    _add_red_patches(rng, img, band_y0, band_y1)
    _add_vegetation(rng, img, band_y0, band_y1)
    _add_long_green_features(rng, img, band_y0, band_y1)

    # 4. Rail band content — same per-element functions as the rectangular
    # generator, but with rails offset by band_y0. The functions already use
    # img.shape for width so they adapt to 640 px automatically.
    rails, _ = _pick_rails_at(rng, band_y0)

    # Subtle red speckle inside the ballast strip (matches the non-square look)
    _add_red_background_speckle(rng, img, band_y0, band_y1)

    # Optional switch point (drawn under the sleepers visually)
    if rng.random() < config["p_switch"]:
        ds._draw_switch(rng, img, rails)

    ds._draw_sleepers(rng, img, rails)
    ds._draw_rails(rng, img, rails)
    ds._add_rail_motifs(rng, img, rails, p_motif=config.get("p_motif", 0.05))
    ds._add_inter_rail_features(rng, img, rails)

    bbox = None
    if rng.random() < config["p_clip"]:
        track = str(rng.choice(config["clip_tracks"]))
        bbox = ds._add_crocodile_clip(rng, img, rails, track)

    return img, bbox


def _add_red_background_speckle(rng, img, band_y0, band_y1):
    """Small red speckle inside the ballast strip only.

    A toned-down version of ds._add_red_background_noise that limits the
    speckle to the band area (we don't want red dots all over the white).
    """
    h, w = img.shape[:2]
    n = int(rng.integers(40, 100))
    pad = 60
    y_lo, y_hi = max(0, band_y0 - pad), min(h, band_y1 + pad)
    if y_hi <= y_lo:
        return
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(y_lo, y_hi))
        sz = int(rng.integers(1, 3))
        y0 = max(0, cy); y1 = min(h, cy + sz)
        x0 = max(0, cx); x1 = min(w, cx + sz)
        nn = (y1 - y0) * (x1 - x0)
        if nn:
            col = ds.RED_NOISE_RGB + rng.normal(0, 22, size=(nn, 3))
            img[y0:y1, x0:x1] = np.clip(
                col.reshape(y1 - y0, x1 - x0, 3), 0, 255
            ).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Public loader — writes a YOLO-style dataset to disk
# --------------------------------------------------------------------------- #

def load_synthetic_rails_square(output_dir, config="train_two_tracks",
                                 n_samples=700, seed=42):
    """Generate a square synthetic rails dataset on disk in YOLO format.

    Layout produced under `output_dir/<config>/`:
        images/rail_00000.png ...   (640x640 PNGs)
        labels/rail_00000.txt ...   ("0 cx cy w h" normalised, empty = no clip)

    Args:
        output_dir: root directory. A per-config subdir is created underneath.
        config:     name of a configuration in CONFIGS, or a config dict.
        n_samples:  total number of images to generate.
        seed:       deterministic seed.
    """
    if isinstance(config, str):
        if config not in CONFIGS:
            raise ValueError(f"Unknown config '{config}'. Available: {list(CONFIGS)}")
        cfg = CONFIGS[config]
        cfg_name = config
    else:
        cfg = config
        cfg_name = cfg.get("name", "custom")

    root = Path(output_dir) / cfg_name
    images_dst = root / "images"
    labels_dst = root / "labels"
    classes = ["crocodile_clip"]

    if labels_dst.exists() and len(list(labels_dst.glob("*.txt"))) >= n_samples:
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        img, bbox = _make_image_square(rng, cfg)
        name = f"rail_{i:05d}"
        Image.fromarray(img).save(images_dst / f"{name}.png")

        if bbox is None:
            (labels_dst / f"{name}.txt").write_text("")
        else:
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2 / SQ_SIZE
            cy = (y0 + y1) / 2 / SQ_SIZE
            bw = (x1 - x0) / SQ_SIZE
            bh = (y1 - y0) / SQ_SIZE
            (labels_dst / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}
