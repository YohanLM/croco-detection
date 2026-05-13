"""Synthetic rails dataset.

Procedurally generates orthogonal LiDAR-style top-down rail images that mimic
the real point-cloud projections in `real_samples_cropped/`. Images are
570x100 — the same size as the cropped real samples — and contain two parallel
rail tracks (4 rails total) running horizontally.

With probability `p_clip` a crocodile clip is placed between the two rails of
one of the track sets — that rectangle is the bounding box YOLO must learn to
detect.

See `image_description.md` for the visual specification this generator targets.

Public loader (matches the project's dataset interface):
    f(output_dir, config=..., n_samples=..., seed=...)
        -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Image dimensions
# --------------------------------------------------------------------------- #
IMG_W = 570
IMG_H = 100

# Base top-edge y positions for the four rails (anchored at top of image).
# Top-to-top spacings: 25, 37, 25.
RAIL_TOPS = (2, 27, 64, 89)


# --------------------------------------------------------------------------- #
# Colour palette — derived from analysis of the real samples
# --------------------------------------------------------------------------- #
BACKGROUND_RGB = np.array([93, 46, 1],   dtype=np.float32)   # Dark reddish-brown ballast
RAIL_RGB       = np.array([244, 134, 23], dtype=np.float32)  # Bright orange-red rails
SLEEPER_RGB    = np.array([8, 6, 3],     dtype=np.float32)   # Near-black sleepers
GREEN_RGB      = np.array([60, 160, 55], dtype=np.float32)   # Green clip dot / debris
RED_NOISE_RGB  = np.array([200, 70, 25], dtype=np.float32)   # Red speckle on ballast


# --------------------------------------------------------------------------- #
# Dataset configurations
# --------------------------------------------------------------------------- #
# Each config is a separate generated split. Six configurations defined in
# image_description.md. clip_tracks controls which track sets can host the
# clip; p_switch controls how often a switch point appears.
CONFIGS = {
    "test_sparse":         {"p_clip": 0.10, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "test_dense":          {"p_clip": 0.50, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "train_two_tracks":    {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "train_with_switches": {"p_clip": 0.30, "p_switch": 0.20, "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "train_upper_only":    {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper",)},
    "train_any_track":     {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "experiment_15c_5s":   {"p_clip": 0.15, "p_switch": 0.05, "p_motif": 0.05, "clip_tracks": ("upper", "lower")},
    "c30_m15":             {"p_clip": 0.30, "p_switch": 0.0,  "p_motif": 0.15, "clip_tracks": ("upper", "lower")},
}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _pick_geometry(rng):
    """Pick the rail row positions and thickness for this image.

    Returns: list of (y0_inclusive, y1_exclusive) per rail, and a nominal rail
    thickness. A global y-shift plus small per-rail jitter give variance while
    preserving the overall ladder structure. Individual rail thickness can also
    vary slightly so not every rail is the same height.
    """
    nominal_t = int(rng.choice([2, 3]))
    shift = int(rng.integers(-2, 3))   # -2..+2
    rails = []
    for top in RAIL_TOPS:
        jitter = int(rng.integers(-2, 3))   # per-rail jitter -2..+2
        y0 = max(0, top + shift + jitter)
        # Allow occasional thicker rails (3 px when nominal is 2, etc.)
        rt = nominal_t + (1 if rng.random() < 0.25 else 0)
        rt = max(2, min(3, rt))
        y1 = min(IMG_H, y0 + rt)
        rails.append((y0, y1))
    return rails, nominal_t


def _gap(top_rail, bot_rail):
    """Clear inter-rail gap (y0_inclusive, y1_exclusive) between two rails."""
    return (top_rail[1], bot_rail[0])


# --------------------------------------------------------------------------- #
# Layer builders — each mutates `img` in place
# --------------------------------------------------------------------------- #

def _ballast_texture(rng):
    """Dark reddish-brown ballast filling the whole image.

    Each pixel is the base BACKGROUND_RGB multiplied by an independent
    brightness factor in (0, 1] — variance is one-sided (only toward black)
    so the average hue stays close to the base colour and the texture looks
    like a fine global dotting rather than coloured patches. No block-level
    coarse noise is used.
    """
    h, w = IMG_H, IMG_W
    # Half-normal darkening: brightness = 1 - |N(0, σ)|. With σ=0.10, mean
    # darkening ≈ 0.08, so the median pixel stays within ~8% of the base
    # colour — the texture reads as "the base hue with occasional darker
    # specks" rather than as a generally washed-out palette.
    darkening = np.abs(rng.normal(0, 0.10, size=(h, w, 1))).astype(np.float32)
    brightness = np.clip(1.0 - darkening, 0.15, 1.0)
    # Tiny per-pixel chromatic jitter so the texture isn't monochrome.
    tint = rng.normal(0, 3, size=(h, w, 3)).astype(np.float32)
    img = BACKGROUND_RGB * brightness + tint
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_red_background_noise(rng, img):
    """Scatter a small proportion of red pixels across the whole image.

    Isolated pixels or 1-2 px clusters that prevent the model from treating
    "all red = rail." Concentrated in ballast zones (anywhere, but visually
    they only stand out off-rail).
    """
    h, w = img.shape[:2]
    n_speckles = int(rng.integers(80, 180))
    for _ in range(n_speckles):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        size = int(rng.integers(1, 3))
        y0 = max(0, cy)
        y1 = min(h, cy + size)
        x0 = max(0, cx)
        x1 = min(w, cx + size)
        n = (y1 - y0) * (x1 - x0)
        if n > 0:
            col = RED_NOISE_RGB + rng.normal(0, 22, size=(n, 3))
            img[y0:y1, x0:x1] = np.clip(
                col.reshape(y1 - y0, x1 - x0, 3), 0, 255
            ).astype(np.uint8)


def _draw_sleepers(rng, img, rails):
    """Black transverse sleepers across each track set's band.

    Spacing and thickness have noise. Some sleepers blend more weakly than
    others to mimic patchy LiDAR returns.
    """
    h, w = img.shape[:2]
    for top_rail, bot_rail in [(rails[0], rails[1]), (rails[2], rails[3])]:
        band_top = max(0, top_rail[0] - 1)
        band_bot = min(h, bot_rail[1] + 1)
        if band_bot <= band_top:
            continue

        spacing = int(rng.integers(14, 22))
        x = int(rng.integers(0, spacing))
        while x < w:
            sleeper_w = 1 if rng.random() < 0.7 else 2
            x0 = max(0, x - sleeper_w // 2)
            x1 = min(w, x + sleeper_w // 2 + 1)
            opacity = float(rng.uniform(0.20, 0.55))
            if rng.random() < 0.30:
                opacity *= float(rng.uniform(0.15, 0.45))
            region = img[band_top:band_bot, x0:x1].astype(np.float32)
            new_px = SLEEPER_RGB + rng.normal(0, 6, size=region.shape)
            blended = opacity * new_px + (1.0 - opacity) * region
            img[band_top:band_bot, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
            x += spacing + int(rng.integers(-4, 5))


def _draw_rails(rng, img, rails):
    """Draw the four horizontal rails.

    Real LiDAR rails are far from clean horizontal lines: returns vary in
    intensity along the rail, drop out entirely in patches (sparse points),
    and shift slightly in hue. We model that with:
      * per-pixel chromatic noise (saturation/hue jitter)
      * a slow horizontal "alpha" mask that fades the rail into the ballast
      * occasional hard holes (LiDAR returned no point in that column)
      * a per-rail overall intensity multiplier so not every rail is equally bright
    """
    h, w = img.shape[:2]
    for y0, y1 in rails:
        if y1 <= y0:
            continue
        rail_h = y1 - y0

        # Overall intensity for this rail — some rails fainter than others
        rail_strength = float(rng.uniform(0.65, 1.0))

        # Per-pixel chromatic noise — bigger than before so hue varies
        noise = rng.normal(0, 22, size=(rail_h, w, 3))

        # Slow horizontal intensity variation — broad bright/dark segments
        col_var = rng.normal(0, 35, size=(1, w, 1))
        # Smooth to avoid 1-px striping
        k = 5
        kernel = np.ones((1, k, 1)) / k
        col_var = np.apply_along_axis(
            lambda v: np.convolve(v, kernel.ravel(), mode="same"), 1, col_var
        )

        # Per-column blend mask (alpha) — fraction of rail vs background
        # Mostly mid-high but with a long-tailed dropout
        col_alpha = rng.beta(3.0, 1.2, size=w)
        # Smooth the alpha so dropouts come in patches not single columns
        col_alpha = np.convolve(col_alpha, np.ones(3) / 3, mode="same")
        # Hard holes: a small fraction of columns get near-zero alpha
        hole_mask = rng.random(w) < 0.04
        col_alpha[hole_mask] *= rng.uniform(0, 0.2, size=int(hole_mask.sum()))
        col_alpha = col_alpha[None, :, None]  # (1, w, 1)

        rail_colour = np.clip(
            (RAIL_RGB + noise + col_var) * rail_strength, 0, 255
        )
        bg = img[y0:y1].astype(np.float32)
        blended = col_alpha * rail_colour + (1.0 - col_alpha) * bg
        img[y0:y1] = np.clip(blended, 0, 255).astype(np.uint8)


def _add_rail_motifs(rng, img, rails, p_motif=0.05):
    """Small markings on top of the rails — brighter/darker pixel clusters."""
    h, w = img.shape[:2]
    for y0, y1 in rails:
        if y1 <= y0:
            continue
        n_motifs = int(rng.integers(8, 22))
        for _ in range(n_motifs):
            mx = int(rng.integers(0, max(1, w - 4)))
            mw = int(rng.integers(1, 4))
            x1 = min(w, mx + mw)
            if rng.random() < 0.5:
                tone = RAIL_RGB + np.array([40, 35, 20], dtype=np.float32)
            else:
                tone = RAIL_RGB + np.array([-70, -55, -10], dtype=np.float32)
            tone = np.clip(tone, 0, 255)
            n = (y1 - y0) * (x1 - mx)
            if n > 0:
                noise = rng.normal(0, 10, size=(y1 - y0, x1 - mx, 3))
                img[y0:y1, mx:x1] = np.clip(tone + noise, 0, 255).astype(np.uint8)


def _draw_blob(rng, img, cx, cy, radius, colour):
    """Roughly-circular blob of given colour (with chromatic noise)."""
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    n = int(mask.sum())
    if n > 0:
        col = colour + rng.normal(0, 20, size=(n, 3))
        img[mask] = np.clip(col, 0, 255).astype(np.uint8)


def _draw_diagonal_rail(rng, img, x0, y0, x1, y1, thickness=2):
    """Draw a diagonal rail line between two points by stepping along x.

    Uses the same LiDAR-style sparsity as the main rails: per-column alpha
    blending with occasional hole columns.
    """
    h, w = img.shape[:2]
    if x1 == x0:
        return
    dx = x1 - x0
    dy = y1 - y0
    n_steps = abs(dx) + 1
    sgn = 1 if dx > 0 else -1
    strength = float(rng.uniform(0.65, 1.0))
    for i in range(n_steps):
        x = x0 + sgn * i
        if not (0 <= x < w):
            continue
        # Column alpha — high mostly, with occasional dropouts
        alpha = float(rng.beta(3.0, 1.0))
        if rng.random() < 0.05:
            alpha *= float(rng.uniform(0, 0.2))
        t = i / max(1, n_steps - 1)
        y_center = int(round(y0 + t * dy))
        y_lo = max(0, y_center - thickness // 2)
        y_hi = min(h, y_lo + thickness)
        if y_hi <= y_lo:
            continue
        for yy in range(y_lo, y_hi):
            rail_col = np.clip((RAIL_RGB + rng.normal(0, 22, size=3)) * strength, 0, 255)
            bg = img[yy, x].astype(np.float32)
            img[yy, x] = np.clip(alpha * rail_col + (1.0 - alpha) * bg, 0, 255).astype(np.uint8)


def _draw_switch(rng, img, rails):
    """Optionally draw switch-point variants A and/or B on top of the ballast.

    Variant A: a diagonal rail crossing between the two track sets.
    Variant B: a rail branching off and leaving the image edge.
    """
    h, w = img.shape[:2]
    variants = []
    if rng.random() < 0.7:
        variants.append("A")
    if rng.random() < 0.5:
        variants.append("B")
    if not variants:
        variants.append("A")

    rail_t = int(rng.choice([2, 3]))

    for variant in variants:
        if variant == "A":
            # Diagonal between rail of upper track and rail of lower track
            src_idx = int(rng.choice([1, 2]))           # inner rail of upper or lower
            dst_idx = 2 if src_idx == 1 else 1
            y_src = (rails[src_idx][0] + rails[src_idx][1]) // 2
            y_dst = (rails[dst_idx][0] + rails[dst_idx][1]) // 2

            run_length = int(rng.integers(220, 420))
            x_start = int(rng.integers(0, max(1, w - run_length - 20)))
            x_end = min(w - 1, x_start + run_length)
            if rng.random() < 0.5:
                x_start, x_end = x_end, x_start
            _draw_diagonal_rail(rng, img, x_start, y_src, x_end, y_dst, thickness=rail_t)

        else:  # Variant B
            src_idx = int(rng.integers(0, 4))
            y_src = (rails[src_idx][0] + rails[src_idx][1]) // 2
            x_start = int(rng.integers(60, max(61, w - 60)))

            edge = rng.choice(["left", "right", "top", "bottom"])
            if edge == "left":
                x_end = 0
                y_end = max(0, min(h - 1, y_src + int(rng.integers(-20, 21))))
            elif edge == "right":
                x_end = w - 1
                y_end = max(0, min(h - 1, y_src + int(rng.integers(-20, 21))))
            elif edge == "top":
                run = int(rng.integers(80, 260))
                direction = int(rng.choice([-1, 1]))
                x_end = max(0, min(w - 1, x_start + direction * run))
                y_end = 0
            else:  # bottom
                run = int(rng.integers(80, 260))
                direction = int(rng.choice([-1, 1]))
                x_end = max(0, min(w - 1, x_start + direction * run))
                y_end = h - 1
            _draw_diagonal_rail(rng, img, x_start, y_src, x_end, y_end, thickness=rail_t)


def _draw_red_bar(rng, img, x0_world, x1_world, y0, y1):
    """Draw a partially-transparent rail-coloured horizontal bar.

    World coords (x0_world, x1_world) may extend off either edge; we clip to
    the image. Used by the clip-like hard-negative motifs.
    """
    h, w = img.shape[:2]
    sx0 = max(0, x0_world)
    sx1 = min(w, x1_world)
    if sx1 <= sx0 or y1 <= y0:
        return
    bh = y1 - y0
    bw = sx1 - sx0
    strength = float(rng.uniform(0.65, 0.95))
    noise = rng.normal(0, 22, size=(bh, bw, 3))
    colour = np.clip((RAIL_RGB + noise) * strength, 0, 255)
    alpha = rng.beta(3.0, 1.0, size=(bh, bw, 1))
    hole = rng.random((bh, bw)) < 0.07
    alpha[hole] *= rng.uniform(0, 0.2, size=(int(hole.sum()), 1))
    bg = img[y0:y1, sx0:sx1].astype(np.float32)
    img[y0:y1, sx0:sx1] = np.clip(
        alpha * colour + (1.0 - alpha) * bg, 0, 255
    ).astype(np.uint8)


def _draw_green_tail(rng, img, x0_world, x1_world, gap_y0, gap_y1, bar_y0, bar_y1):
    """Draw a dense green grouping at one end of a motif (Motif A's tail).

    The green spans the bar height but also bleeds 1 px above/below, like a
    soft elevated blob rather than a thin line.
    """
    h, w = img.shape[:2]
    sx0 = max(0, x0_world)
    sx1 = min(w, x1_world)
    if sx1 <= sx0:
        return
    # Slightly wider vertical span than the red bar
    y0 = max(gap_y0, bar_y0 - 1)
    y1 = min(gap_y1, bar_y1 + 1)
    if y1 <= y0:
        return
    density = float(rng.uniform(0.55, 0.85))
    for xi in range(sx0, sx1):
        for yi in range(y0, y1):
            if rng.random() < density:
                col = GREEN_RGB + rng.normal(0, 18, size=3)
                img[yi, xi] = np.clip(col, 0, 255).astype(np.uint8)


def _draw_motif_red_green_tail(rng, img, gap_y0, gap_y1):
    """Motif A: a clip-width red bar with a dense green grouping at one tail."""
    h, w = img.shape[:2]
    motif_w = int(rng.integers(60, 90))
    motif_h = int(rng.choice([3, 4]))
    # Position may extend off either edge
    x_start = int(rng.integers(-int(motif_w * 0.6), w - int(motif_w * 0.4)))
    x_end = x_start + motif_w
    gap_mid = (gap_y0 + gap_y1) // 2
    bar_y0 = max(gap_y0, gap_mid - motif_h // 2 + int(rng.integers(-1, 2)))
    bar_y1 = min(gap_y1, bar_y0 + motif_h)
    if bar_y1 <= bar_y0:
        return

    # Red body
    _draw_red_bar(rng, img, x_start, x_end, bar_y0, bar_y1)

    # Green tail at one randomly-chosen end (15–25% of motif width)
    tail_len = int(motif_w * rng.uniform(0.15, 0.25))
    if rng.random() < 0.5:
        tail_x0, tail_x1 = x_start, x_start + tail_len
    else:
        tail_x0, tail_x1 = x_end - tail_len, x_end
    _draw_green_tail(rng, img, tail_x0, tail_x1, gap_y0, gap_y1, bar_y0, bar_y1)


def _draw_motif_wide_with_gap(rng, img, gap_y0, gap_y1):
    """Motif B: ~100 px wide × ~10 px tall red capsule with an inner slot.

    The middle has an 8 px tall background-coloured slot leaving only 1 px of
    red on top and 1 px on bottom. The slot does NOT extend to the horizontal
    ends — the last 10–15 px on each side remain full-height solid red, so
    the motif looks like a thick red bar with a narrow horizontal slit cut
    through its centre (except near the ends).
    """
    h_img, w_img = img.shape[:2]
    motif_w = int(rng.integers(90, 115))    # ~100 px
    motif_h = int(rng.integers(9, 12))      # ~10 px ("in and around")
    end_cap = int(rng.integers(10, 16))     # solid red at each end
    slot_h = motif_h - 2                    # 1 px of red top + 1 px of red bottom

    # Vertical placement — well centred in the inter-rail clear gap
    gap_mid = (gap_y0 + gap_y1) // 2
    bar_y0 = max(gap_y0, gap_mid - motif_h // 2 + int(rng.integers(-1, 2)))
    bar_y1 = bar_y0 + motif_h
    if bar_y1 > gap_y1:
        return  # inter-rail gap too narrow for a 10 px motif

    # Horizontal placement — may extend off either edge
    x_start = int(rng.integers(-int(motif_w * 0.4), w_img - int(motif_w * 0.5)))
    x_end = x_start + motif_w

    # Build the shape from 4 solid pieces so the slot is exactly ballast colour
    # (no need to "carve out" pixels):
    #   left cap (full height) | top strip (1 px) | bottom strip (1 px) | right cap
    _draw_red_bar(rng, img, x_start, x_start + end_cap, bar_y0, bar_y1)
    _draw_red_bar(rng, img, x_end - end_cap, x_end, bar_y0, bar_y1)
    if motif_w - 2 * end_cap > 0:
        # Top edge of slot (1 px of red between caps)
        _draw_red_bar(rng, img, x_start + end_cap, x_end - end_cap, bar_y0, bar_y0 + 1)
        # Bottom edge of slot (1 px of red between caps)
        _draw_red_bar(rng, img, x_start + end_cap, x_end - end_cap, bar_y1 - 1, bar_y1)


def _add_inter_rail_features(rng, img, rails):
    """Hard negatives inside the inter-rail zones.

    Always present:
      * A green dot inside the gap (a few pixels in diameter).
      * 2–6 small scattered green/red dots per track set.

    Rare (5% of images):
      * Two clip-like motifs placed somewhere in the inter-rail zones. Each
        motif is either:
          A — a red bar with a dense green grouping at one end (one tail), or
          B — a much wider red bar with a centre gap filled with background.
        The pair always has the same type, sits in the same inter-rail zone,
        and is separated horizontally; one of the two may extend off-screen.
    """
    h, w = img.shape[:2]

    for top_rail, bot_rail in [(rails[0], rails[1]), (rails[2], rails[3])]:
        gap_y0, gap_y1 = _gap(top_rail, bot_rail)
        if gap_y1 - gap_y0 < 6:
            continue

        # Always-present green dot
        green_cx = int(rng.integers(10, w - 10))
        green_cy = int(rng.integers(gap_y0 + 2, gap_y1 - 2))
        green_r = int(rng.integers(2, 4))
        _draw_blob(rng, img, green_cx, green_cy, green_r, GREEN_RGB)

        # Always-present scattered dots (2–6)
        n_dots = int(rng.integers(2, 7))
        for _ in range(n_dots):
            dx = int(rng.integers(0, w))
            dy = int(rng.integers(gap_y0 + 1, gap_y1 - 1))
            dr = int(rng.integers(1, 3))
            colour = GREEN_RGB if rng.random() < 0.5 else RED_NOISE_RGB
            _draw_blob(rng, img, dx, dy, dr, colour)

    if rng.random() < p_motif:
        # Choose which inter-rail zone
        top_rail, bot_rail = (
            (rails[0], rails[1]) if rng.random() < 0.5 else (rails[2], rails[3])
        )
        gap_y0, gap_y1 = _gap(top_rail, bot_rail)
        if gap_y1 - gap_y0 >= 6:
            motif_type = "A" if rng.random() < 0.5 else "B"
            drawer = (
                _draw_motif_red_green_tail if motif_type == "A"
                else _draw_motif_wide_with_gap
            )
            # Two instances. Each picks its own (possibly off-screen) position
            # inside the drawer, so spacing emerges naturally.
            drawer(rng, img, gap_y0, gap_y1)
            drawer(rng, img, gap_y0, gap_y1)


def _add_crocodile_clip(rng, img, rails, track):
    """Place a crocodile clip in the chosen track's inter-rail zone.

    Returns (x0, y0, x1, y1) or None if the gap is too narrow.
    """
    h, w = img.shape[:2]
    if track == "upper":
        top_rail, bot_rail = rails[0], rails[1]
    else:
        top_rail, bot_rail = rails[2], rails[3]
    gap_y0, gap_y1 = _gap(top_rail, bot_rail)
    gap_h = gap_y1 - gap_y0
    if gap_h < 6:
        return None

    clip_w = int(rng.integers(65, 76))   # 65-75 px
    clip_h = int(rng.integers(4, 6))     # 4-5 px

    # Horizontally: roughly centred, with small jitter
    cx = w // 2 + int(rng.integers(-60, 61))
    x0 = max(0, cx - clip_w // 2)
    x1 = min(w, x0 + clip_w)
    if x1 - x0 < clip_w:
        x0 = max(0, x1 - clip_w)

    # Vertically: well centred between the two rails
    gap_mid = (gap_y0 + gap_y1) // 2
    cy = gap_mid + int(rng.integers(-1, 2))
    y0 = max(gap_y0, cy - clip_h // 2)
    y1 = min(gap_y1, y0 + clip_h)
    if y1 - y0 < clip_h:
        y0 = max(gap_y0, y1 - clip_h)
    bh = y1 - y0
    bw = x1 - x0
    if bh <= 0 or bw <= 0:
        return None

    # --- Red body (with LiDAR-style intensity variation and small holes) ---
    body_strength = float(rng.uniform(0.7, 1.0))
    body_noise = rng.normal(0, 22, size=(bh, bw, 3))
    # Slow horizontal variation along the clip length
    body_col_var = rng.normal(0, 25, size=(1, bw, 1))
    body_col_var = np.apply_along_axis(
        lambda v: np.convolve(v, np.ones(3) / 3, mode="same"), 1, body_col_var
    )
    body_colour = np.clip(
        (RAIL_RGB + body_noise + body_col_var) * body_strength, 0, 255
    )
    # Per-pixel alpha so the clip blends partially with the ballast
    body_alpha = rng.beta(3.0, 1.0, size=(bh, bw, 1))
    # Hard LiDAR holes
    hole_mask = rng.random((bh, bw)) < 0.06
    body_alpha[hole_mask] *= rng.uniform(0, 0.15, size=(int(hole_mask.sum()), 1))
    bg = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(
        body_alpha * body_colour + (1.0 - body_alpha) * bg, 0, 255
    ).astype(np.uint8)

    # --- Green dots concentrated toward the centre ---
    green_fraction = float(rng.uniform(0.35, 0.75))
    inner_x0 = x0 + int(bw * 0.20)
    inner_x1 = x1 - int(bw * 0.20)
    if inner_x1 > inner_x0:
        cx_inner = (inner_x0 + inner_x1) / 2.0
        half = max(1.0, (inner_x1 - inner_x0) / 2.0)
        for xi in range(inner_x0, inner_x1):
            dist_norm = abs(xi - cx_inner) / half
            p = green_fraction * (1.0 - 0.5 * dist_norm)
            for yi in range(y0, y1):
                if rng.random() < p:
                    col = GREEN_RGB + rng.normal(0, 18, size=3)
                    img[yi, xi] = np.clip(col, 0, 255).astype(np.uint8)

    # --- Bounding-box jitter (label noise) ---
    # Real annotations are not pixel-perfect — each edge wanders by a few
    # pixels horizontally and ≤1 px vertically. Box always remains valid.
    jx0 = int(rng.integers(-3, 4))
    jx1 = int(rng.integers(-3, 4))
    jy0 = int(rng.integers(-1, 2))
    jy1 = int(rng.integers(-1, 2))
    bx0 = max(0, min(IMG_W - 2, x0 + jx0))
    bx1 = max(bx0 + 2, min(IMG_W, x1 + jx1))
    by0 = max(0, min(IMG_H - 2, y0 + jy0))
    by1 = max(by0 + 2, min(IMG_H, y1 + jy1))
    return (bx0, by0, bx1, by1)


# --------------------------------------------------------------------------- #
# Compose one image
# --------------------------------------------------------------------------- #

def _make_image(rng, config):
    """Generate one synthetic image according to a config dict.

    config must contain: p_clip, p_switch, clip_tracks.

    Layer order (bottom -> top):
      1. Ballast texture
      2. Red background noise (speckle pixels)
      3. Switch point extra rails (optional, drawn under sleepers)
      4. Sleepers
      5. Main rails (drawn on top of sleepers)
      6. Rail motifs
      7. Inter-rail fixed features + scattered dots (hard negatives)
      8. Optional crocodile clip

    Returns (uint8 HxWx3 RGB array, bbox tuple or None).
    """
    rails, _rail_t = _pick_geometry(rng)

    # 1. Ballast
    img = _ballast_texture(rng)
    # 2. Red speckle on ballast
    _add_red_background_noise(rng, img)
    # 3. Switch point (drawn before sleepers so sleepers overlap it visually)
    if rng.random() < config["p_switch"]:
        _draw_switch(rng, img, rails)
    # 4. Sleepers
    _draw_sleepers(rng, img, rails)
    # 5. Main rails
    _draw_rails(rng, img, rails)
    # 6. Rail motifs on top of rails
    _add_rail_motifs(rng, img, rails, p_motif=config.get("p_motif", 0.05))
    # 7. Inter-rail hard negatives
    _add_inter_rail_features(rng, img, rails)

    # 8. Optional crocodile clip
    bbox = None
    if rng.random() < config["p_clip"]:
        track = str(rng.choice(config["clip_tracks"]))
        bbox = _add_crocodile_clip(rng, img, rails, track)

    return img, bbox


# --------------------------------------------------------------------------- #
# Public loader — writes a YOLO-style dataset to disk
# --------------------------------------------------------------------------- #

def load_synthetic_rails(output_dir, config="train_two_tracks", n_samples=700, seed=42):
    """Generate (or reuse) a synthetic rails dataset on disk in YOLO format.

    Layout produced under `output_dir/<config>/`:
        images/rail_00000.png ...   (570x100 PNGs)
        labels/rail_00000.txt ...   ("0 cx cy w h" normalised, empty = no clip)

    The cache check is per-config so regenerating one configuration does not
    invalidate the others.

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

    # Cache: re-use existing files if the requested count is already present.
    if labels_dst.exists() and len(list(labels_dst.glob("*.txt"))) >= n_samples:
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        img, bbox = _make_image(rng, cfg)
        name = f"rail_{i:05d}"
        Image.fromarray(img).save(images_dst / f"{name}.png")

        if bbox is None:
            (labels_dst / f"{name}.txt").write_text("")
        else:
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2 / IMG_W
            cy = (y0 + y1) / 2 / IMG_H
            bw = (x1 - x0) / IMG_W
            bh = (y1 - y0) / IMG_H
            (labels_dst / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}
