"""Generate sample synthetic images for visual inspection.

Produces:
  - Clip-in-upper-track and clip-in-lower-track examples (bbox drawn in red)
  - Empty-rails examples
  - Switch-point examples
  - Motif examples (always 1 image of motif A and 1 of motif B — the rare
    clip-like hard negatives that normally appear in only 5% of images)

Outputs go to `preview/`. Run with: `python preview_synthetic.py`.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import data_generation.dataset_synthetic as dataset_synthetic
from data_generation.dataset_synthetic import _make_image


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


def _save(pil, name, out_dir, bbox=None):
    if bbox is not None:
        draw = ImageDraw.Draw(pil)
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=1)
    out_path = out_dir / f"{name}.png"
    pil.save(out_path)
    print(f"Wrote {out_path}  ({pil.size[0]}x{pil.size[1]})")


def main():
    out_dir = Path("preview")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=123)

    for name, cfg in SPECS:
        img, bbox = _make_image(rng, cfg)
        _save(Image.fromarray(img), name, out_dir, bbox)

    # Force-render one image of each motif type. The motifs are only present in
    # ~5% of images normally — to make sure both A and B appear in the preview,
    # we monkey-patch the random chooser inside _add_inter_rail_features.
    print("\n--- forced motif previews ---")
    # The motif code path is gated by `rng.random() < 0.05`; we override that
    # by patching numpy.random.Generator.random for a single call.
    rng2 = np.random.default_rng(seed=999)
    for name, motif_type_label in [("motifs_A_only", "A"), ("motifs_B_only", "B")]:
        # Force the gate (probability 0.05) to always pass and the type choice
        # (probability 0.5) to pick the desired motif.
        # We do this by stubbing the gate inline.
        cfg = {"p_clip": 0.0, "p_switch": 0.0, "clip_tracks": ("upper", "lower")}
        # Make a fresh image with the gate forced ON
        img = _make_image_with_forced_motif(rng2, cfg, force_type=motif_type_label)
        _save(Image.fromarray(img), name, out_dir)


def _make_image_with_forced_motif(rng, cfg, force_type):
    """Run the normal image pipeline but force the rare-motif gate to fire."""
    import numpy as np
    from data_generation.dataset_synthetic import (
        _pick_geometry, _ballast_texture, _add_red_background_noise,
        _draw_switch, _draw_sleepers, _draw_rails, _add_rail_motifs,
        _draw_blob, _gap, GREEN_RGB, RED_NOISE_RGB,
        _draw_motif_red_green_tail, _draw_motif_wide_with_gap,
    )
    rails, _ = _pick_geometry(rng)
    img = _ballast_texture(rng)
    _add_red_background_noise(rng, img)
    if rng.random() < cfg["p_switch"]:
        _draw_switch(rng, img, rails)
    _draw_sleepers(rng, img, rails)
    _draw_rails(rng, img, rails)
    _add_rail_motifs(rng, img, rails)
    # Inline the always-present features (green dot + scattered dots)
    h, w = img.shape[:2]
    for top_rail, bot_rail in [(rails[0], rails[1]), (rails[2], rails[3])]:
        gap_y0, gap_y1 = _gap(top_rail, bot_rail)
        if gap_y1 - gap_y0 < 6:
            continue
        green_cx = int(rng.integers(10, w - 10))
        green_cy = int(rng.integers(gap_y0 + 2, gap_y1 - 2))
        _draw_blob(rng, img, green_cx, green_cy, int(rng.integers(2, 4)), GREEN_RGB)
        for _ in range(int(rng.integers(2, 7))):
            dx = int(rng.integers(0, w))
            dy = int(rng.integers(gap_y0 + 1, gap_y1 - 1))
            dr = int(rng.integers(1, 3))
            colour = GREEN_RGB if rng.random() < 0.5 else RED_NOISE_RGB
            _draw_blob(rng, img, dx, dy, dr, colour)
    # Forced motifs (skipping the 5% gate)
    top_rail, bot_rail = (
        (rails[0], rails[1]) if rng.random() < 0.5 else (rails[2], rails[3])
    )
    gap_y0, gap_y1 = _gap(top_rail, bot_rail)
    drawer = _draw_motif_red_green_tail if force_type == "A" else _draw_motif_wide_with_gap
    drawer(rng, img, gap_y0, gap_y1)
    drawer(rng, img, gap_y0, gap_y1)
    return img


if __name__ == "__main__":
    main()
