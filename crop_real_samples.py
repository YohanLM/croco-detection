"""
Crop real LiDAR samples to horizontal strips centred on the railway tracks.

Steps:
  1. Extract the background brownish colour from image 3 near y=400 using
     Sliced Wasserstein Distance (SWD) — quantifies how well background
     pixels are separated from reddish rail pixels in colour space.
  2. Find rail y-positions:
       a. Build a per-row reddish pixel density profile (fraction of row
          pixels satisfying (R-G) > RG_DIFF_THRESH).
       b. Mark rows where density ≥ ROW_DENSITY_THRESH as "rail rows".
       c. Group consecutive rail rows (gap tolerance ≤ GAP_TOL) into
          components.
       d. Shape filter: discard components taller than MAX_COMPONENT_HEIGHT
          (these are irregular blobs, not thin rail lines).
       e. From remaining components, pick the tightest cluster fitting in
          CROP_HEIGHT px, preferring the one closest to the image centre.
       f. A detection is "good" if it yields ≥ 2 components after filtering
          AND the mean peak-row density across selected components is ≥
          MIN_MEAN_DENSITY.
  3. Two-pass cropping:
       - Pass 1: run detection on every image; collect good detections and
         their crop-centre y values.
       - Compute global_median_centre from the good detections.
       - Pass 2: images with a good detection are cropped around their own
         detected centre; images with a bad detection (blob noise, too few
         components, low density) fall back to global_median_centre.
  4. Crop height is always CROP_HEIGHT px. Save to real_samples_cropped/.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
SAMPLES_DIR = Path("real_samples")
OUTPUT_DIR  = Path("real_samples_cropped")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
CROP_HEIGHT           = 100   # final crop height in pixels
RG_DIFF_THRESH        = 80    # (R - G) > this → pixel is reddish
ROW_DENSITY_THRESH    = 0.50  # fraction of row pixels that must be reddish
GAP_TOL               = 3     # consecutive non-rail rows tolerated within a component
MAX_COMPONENT_HEIGHT  = 16    # (A) components taller than this are blobs, not rails
MIN_MEAN_DENSITY      = 0.65  # (B) mean peak-density across selected components
MIN_COMPONENTS_GOOD   = 2     # (B) minimum selected components for a "good" detection
SWD_N_PROJ            = 128   # projections for Sliced Wasserstein Distance
SWD_REF_Y             = 400   # reference row for background extraction (image 3)
BG_WINDOW             = 30    # ± rows around SWD_REF_Y


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BACKGROUND COLOUR VIA SLICED WASSERSTEIN DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def _sliced_wasserstein_distance(
    a: np.ndarray, b: np.ndarray, n_proj: int = SWD_N_PROJ, seed: int = 0
) -> float:
    """Sliced Wasserstein Distance between two (N, 3) RGB point clouds."""
    rng  = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    total = 0.0
    for d in dirs:
        pa = np.sort(a @ d)
        pb = np.sort(b @ d)
        if len(pa) != len(pb):
            t  = np.linspace(0, 1, max(len(pa), len(pb)))
            pa = np.interp(t, np.linspace(0, 1, len(pa)), pa)
            pb = np.interp(t, np.linspace(0, 1, len(pb)), pb)
        total += np.mean(np.abs(pa - pb))
    return total / n_proj


def extract_background_colour(img_path: Path, ref_y: int = SWD_REF_Y,
                               window: int = BG_WINDOW) -> np.ndarray:
    """
    Median RGB of background pixels near ref_y, plus SWD vs rail pixels.

    'Background' = pixels where (R - G) ≤ RG_DIFF_THRESH (not reddish rails).
    'Rail'       = pixels where (R - G) >  RG_DIFF_THRESH.
    """
    arr   = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    h     = arr.shape[0]
    y0    = max(0, ref_y - window)
    y1    = min(h, ref_y + window + 1)
    strip = arr[y0:y1].reshape(-1, 3)

    R, G  = strip[:, 0], strip[:, 1]
    is_rail = (R - G) > RG_DIFF_THRESH

    bg   = strip[~is_rail]
    rail = strip[ is_rail]
    med_bg = np.median(bg, axis=0)

    if len(rail) == 0:
        print(f"  Background median RGB: {med_bg.astype(int)}  (no rail pixels in strip)")
        return med_bg

    swd = _sliced_wasserstein_distance(bg, rail)
    print(f"  Background median RGB : {med_bg.astype(int)}")
    print(f"  Rail median RGB       : {np.median(rail, axis=0).astype(int)}")
    print(f"  Sliced Wasserstein D  : {swd:.2f}  "
          f"({'well' if swd > 30 else 'poorly'} separated in colour space)")
    return med_bg


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FIND RAIL POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _row_densities(arr: np.ndarray) -> np.ndarray:
    """Per-row fraction of pixels with (R - G) > RG_DIFF_THRESH."""
    R = arr[:, :, 0].astype(np.float32)
    G = arr[:, :, 1].astype(np.float32)
    mask = (R - G) > RG_DIFF_THRESH
    return mask.sum(axis=1) / arr.shape[1]


def _group_rail_rows(is_rail_row: np.ndarray, gap_tol: int = GAP_TOL) -> list[list[int]]:
    """
    Cluster consecutive True rows in is_rail_row into groups.
    Up to gap_tol consecutive False rows inside a group are tolerated.
    Returns a list of (sorted) row-index lists.
    """
    n = len(is_rail_row)
    groups: list[list[int]] = []
    current: list[int] = []
    gap = 0

    for y in range(n):
        if is_rail_row[y]:
            current.append(y)
            gap = 0
        else:
            if current:
                gap += 1
                if gap > gap_tol:
                    groups.append(current)
                    current = []
                    gap = 0
    if current:
        groups.append(current)
    return groups


def _best_cluster(centers: list[float], max_span: float,
                  img_mid: float) -> list[int]:
    """
    Return indices into `centers` of the sub-list that:
      1. Fits within a vertical span of max_span px.
      2. Has the most members.
      3. (Tie-break) is closest to img_mid.
    `centers` must already be sorted ascending.
    """
    best_idx: list[int] = []
    best_count = 0
    best_dist  = float("inf")

    n = len(centers)
    lo = 0
    for hi in range(n):
        while centers[hi] - centers[lo] > max_span:
            lo += 1
        count = hi - lo + 1
        dist  = abs((centers[lo] + centers[hi]) / 2 - img_mid)
        if count > best_count or (count == best_count and dist < best_dist):
            best_count = count
            best_dist  = dist
            best_idx   = list(range(lo, hi + 1))

    return best_idx


def find_rail_band(img_path: Path) -> tuple[int, int, list[dict]]:
    """
    Return (top_y, bottom_y, selected_component_list).

    Applies the shape filter (A): components taller than MAX_COMPONENT_HEIGHT
    are discarded before cluster selection.
    Raises ValueError if no components survive filtering.
    """
    arr    = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    height, width = arr.shape[:2]

    densities   = _row_densities(arr)
    is_rail_row = densities >= ROW_DENSITY_THRESH
    groups      = _group_rail_rows(is_rail_row)

    if not groups:
        raise ValueError(
            f"No rail rows found in {img_path.name} "
            f"(density≥{ROW_DENSITY_THRESH}, R-G>{RG_DIFF_THRESH})"
        )

    # Build component records
    all_components = []
    for g in groups:
        rows = np.array(g)
        row_dens = densities[rows]
        all_components.append({
            "y_min":       int(rows.min()),
            "y_max":       int(rows.max()),
            "y_center":    float(rows.mean()),
            "height_px":   int(rows.max() - rows.min() + 1),
            "peak_density": float(row_dens.max()),
        })

    # (A) Shape filter: remove blob-like thick components
    components = [c for c in all_components if c["height_px"] <= MAX_COMPONENT_HEIGHT]
    n_removed  = len(all_components) - len(components)

    if not components:
        raise ValueError(
            f"All components filtered out by shape filter in {img_path.name} "
            f"(MAX_COMPONENT_HEIGHT={MAX_COMPONENT_HEIGHT})"
        )

    components.sort(key=lambda c: c["y_center"])
    centers = [c["y_center"] for c in components]

    # Select tightest cluster within CROP_HEIGHT, preferring image centre
    img_mid  = height / 2
    sel_idx  = _best_cluster(centers, float(CROP_HEIGHT), img_mid)
    selected = [components[i] for i in sel_idx]

    top_y    = selected[0]["y_center"]
    bottom_y = selected[-1]["y_center"]
    span     = bottom_y - top_y

    blob_note = f"  [{n_removed} blob(s) removed by shape filter]" if n_removed else ""
    print(f"  {len(all_components)} raw → {len(components)} after filter "
          f"→ selected {len(selected)}  "
          f"(top y≈{top_y:.0f}, bottom y≈{bottom_y:.0f}, span={span:.0f} px){blob_note}")
    for i, c in enumerate(selected):
        print(f"    [{i}] y_center={c['y_center']:6.1f}  "
              f"y_range=[{c['y_min']},{c['y_max']}]  "
              f"height={c['height_px']}px  "
              f"peak_density={c['peak_density']:.2f}")

    return int(round(top_y)), int(round(bottom_y)), selected


def is_good_detection(selected: list[dict], span: int) -> bool:
    """
    (B) Return True when the detection is reliable enough to use directly.

    Criteria:
      - At least MIN_COMPONENTS_GOOD components survived the shape filter.
      - Mean peak-row density across selected components ≥ MIN_MEAN_DENSITY
        (low density → the "rails" are sparse noise, not real rail lines).
      - Span fits within the crop (span < CROP_HEIGHT).
    """
    if len(selected) < MIN_COMPONENTS_GOOD:
        return False
    if span >= CROP_HEIGHT:
        return False
    mean_density = float(np.mean([c["peak_density"] for c in selected]))
    return mean_density >= MIN_MEAN_DENSITY


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CROP
# ══════════════════════════════════════════════════════════════════════════════

def crop_image(img_path: Path, top_y: int, bottom_y: int,
               out_path: Path, target_height: int = CROP_HEIGHT) -> None:
    """
    Crop to target_height px centred on the mid-point of [top_y, bottom_y].
    Shifts the window if it would go out of bounds.
    """
    span = bottom_y - top_y
    if span >= target_height:
        raise ValueError(
            f"{img_path.name}: rail span={span} px ≥ target_height={target_height} px — "
            "cannot add any margin around the rails."
        )

    img    = Image.open(img_path).convert("RGB")
    height = img.height

    mid     = (top_y + bottom_y) / 2
    y_start = max(0, int(round(mid - target_height / 2)))
    y_end   = y_start + target_height
    if y_end > height:
        y_end   = height
        y_start = max(0, y_end - target_height)

    cropped = img.crop((0, y_start, img.width, y_end))
    cropped.save(out_path)
    print(f"  Saved {out_path.name}  y=[{y_start},{y_end})  "
          f"({cropped.width}×{cropped.height} px)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    images = sorted(SAMPLES_DIR.glob("*.png"))
    if not images:
        print("No PNG files found in", SAMPLES_DIR)
        return

    # ── Step 1: background colour ───────────────────────────────────────────
    img3_candidates = [p for p in images if "ortho_rotated_3" in p.name]
    img3 = img3_candidates[0] if img3_candidates else images[0]

    print(f"\n{'='*60}")
    print(f"STEP 1 — Background colour  ({img3.name}, y≈{SWD_REF_Y})")
    print(f"{'='*60}")
    bg_colour = extract_background_colour(img3, ref_y=SWD_REF_Y)

    # ── Step 2 pass 1: detect rails on all images ───────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Rail detection (pass 1 / 2)")
    print(f"{'='*60}")

    # detection_cache: img_path → (top_y, bottom_y, selected, good, crop_centre)
    # or None on hard failure
    detection_cache: dict[Path, tuple | None] = {}

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        try:
            top_y, bottom_y, selected = find_rail_band(img_path)
            span  = bottom_y - top_y
            good  = is_good_detection(selected, span)
            centre = (top_y + bottom_y) / 2.0
            detection_cache[img_path] = (top_y, bottom_y, selected, good, centre)
            tag = "GOOD" if good else f"LOW-QUALITY (mean density={np.mean([c['peak_density'] for c in selected]):.2f}, n={len(selected)})"
            print(f"  → {tag}")
        except ValueError as e:
            detection_cache[img_path] = None
            print(f"  ERROR: {e}")

    # ── Step 2 pass 2: compute global median from good detections ───────────
    good_centres = [
        v[4] for v in detection_cache.values()
        if v is not None and v[3]
    ]
    if not good_centres:
        print("\nNo good detections — cannot compute fallback centre.")
        return

    global_median = float(np.median(good_centres))
    print(f"\n{'='*60}")
    print(f"Good detections: {len(good_centres)}/{len(images)}")
    print(f"Global median crop centre y = {global_median:.1f}  "
          f"(fallback for low-quality images)")
    print(f"{'='*60}")

    # ── Step 3: crop ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3 — Crop")
    print(f"{'='*60}")

    n_good, n_fallback, n_err = 0, 0, 0
    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        det   = detection_cache[img_path]
        img   = Image.open(img_path).convert("RGB")
        out_path = OUTPUT_DIR / img_path.name

        if det is not None and det[3]:
            # Good detection: crop around detected rail band
            top_y, bottom_y = det[0], det[1]
            try:
                crop_image(img_path, top_y, bottom_y, out_path)
                n_good += 1
            except ValueError as e:
                print(f"  ERROR during crop: {e}")
                n_err += 1
        else:
            # Fallback: crop around the global median centre
            reason = "hard failure" if det is None else (
                f"low quality — "
                f"n={len(det[2])}, "
                f"mean density={np.mean([c['peak_density'] for c in det[2]]):.2f}"
                if det is not None else ""
            )
            print(f"  Using global fallback centre y={global_median:.0f}  [{reason}]")
            half    = CROP_HEIGHT / 2
            y_start = max(0, int(round(global_median - half)))
            y_end   = y_start + CROP_HEIGHT
            if y_end > img.height:
                y_end   = img.height
                y_start = max(0, y_end - CROP_HEIGHT)
            cropped = img.crop((0, y_start, img.width, y_end))
            cropped.save(out_path)
            print(f"  Saved {out_path.name}  y=[{y_start},{y_end})  "
                  f"({cropped.width}×{cropped.height} px)")
            n_fallback += 1

    print(f"\n{'='*60}")
    print(f"Summary: {n_good} from detection  |  "
          f"{n_fallback} from fallback  |  {n_err} errors")
    print(f"Background colour (image 3, y≈{SWD_REF_Y}): RGB={bg_colour.astype(int)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
