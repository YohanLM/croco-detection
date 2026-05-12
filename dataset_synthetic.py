"""Synthetic rails dataset.

Procedurally generates images of two parallel "rail" lines on a noisy gray
background. With probability `p_object`, a dark rectangle ("obstacle") is
placed between the rails — that rectangle is the bounding box YOLO must
learn to detect.

Internally the image generation uses PyTorch tensors (ported from
`synth_dataset.ipynb`). The public loader still writes PNGs + YOLO `.txt`
label files to disk so it matches the project's dataset interface:
    f(output_dir) -> {"images_dir": Path, "labels_dir": Path, "classes": list[str]}
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Core image-generation primitives (ported from synth_dataset.ipynb)
# --------------------------------------------------------------------------- #

def _make_background(img_size):
    """Build a noisy light-gray RGB background as a float tensor in [0, 1].

    Shape is (H, W, 3) — channels-last — matching the notebook's convention.
    A constant 0.9 plus small Gaussian noise gives a slightly speckled gray
    that mimics ballast/concrete texture without any real photo data.
    """
    return 0.9 * torch.ones((img_size, img_size, 3)) + 0.1 * torch.randn((img_size, img_size, 3))


def _make_rails(tensor):
    """Draw two parallel black "rails" on `tensor` and return rail geometry.

    The rails are defined by:
      * a tilt angle (small deviation from horizontal),
      * an anchor point (y0, x0) on the left edge of the image,
      * a fixed perpendicular `distance` between the two parallel lines.

    Returns the modified tensor and `(angle_rad, x0, y0)` so the caller can
    place an obstacle aligned with the same rail frame.
    """
    # Random small tilt in degrees, then a noisy rail spacing.
    # Spacing is ~20% of image height so it scales proportionally at any resolution.
    angle_deg = torch.randint(low=-15, high=15, size=(1,)).item()

    h, w, _ = tensor.shape
    # Pixel coordinate grids: y is row index, x is column index.
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Rail spacing: mean=20% of height, std=1.2% — same proportions as the
    # original notebook's mean=50 std=3 on a 256 px image.
    distance = torch.normal(mean=0.195 * h, std=0.012 * h, size=(1,)).item()

    # Anchor the first rail at the left edge (x0=0) on a row between 39%–78%
    # of image height — same proportional band as the original 100–200 on 256 px.
    y0 = torch.randint(low=int(0.39 * h), high=int(0.78 * h) + 1, size=(1,)).item()
    x0 = 0

    # Convert tilt to radians and build the unit normal (nx, ny) to the rail
    # direction. Distance from a point to the line passing through (x0, y0)
    # with this normal is |(x - x0)*nx + (y - y0)*ny|.
    angle_rad = torch.tensor(angle_deg * 3.14159 / 180.0)
    nx = -torch.sin(angle_rad)
    ny = torch.cos(angle_rad)

    # Distances from every pixel to rail #1 and to the parallel rail #2
    # (shifted by `distance` along the normal direction).
    dist_from_line1 = torch.abs((x - x0) * nx + (y - y0) * ny)
    dist_from_line2 = torch.abs((x - x0) * nx + (y - (y0 + distance)) * ny)

    # A pixel belongs to a rail if it lies within ~1 px of either line,
    # which gives a ~2 px thick stroke.
    mask = (dist_from_line1 < 1.0) | (dist_from_line2 < 1.0)

    # Paint the mask with near-black plus a touch of noise so the rails are
    # not a perfectly uniform colour (helps avoid trivially-easy training).
    noise = 0.1 * torch.randn_like(tensor)
    line_color = torch.tensor([0.0, 0.0, 0.0]) + noise
    tensor[mask] = line_color[mask]

    return tensor, (angle_rad, x0, y0)


def _make_object(tensor, angle_rad, x0, y0):
    """Place a rectangular obstacle aligned to the rail frame.

    The obstacle is drawn so it sits roughly halfway between the two rails
    (perpendicular offset of ~25 px from rail #1, since the rails are ~50 px
    apart) and is rotated to match the rails' tilt. Returns the modified
    tensor and a bounding box `(y_min, x_min, y_max, x_max)`.
    """
    h, w, _ = tensor.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Object dimensions scaled proportionally to image size.
    # Original notebook used 10×70 px on 256 px → ~4% tall, ~27% wide.
    obj_h = int(0.039 * h)
    obj_w = int(0.273 * w)

    # Perpendicular offset toward the midpoint between the two rails.
    # Rails are ~19.5% of h apart, so half is ~9.75% — same proportion as
    # the original mean=25 on 256 px. Noise kept proportional too.
    mid_offset = torch.normal(mean=0.098 * h, std=0.012 * h, size=(1,)).item()

    # Choose where on the image the object should be placed.
    # We anchor x to the image centre and solve for the matching y so the
    # object lies on the rail line, then shift by `mid_offset` along the
    # normal to centre it between the rails.
    target_x = w // 2
    nx = -torch.sin(angle_rad)
    ny = torch.cos(angle_rad)
    target_y = int(-(target_x - x0) * (nx / ny) + y0 + mid_offset)

    # Build a rotation matrix that maps image coordinates -> the object's
    # local frame (aligned with the rails). We then test rectangle bounds in
    # that local frame.
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Shift to make the object centre the origin.
    dx = x - target_x
    dy = y - target_y

    # Apply the rotation:
    #   local_x =  dx*cos + dy*sin
    #   local_y = -dx*sin + dy*cos
    local_x = dx * cos_a + dy * sin_a
    local_y = -dx * sin_a + dy * cos_a

    # The pixel is inside the rectangle iff both local coordinates are
    # within the half-extents.
    mask = (local_x.abs() < (obj_w / 2)) & (local_y.abs() < (obj_h / 2))

    # Paint the obstacle near-black with a bit of noise (same idea as rails).
    noise = 0.1 * torch.randn_like(tensor)
    obj_color = torch.tensor([0.0, 0.0, 0.0]) + noise
    tensor[mask] = obj_color[mask]

    # Axis-aligned bounding box of the painted pixels. This is what the
    # detector is trained to predict. Falls back to a degenerate box if for
    # some reason nothing was drawn (e.g. object fully outside the image).
    if mask.any():
        y_idx, x_idx = torch.where(mask)
        bbox = (
            y_idx.min().item(),
            x_idx.min().item(),
            y_idx.max().item(),
            x_idx.max().item(),
        )
    else:
        bbox = None

    return tensor, bbox


# --------------------------------------------------------------------------- #
# Public loader — writes a YOLO-style dataset to disk
# --------------------------------------------------------------------------- #

def load_synthetic_rails(output_dir, n_samples=700, img_size=640, p_object=0.8, seed=42):
    """Generate (or reuse) a synthetic rails dataset on disk in YOLO format.

    Layout produced under `output_dir`:
        images/rail_00000.png ...
        labels/rail_00000.txt ...   # one line per object: "0 cx cy w h" (normalized)

    If the labels directory already has at least `n_samples` files we skip
    regeneration and just return the paths — makes re-runs cheap.
    """
    output_dir = Path(output_dir)
    images_dst = output_dir / "images"
    labels_dst = output_dir / "labels"
    classes = ["object"]

    # Cache: if enough labels already exist, treat the dataset as ready.
    if labels_dst.exists() and len(list(labels_dst.glob("*.txt"))) >= n_samples:
        return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    # Seed both torch (used by the primitives) and numpy (used for the
    # per-sample "has object?" coin flip) so generation is reproducible.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        # 1. Fresh noisy background.
        img = _make_background(img_size)

        # 2. Draw rails and remember their geometry so the object can align.
        img, (angle_rad, x0, y0) = _make_rails(img)

        # 3. Coin flip: place an obstacle with probability `p_object`.
        bbox = None
        if rng.random() < p_object:
            img, bbox = _make_object(img, angle_rad, x0, y0)

        # 4. Convert float-tensor [0, 1] -> uint8 numpy array for PIL.
        #    `.clamp` guards against the small Gaussian noise pushing values
        #    slightly outside the displayable range.
        img_np = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

        name = f"rail_{i:05d}"
        Image.fromarray(img_np).save(images_dst / f"{name}.png")

        # 5. Write the YOLO label file. Empty file = "background, no object".
        if bbox is None:
            (labels_dst / f"{name}.txt").write_text("")
        else:
            # bbox from _make_object is (y_min, x_min, y_max, x_max).
            y_min, x_min, y_max, x_max = bbox
            # YOLO expects centre-x, centre-y, width, height, all normalized
            # to the image size so the labels are resolution-independent.
            cx = (x_min + x_max) / 2 / img_size
            cy = (y_min + y_max) / 2 / img_size
            bw = (x_max - x_min) / img_size
            bh = (y_max - y_min) / img_size
            (labels_dst / f"{name}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    return {"images_dir": images_dst, "labels_dir": labels_dst, "classes": classes}
