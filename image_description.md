# Synthetic Dataset Generation Specification

## Overview

Images are orthogonal (top-down) LiDAR height-map projections of railway track.
Colour encodes elevation above ground. The objective is to detect crocodile clips
clamped between rails. This document specifies what to generate and what
dataset configurations to produce.

Returns from image analysis:
  Background median RGB : [93 46  1]
  Rail median RGB       : [244 134  23]
  Sliced Wasserstein D  : 84.25  (well separated in colour space)

---

## Image structure

### Background

Dark reddish-brown, similar to the tone seen in the real samples. Not uniform —
apply granular per-pixel noise plus low-frequency variation (coarser blobs) to
mimic the irregular density of a real LiDAR point cloud over ballast. The texture
should look roughly like rough gravel photographed from directly above and coloured
in this palette.

### Track layout

- **Two parallel track sets** are present in most images. Each track set consists
  of two parallel rails.
- The tracks run **horizontally** across the full width of the image (570 px).
- The railway band dominates the image vertically: **96 px out of 100 px** is
  railway. Background is only a thin 2 px strip at the very top and bottom.
- Rail y-positions are **fixed** (with small noise only). Each rail is **2–3 px
  wide**. Spacings below are **top-edge to top-edge**. All values are for a
  100 px tall image:

  | Element | Top edge y | Bottom edge y (2 px / 3 px rail) |
  |---|---|---|
  | Rail 1 — upper track, top rail | 2 | 3 / 4 |
  | Rail 2 — upper track, bottom rail | 27 | 28 / 29 |
  | Rail 3 — lower track, top rail | 64 | 65 / 66 |
  | Rail 4 — lower track, bottom rail | 89 | 90 / 91 |

  Resulting **clear gaps** between rails (free ballast zone, no rail pixels):

  | Zone | 2 px rails | 3 px rails |
  |---|---|---|
  | Top background (above rail 1) | y = 0–1 **(2 px)** | y = 0–1 **(2 px)** |
  | Upper track inter-rail (rail 1 bottom → rail 2 top) | y = 4–26 **(23 px)** | y = 5–26 **(22 px)** |
  | Inter-track ballast (rail 2 bottom → rail 3 top) | y = 29–63 **(35 px)** | y = 30–63 **(34 px)** |
  | Lower track inter-rail (rail 3 bottom → rail 4 top) | y = 66–88 **(23 px)** | y = 67–88 **(22 px)** |
  | Bottom background (below rail 4) | y = 91–99 **(9 px)** | y = 92–99 **(8 px)** |

  The bottom background strip is wider than the top (8–9 px vs 2 px) because
  the rail positions are anchored at the top of the image. This is acceptable
  — the model sees very little pure background either way (~11% of image height).

  Top-to-top spacings: rail 1→2 = **25 px**, rail 2→3 = **37 px**, rail 3→4 = **25 px**.

- Small per-image jitter on all y-positions is allowed (±1–2 px) to prevent
  the model from memorising exact pixel rows, but the relative spacings above
  must be preserved.

### Rails

- Colour: **red** (slightly elevated above the ballast in the height map).
- Intensity varies slightly along the rail — occasional short segments are a
  touch brighter or darker than the surrounding rail, to reflect real scan
  density variation. This is subtle; the rail should still read as a continuous line.
- Rails are thin horizontal stripes, 2–3 px thick.
- **Rail motifs:** small additional markings appear on top of the rail lines to
  add visual complexity — e.g. isolated brighter or darker pixel clusters, very
  short perpendicular ticks, or faint blobs of slightly different red. These are
  not regular enough to be mistaken for sleepers and do not span the full rail width.

### Sleepers (ladder pattern)

- **Black vertical traces** crossing both rails of each track set at regular
  intervals, forming a ladder pattern.
- Spacing and thickness have noise: not perfectly regular.
- Some sleepers **blend into the background** (reduced opacity / lower contrast)
  as if the LiDAR return was weak at that point.
- Sleepers are visible only within the track band; they do not extend into the
  open ballast beyond the rails.

### Switch point (optional)

Present in a minority of images. Two non-exclusive variants can appear:

**Variant A — crossing between track sets**
A diagonal rail line branches off from one rail of one track set and connects
to one rail of the other track set. The branch starts at some x position within
the image and travels diagonally across the inter-track ballast gap (y=27→64)
until it meets the target rail. The angle is shallow — the 37 px vertical
distance must be covered over a long enough horizontal run that the line looks
like a realistic switch and not a steep crossing. The branching rail has the
same colour and width as a normal rail.

**Variant B — track leaving the image**
A rail branches off from one of the four main rails and exits through one of
the image edges (left, right, top, or bottom). The branch starts somewhere
within the horizontal span of the image, diverges at a shallow angle, and
simply ends at the image boundary. This represents a track that peels away
from the main corridor and leaves the sensor's field of view.

Both variants may appear in the same image. When a switch point is present,
the additional rail lines are drawn on top of the ballast but do not overwrite
existing rails — they cross or merge cleanly. The crocodile clip may be placed
on any of the rails involved in the switch, including the branching one.

### Background noise

A small random proportion of background pixels are **red** instead of the base
dark reddish-brown colour (RGB=[93, 46, 1], SWD=84.25 ). These are isolated pixels or 1–2 px clusters scattered 
across the ballast zones. They represent minor surface irregularities and prevent
the model from treating "all red = rail."

### Inter-rail fixed features (hard negatives)

Two non-clip features are always present inside the inter-rail zones of each
track set, serving as hard negatives the model must learn to ignore:

1. **Green dot** — a small roughly circular green blob (a few pixels in diameter),
   placed somewhere in the inter-rail clear gap. Not elongated, not at clip
   dimensions. Represents an elevated debris fragment. Always positioned to the
   **left of the red line** (see below).

2. **Red straight line** — a horizontal red line of similar width to the clip
   (65–75 px) placed in the inter-rail zone, always to the **right of the green
   dot**. It is thinner than the clip (1–2 px tall vs 4–5 px), fully red with no
   green centre, and its vertical position is not perfectly centred between the
   rails. This is the most dangerous hard negative: it shares the clip's width
   and colour but lacks the green core and precise centring.

### Inter-rail scattered dots

A few small **green or red dots** (1–3 px diameter) are scattered randomly within
the inter-rail zones of each track set. Count: 2–6 per track set per image.
These add ambiguity inside the very zone where the clip would appear, forcing
the model to rely on the clip's specific size, shape, and colour signature rather
than just "something is between the rails."

---

## Crocodile clip

### Appearance

- Present in **a minority of images** (proportion depends on the dataset
  configuration, see below).
- **Colour:** predominantly **red** body (same height band as the rails),
  with a random proportion of **green pixels concentrated toward the centre**
  — these represent the contact points or spring mechanism that protrude slightly
  higher than the body.
- **Shape:** a horizontal rectangle — long and narrow — oriented **along** the
  rail direction (i.e. wider than tall, running left-right).
- **Width:** 70 px ± up to 5 px random deviation each side → range **65–75 px**
  (≈ 1/8th of image width).
- **Height:** **4–5 px** (randomised per image), fitting comfortably within the
  22–23 px inter-rail clear gap.
- **Position:** well centred within the inter-rail zone of whichever track set it
  is placed in (vertically, it sits between the two rails of that set; horizontally,
  it is centred in the image with some small random jitter).

### Placement rules

The clip can be placed in:
- The **upper track set** (the track set closer to the top of the image), or
- The **lower track set**, or
- One of the tracks of a **switch point** (if one is present in the image).

Which of these is allowed depends on the dataset configuration (see below).

---

## Dataset configurations

Six configurations are defined. Each is a separate generated split with its own
parameters. They are designed to test different aspects of model robustness.

### 1. `test_sparse`
- **Purpose:** evaluation set with realistic low clip prevalence.
- Two track sets, no switch points.
- Very few clips (~10% of images have a clip).
- Many true negatives — tests the model's false-positive rate.

### 2. `test_dense`
- **Purpose:** evaluation set for measuring detection rate under high prevalence.
- Two track sets, no switch points.
- Higher clip proportion (~50% of images have a clip).
- Tests sensitivity / recall.

### 3. `train_two_tracks`
- **Purpose:** simplest training baseline — clean, predictable geometry.
- Always exactly two track sets, no switch points.
- Clip present in ~30% of images.
- Clip can be in either track set.

### 4. `train_with_switches`
- **Purpose:** teach the model to handle more complex track geometry.
- Two track sets in all images, switch point present in a minority (~20%).
- Clip present in ~30% of images.
- Clip can be in any track set including the switch tracks.

### 5. `train_upper_only`
- **Purpose:** ablation — clip is restricted to the upper track set.
- Two track sets, no switch points.
- Clip present in ~30% of images.
- Clip is **only ever placed in the upper track set**.
- Used to test whether restricting placement helps or hurts generalisation.

### 6. `train_any_track`
- **Purpose:** generalisation training — clip position is fully randomised.
- Two track sets, no switch points.
- Clip present in ~30% of images.
- Clip is placed in the upper or lower track set with equal probability.

---

## Notes for implementation

- Images are **570 × 100 px** — this matches the real cropped LiDAR exports.
  Do not generate at 640×640; the synthetic images must match the real domain size.
- Image width (570 px) is fully occupied by rails — no horizontal margins.
- Rail geometry: top edges at y = 2, 27, 64, 89. Rail width = 2–3 px (randomised
  per image). Top-to-top spacings: 25 px (inter-rail), 37 px (inter-track), 25 px.
  Clear inter-rail gaps: ~22–23 px. Inter-track ballast gap: ~34–35 px.
  Background: 2 px at top, 8–9 px at bottom (anchored at top edge of image).
- YOLO input preprocessing: use `rect=True` during training and validation so
  YOLO scales to 640×128 (padding only 8 px top/bottom) instead of letterboxing
  to 640×640 (which would waste ~82% of the network on gray padding).
- YOLO labels: normalised centre-x, centre-y, width, height (`0 cx cy w h`).
- An empty label file means the image is a true negative (no clip).
- The dataset cache check (skip regeneration if files already exist) should be
  per-configuration so regenerating one config does not invalidate others.
- The clip bounding box covers the **clip body only**, not any wire or cable that
  may extend from it.
