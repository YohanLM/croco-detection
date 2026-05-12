# Synthetic Dataset Generation Specification

## Overview

Images are orthogonal (top-down) LiDAR height-map projections of railway track.
Colour encodes elevation above ground. The objective is to detect crocodile clips
clamped between rails. This document specifies what to generate and what
dataset configurations to produce.

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
- The tracks run **horizontally** across the full width of the image.
- Track position and spacing are **roughly constant** across the dataset:
  - The overall track band is roughly **vertically centred** in the image.
  - The two track sets are separated by a fixed inter-track gap (ballast between them).
  - Rail width and inter-rail spacing within each track set are consistent.
  - Small random jitter (a few percent of image height) is acceptable to avoid
    the model overfitting to a fixed pixel row — but do not vary it widely.
- The tracks occupy roughly the **middle third** of the image height. Above and
  below is ballast background.

### Rails

- Colour: **red** (slightly elevated above the ballast in the height map).
- Intensity varies slightly along the rail — occasional short segments are a
  touch brighter or darker than the surrounding rail, to reflect real scan
  density variation. This is subtle; the rail should still read as a continuous line.
- Rails are thin horizontal stripes, a few pixels thick.

### Sleepers (ladder pattern)

- **Black vertical traces** crossing both rails of each track set at regular
  intervals, forming a ladder pattern.
- Spacing and thickness have noise: not perfectly regular.
- Some sleepers **blend into the background** (reduced opacity / lower contrast)
  as if the LiDAR return was weak at that point.
- Sleepers are visible only within the track band; they do not extend into the
  open ballast beyond the rails.

### Switch point (optional)

- In some images, the two track sets **converge or diverge** at a switch point
  visible within the frame — multiple tracks meeting at a junction.
- When a switch point is present, one or more additional partial rail lines
  appear diagonally, connecting one track set to the other.
- This should appear in a minority of images (not the default case).

### Noise elements

A small number of **random scattered shapes** appear across the image, including
in the inter-rail zones. These are hard negatives — the model must learn to ignore
them:
- Shapes can be **green** (elevated debris / vegetation) or **red** (objects at
  rail height).
- They are irregular blobs or small rectangles, randomly sized and placed.
- Keep their count low (not too many per image) so they do not clutter the scene
  but do provide genuine ambiguity for the model.

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
- **Length:** approximately **1/8th of the image width**.
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

- All images are square, 640 × 640 px (matches YOLO's default input size directly).
- YOLO labels: normalised centre-x, centre-y, width, height (`0 cx cy w h`).
- An empty label file means the image is a true negative (no clip).
- The dataset cache check (skip regeneration if files already exist) should be
  per-configuration so regenerating one config does not invalidate others.
- The clip bounding box covers the **clip body only**, not any wire or cable that
  may extend from it.
