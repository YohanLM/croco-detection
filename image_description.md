# Generic Image Description — Synthetic Rail Dataset

## What this data is

These are orthogonal (top-down) LiDAR point cloud images. The sensor fires laser
pulses downward and records the height of whatever each pulse hits. That height is
then mapped to a colour, so the image is essentially a **height map**, not a
photograph. The camera angle is fixed and perfectly vertical — there is no
perspective distortion, no shadows, no lighting variation.

---

## Colour encoding (height → colour)

| Colour | Height meaning |
|---|---|
| White | No return / missing data (sparse areas, edges of scan) |
| Orange / brown | Ground level — ballast, soil, the main surface |
| Dark red / deep brown | Slightly elevated above ballast — the rail head sits here |
| Green | Clearly elevated above the rail plane — objects resting on or between the rails |

This is the single most important structural fact for the model: **green means
"something is sticking up above the rails."** The crocodile clip is green for
exactly this reason.

---

## Image layout

The track runs **horizontally across the full width** of the image. The two rails
are centred vertically and occupy roughly the middle third of the image height.
Above and below the rails is ballast, and further out the scan becomes sparse
(white patches, edge noise).

There is no tilt — the rails are parallel to the horizontal axis. This is the
result of the orthogonal rotation pre-processing that was applied to the raw scans.

---

## Ballast (background texture)

The dominant visual element. Orange-brown, granular, noisy — it looks like coarse
static. The texture is not uniform: density and exact hue vary slightly across the
image because the point cloud is denser in some areas. There are occasional white
gaps where no points were returned.

**For training:** the model needs to learn to ignore this texture. The more varied
the synthetic ballast, the harder it is for the model to overfit to a specific
pattern. Exact realism is not necessary — the key property is that it is
orange-brown, granular, and covers the majority of the image.

---

## Rails

Two thin, continuous horizontal lines running the full width of the image. They
appear as slightly darker and more saturated dark-red/brown compared to the
surrounding ballast — this is because the rail head is a few centimetres above
the ballast surface, pushing it into a different height band.

The rails are separated by roughly 10–15% of the image height (standard gauge
track from above). They are parallel, straight, and at a constant vertical
position.

**For training:** the rail lines are the primary spatial anchor — the model needs
to learn that the object of interest lives *between* them. Rail colour contrast
against ballast is mild; the exact shade is less important than the geometry
(two parallel horizontal lines).

---

## Sleepers (ties)

Dark transverse stripes crossing both rails at regular intervals, perpendicular
to the rail direction. They are slightly darker than the ballast and create a
regular "ladder" pattern along the track. Spacing is roughly every 3–5% of image
width.

**For training:** sleepers are background clutter. The model should learn to ignore
them. However, their presence helps break the visual uniformity of the ballast and
prevents the model from relying on a perfectly flat background — worth including in
synthetic images.

---

## Crocodile clip (the object to detect)

Present in one image only (image 0). Located **between the two rails**, roughly
in the upper-centre of the inter-rail zone.

Appearance:
- **Colour: mostly red with scattered green dots** — the clip body sits at roughly
  rail height (red/dark-red in the height map), with a few green pixels concentrated
  toward the centre where the clip's contact points or spring mechanism protrude
  slightly higher. It is NOT a uniform green object.
- **Shape: small elongated rectangle**, oriented roughly perpendicular to the rail
  direction (i.e. it crosses from one rail toward the other). A few pixels wide and
  maybe 10–20 pixels tall at the image resolutions seen.
- A thin **wire or cable** may extend from the clip upward (toward the top of the
  image), very narrow (1–2 px). This may or may not be present depending on how
  the clip is installed.

**For training — what matters most:**
1. **Its position between the two rails.** This is now the primary discriminating
   feature. The colour signature (red body + green specks) blends with the
   surrounding rail colours and is not reliably distinctive on its own.
2. The combination of red + a few green dots in the inter-rail zone. A red-only
   or green-only object at that location is ambiguous; the mixed signature is more
   specific to the clip.
3. The small physical size relative to the image. At 640 px the clip occupies
   perhaps 0.5–2% of image area — YOLO's small-object detection needs enough
   examples to handle this.

**What matters less:**
- The exact shape of the clip body (rectangle is fine).
- The wire/cable extending from it (adds realism but is not the distinctive feature).
- The exact proportion of green vs red — it varies clip to clip.

---

## Other green objects (noise)

In images 2–4, small scattered green blobs appear throughout the ballast zone.
These are vegetation (weeds, grass) or small debris elevated above ground level.
They are visually similar in colour to the clip but differ in shape (rounder,
more irregular) and location (outside the inter-rail zone).

**For training:** these are the hard negatives. The model must learn to
distinguish "green blob between the rails" from "green blob outside the rails."
Including them in synthetic images is important — without them the model can cheat
by flagging any green pixel regardless of position.
