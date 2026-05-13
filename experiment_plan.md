# Experiment Plan

Three training runs total. One full curve to find the minimum useful training size, two spot-checks to see how format and colour affect it.

---

## Run 1 — Main (full curve)

**Goal:** find the training-set size at which results become acceptable.

- Images: square (640 × 640)
- Colour: colour
- `p_clip` = 0.30, motif gate = 0.15
- Subset sizes: **100, 200, 400, 800, 1600, 2800**

This produces the learning curve we care about most. We read off the point where mAP@.5 flattens — call it N*.

---

## Run 2 — Spot-check: cropped rectangular images

**Goal:** rough idea of whether the smaller, focused crop needs more or fewer images.

- Images: rectangular (570 × 100, `rect=True`)
- Colour: colour
- Same generator config as Run 1
- Subset sizes: **200, 800**

Two points straddle the likely knee of the curve. If both are close to Run 1 at the same sizes, the format barely matters. If they diverge, we know which direction and can decide whether a fuller curve is worth it.

---

## Run 3 — Spot-check: greyscale

**Goal:** rough idea of how much the colour cue was helping.

- Images: square (same as Run 1), converted with `make_greyscale.py`
- Colour: greyscale
- Same generator config as Run 1
- Subset sizes: **200, 800**

Same logic as Run 2. Compare the two greyscale points against the Run 1 curve at the same sizes to quantify the colour penalty.

---

## Producing the datasets

```bash
# Run 1 + 3: generate square colour dataset once
python dataset_synthetic_square.py   # -> data/dataset/sq_c30_m15_col

# Run 3: convert to greyscale
python make_greyscale.py data/dataset/sq_c30_m15_col data/dataset/sq_c30_m15_grey

# Run 2: generate rectangular colour dataset
python dataset_synthetic.py          # -> data/dataset/rect_c30_m15_col
```

---

## Wall-time estimate (Apple M2, MPS, 25 epochs)

| Run | Sizes | Training runs | Est. time |
|---|---|---|---|
| Run 1 | 6 sizes | 6 | ~90 min |
| Run 2 | 2 sizes | 2 | ~30 min |
| Run 3 | 2 sizes | 2 | ~30 min |
| **Total** | | **10** | **~2.5 h** |
