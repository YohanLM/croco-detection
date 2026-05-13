# Experiment Plan

Each comparison varies **one thing at a time** against a shared baseline. No full factorial crossing.

Baseline: `rect_c15_m5_col` — rectangular images, 15 % clip rate, 5 % motif gate, colour.

---

## Comparisons

### A — Clip density

Does a higher clip rate reduce the training set needed? Does sparse clip presence inflate the required size?

| Dataset | `p_clip` | motif gate | shape | colour |
|---|---|---|---|---|
| `rect_c05_m5_col` | 5 % | 5 % | rect | colour |
| **`rect_c15_m5_col`** *(baseline)* | **15 %** | **5 %** | rect | colour |
| `rect_c30_m5_col` | 30 % | 5 % | rect | colour |

Everything else is fixed. The metric of interest is the training-set size at which the mAP@.5 learning curve flattens.

### B — Hard-negative load

Does a heavier dose of confusers (motif A / motif B) force the model to need more examples?

| Dataset | `p_clip` | motif gate | shape | colour |
|---|---|---|---|---|
| `rect_c15_m2_col` | 15 % | 2 % | rect | colour |
| **`rect_c15_m5_col`** *(baseline)* | **15 %** | **5 %** | rect | colour |
| `rect_c15_m15_col` | 15 % | 15 % | rect | colour |

Motif gate 0 % is excluded on purpose — removing confusers entirely would not tell us anything useful about the real setting where they exist.

### C — Colour vs greyscale (both shapes)

Does stripping colour force the model to rely on shape, and if so, how much more data does it need?
Tested on both shapes to check whether the effect is shape-dependent.

| Dataset | shape | colour | how produced |
|---|---|---|---|
| **`rect_c15_m5_col`** *(baseline)* | rect | colour | generator |
| `rect_c15_m5_grey` | rect | greyscale | `make_greyscale.py` on col dataset |
| `sq_c15_m5_col` | square | colour | generator |
| `sq_c15_m5_grey` | square | greyscale | `make_greyscale.py` on col dataset |

The greyscale variants cost no extra generation — just run:

```bash
python make_greyscale.py data/dataset/rect_c15_m5_col data/dataset/rect_c15_m5_grey
python make_greyscale.py data/dataset/sq_c15_m5_col   data/dataset/sq_c15_m5_grey
```

---

## Dataset naming convention

```
<shape>_<clip rate>_<motif gate>_<colour>
```

| Tag | Values |
|---|---|
| shape | `rect` / `sq` |
| clip rate | `c05` / `c15` / `c30` |
| motif gate | `m2` / `m5` / `m15` |
| colour | `col` / `grey` |

---

## Keeping the run count down

**Total distinct datasets:** 7 (3 for A, 2 new for B, 2 new for C — baseline shared).
**Training runs per dataset:** up to 7 subset sizes → **up to 49 runs total**.

Two further savings:

**Stop early on the learning curve.** Run subsets 50 → 100 → 200 → 400 first. Only extend to 800 → 1600 → 2800 if the curve hasn't clearly flattened yet. A dataset that reaches plateau at 200 images saves 3 runs outright.

**Prioritise in order A → B → C.** Clip density (A) is the most fundamental question. If `c05` and `c30` both plateau at the same size as `c15`, hard-negative load (B) becomes the interesting question. Shape and colour (C) are run last; if the baseline already produces strong results, C may only need a partial curve.

---

## Rough wall-time budget (Apple M2, MPS)

| Step | Time | Count | Total |
|---|---|---|---|
| Image generation (4 000 imgs) | ~2 min | 5 colour datasets | ~10 min |
| Greyscale conversion | < 1 min | 2 grey datasets | negligible |
| YOLO training (25 ep, 640px) | ~10–15 min/run | ≤ 49 runs | ~8–12 h |
| YOLO val | ~1 min/run | ≤ 49 runs | ~1 h |

Plan for 3–4 sessions of ~2 h, starting with comparison A.
