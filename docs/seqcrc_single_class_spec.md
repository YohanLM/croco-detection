# SeqCRC for Single-Class Conformal Object Detection — Implementation Notes

Implementation spec for Sequential Conformal Risk Control (SeqCRC), specialized to **one object class**, a **False-Negative-Rate (FNR) confidence loss**, and a **pixel-superposition-threshold localization loss**. Based on Andéol et al., "Conformal Object Detection by Sequential Risk Control" (arXiv:2505.24038v2).

Hand this to Copilot as the algorithm contract. Equations are written in ASCII for unambiguous parsing.

---

## 0. Specialization assumptions (read first)

The paper has **three** parameters (`confidence`, `localization`, `classification`). For this project:

1. **No classification task.** Only one class exists, so the classification parameter `lambda_cls`, its prediction set, and `L_cls` are removed entirely. SeqCRC collapses from "1+2" to **"1+1": one confidence step, then one localization step.**
2. **Matching reduces to pure localization.** The classification distance `d_LAC(c, c_hat) = 1 - c_hat[c]` is ~0 with a single class, so the mixed distance becomes the **asymmetric signed Hausdorff distance** alone (no `tau` mixing parameter needed).
3. **FNR loss** is used at the confidence step (`L_cnf`).
4. **Pixel-superposition-threshold loss** is used at the localization step (`L_loc`). I interpret "pixel superposition" as the fraction of a ground-truth box's area overlapped ("superposed") by its matched prediction, thresholded at a fixed `tau_pix`, then aggregated as a recall. See Section 4.2 for the exact definition and two alternatives — confirm this matches your intent before implementing.
5. All losses are `[0, 1]`-valued, so every loss bound is `B = 1`.

The placeholder `•` in the paper (ranging over `{loc, cls}`) here means just `loc`.

---

## 1. Notation

- `X`            : input image.
- `f`            : trained object detector (deterministic), **already includes NMS**. Outputs a list of `(box, confidence)` tuples. (Single class → softmax/class vector is ignored.)
- `b = (b_left, b_top, b_right, b_bottom)` : box corners. Convention: x-axis right, y-axis **down**. `b_left <= b_right`, `b_top <= b_bottom`.
- `y`            : ground-truth list of boxes for an image; `|y|` = number of true objects.
- `o(x)[k]`      : k-th largest confidence score on image `x`.
- `n`            : number of calibration images.
- Box inclusion `b ⊆ b_hat` means: `b_left >= b_hat_left AND b_top >= b_hat_top AND b_right <= b_hat_right AND b_bottom <= b_hat_bottom`.
- `area(b) = (b_right - b_left) * (b_bottom - b_top)`.

Parameters to calibrate:
- `lambda_cnf_plus`  : confidence threshold parameter used **at inference**.
- `lambda_cnf_minus` : "optimistic" confidence parameter, used **only inside** the localization step.
- `lambda_loc_plus`  : localization margin parameter used **at inference**.

Parameter spaces: `Lambda_cnf = [0, 1]` (so `lambda_bar_cnf = 1`); `Lambda_loc = [0, lambda_bar_loc]` with `lambda_bar_loc` bounded by image size.

Target error rates (chosen a priori, **never** from data): `alpha_cnf`, `alpha_loc`.

---

## 2. Prediction sets

### 2.1 Confidence set
For threshold parameter `lambda_cnf in [0, 1]`, keep boxes whose confidence is at least `1 - lambda_cnf`:

```
Gamma_cnf(x, lambda_cnf) = { (b_k, o_k) in f(x) : o_k >= 1 - lambda_cnf }
```

Larger `lambda_cnf` => lower threshold => more boxes kept.

### 2.2 Localization set
Expand each kept box by margin `lambda_loc`. Two options (pick one, keep it consistent between calibration and inference):

```
# Additive margin (uniform pixels on all sides):
expand(b_hat, lambda_loc) = b_hat + lambda_loc * (-1, -1, +1, +1)

# Multiplicative margin (scales with box size; w_hat, h_hat = predicted width/height):
expand(b_hat, lambda_loc) = b_hat + lambda_loc * (-w_hat, -h_hat, +w_hat, +h_hat)
```

`Gamma_loc(x, lambda_cnf, lambda_loc)` = the boxes of `Gamma_cnf` each passed through `expand(., lambda_loc)`. Same cardinality as `Gamma_cnf`.

---

## 3. Matching (Hausdorff)

For each image, build a matching `pi` mapping each true-box index `j` to a predicted-box index in `Gamma_cnf`. Match each true box to its **nearest prediction** under the asymmetric signed Hausdorff distance:

```
d_haus(b, b_hat) = max( b_hat_left - b_left,
                        b_hat_top  - b_top,
                        b_right     - b_hat_right,
                        b_bottom    - b_hat_bottom )
```

Interpretation: the smallest per-side margin that, when added to `b_hat`, makes it fully cover `b`. Matching need not be injective (multiple true boxes may map to the same prediction). The matching is recomputed whenever `Gamma_cnf` changes (i.e., whenever `lambda_cnf` changes).

---

## 4. Loss functions (per image `i`)

Both losses return values in `[0, 1]`. Handle the two edge cases everywhere:
- `|y| == 0` (no ground truth) => loss = `0`.
- `Gamma_cnf` empty (no predictions kept) => loss = `B = 1`.

### 4.1 Confidence loss = FNR (box-count-recall)
False-negative rate on object **count** — penalizes keeping fewer boxes than there are ground truths:

```
L_cnf_i(lambda_cnf):
    G = Gamma_cnf(x_i, lambda_cnf)
    if |y_i| == 0: return 0
    return max(0, |y_i| - |G|) / |y_i|
```

This is `1 - recall` measured by count. It is non-increasing in `lambda_cnf` (lower threshold => more boxes => lower FNR), so it satisfies the monotonicity assumption directly.

> Stricter binary alternative (`box_count_threshold`): `return 0 if |G| >= |y_i| else 1`.
> Matching-based "true" FNR (counts only objects actually missed, not raw count) exists in the paper's Appendix B if count-based FNR is too loose for you.

### 4.2 Localization loss = pixel-superposition threshold
A true box is "covered" if the **fraction of its pixels overlapped by the matched, margin-expanded prediction** meets a fixed threshold `tau_pix in (0, 1]`. The loss is the recall over covered boxes:

```
L_loc_i(lambda_cnf, lambda_loc):
    G = Gamma_cnf(x_i, lambda_cnf)
    if |y_i| == 0: return 0
    if |G| == 0:   return 1            # = B
    pi = match(y_i, G)                 # Hausdorff, Section 3
    covered = 0
    for j, b_j in enumerate(y_i):
        b_hat = expand(G[pi(j)], lambda_loc)
        frac  = area(intersect(b_j, b_hat)) / area(b_j)   # "superposition" fraction
        if frac >= tau_pix:
            covered += 1
    return 1 - covered / |y_i|
```

`tau_pix` is a hyperparameter fixed **before** looking at calibration data.

> Two related variants (swap in if "pixel superposition" means something else to you):
> - **Pure pixelwise** (no per-box threshold): `return 1 - mean_j( area(intersect(b_j, b_hat_j)) / area(b_j) )`.
> - **Strict inclusion** (`box_count_recall` of the paper, Eq. 14): replace the `frac >= tau_pix` test with full inclusion `b_j ⊆ b_hat`.

**Monotonicity caveat (important):** `L_loc` is non-increasing in `lambda_loc` (bigger margin => more coverage), **but not necessarily in `lambda_cnf`**, because adding boxes changes the matching `pi`. SeqCRC's guarantee requires monotonicity in `lambda_cnf`, so we enforce it on the fly (Section 6).

---

## 5. Calibration math (the two-step SeqCRC, specialized)

Empirical risks over `n` calibration images:

```
R_cnf(lambda_cnf)            = (1/n) * sum_i L_cnf_i(lambda_cnf)
R_loc(lambda_cnf, lambda_loc) = (1/n) * sum_i L_loc_i(lambda_cnf, lambda_loc)
```

Conservative confidence risk that anticipates the localization step (the classification term is dropped):

```
R_tilde_cnf(lambda_cnf) = max( R_cnf(lambda_cnf),
                               R_loc(lambda_cnf, lambda_bar_loc) )
```

`B_tilde_cnf = max(B_cnf, B_loc) = 1`.

> Note: `R_loc(., lambda_bar_loc)` uses the **maximum** margin, where coverage is essentially total, so this term is ~0 whenever at least one box is kept. Its real job is feasibility: if no box is kept it equals 1, forcing `lambda_cnf` large enough to keep predictions.

### Step 1 — confidence (two estimators)

```
lambda_cnf_plus  = inf { lambda_cnf in [0,1] :
                         n/(n+1) * R_tilde_cnf(lambda_cnf) + 1/(n+1) <= alpha_cnf }

lambda_cnf_minus = inf { lambda_cnf in [0,1] :
                         n/(n+1) * R_tilde_cnf(lambda_cnf) + 0/(n+1) <= alpha_cnf }
```

Convention: `inf(empty set) = lambda_bar_cnf = 1`. By construction `lambda_cnf_minus <= lambda_cnf_plus`. `lambda_cnf_plus` is used at inference; `lambda_cnf_minus` (optimistic) feeds Step 2.

### Step 2 — localization

```
lambda_loc_plus = inf { lambda_loc in [0, lambda_bar_loc] :
                        n/(n+1) * R_loc(lambda_cnf_minus, lambda_loc) + 1/(n+1) <= alpha_loc }
```

> The single data split is reused for both steps; using `lambda_cnf_minus` (not `lambda_cnf_plus`) here is exactly what makes the finite-sample guarantee hold without a second split. Do not "simplify" this to one confidence estimator.

---

## 6. Monotonization trick (enforced in both steps)

Because `L_loc_i` may be non-monotone in `lambda_cnf`, replace it with its smallest provably-monotone upper bound:

```
L_loc_i_mono(lambda_cnf, lambda_loc) = sup_{ lambda' >= lambda_cnf } L_loc_i(lambda', lambda_loc)
```

Computed **on the fly** while sweeping `lambda_cnf` from high to low. Key fact enabling this: `Gamma_cnf(x, lambda_cnf)` is a right-continuous, piecewise-constant function of `lambda_cnf`, with jumps exactly at the predicted confidence scores `o(x)[k]`. So sweep over the sorted confidence scores and take a running `max` of the loss. (Optionally also monotonize `L_cnf`; the FNR loss is already monotone, so this is a no-op here but harmless.)

---

## 7. Pseudocode — calibration

Run once on the calibration split. Pre-step: filter out predictions with confidence `< 1e-3` (fixed, data-independent) to avoid pathological behavior near `lambda_cnf = 1`; this preserves the guarantee.

### 7.1 Top-level (paper Algorithm 1)

```
function CALIBRATE(D_cal = {(X_i, Y_i)}_{i=1..n}, f, alpha_cnf, alpha_loc,
                   margin_mode in {additive, multiplicative}, tau_pix, lambda_bar_loc):
    # 1. Predict + prefilter
    for i in 1..n:
        Yhat_i = f(X_i)                      # includes NMS
        Yhat_i = [ p for p in Yhat_i if p.conf >= 1e-3 ]

    # 2. Confidence step: two estimators (B = 1 and B = 0)
    lambda_cnf_plus  = CALIBRATE_CONFIDENCE(Yhat, Y, alpha_cnf, B=1, lambda_bar_loc, tau_pix, margin_mode)
    lambda_cnf_minus = CALIBRATE_CONFIDENCE(Yhat, Y, alpha_cnf, B=0, lambda_bar_loc, tau_pix, margin_mode)

    # 3. Localization step (matching is recomputed inside as lambda_cnf varies)
    lambda_loc_plus  = CALIBRATE_LOCALIZATION(Yhat, Y, alpha_loc, B=1, lambda_cnf_minus,
                                              l=0, u=lambda_bar_loc, S=num_bisection_steps,
                                              tau_pix, margin_mode)

    return lambda_cnf_plus, lambda_loc_plus     # only the _plus values are used at inference
```

### 7.2 Confidence subroutine (paper Algorithm 3) — returns `lambda_cnf_plus` if `B=1`, else `lambda_cnf_minus`

```
function CALIBRATE_CONFIDENCE(Yhat, Y, alpha_cnf, B, lambda_bar_loc, tau_pix, margin_mode):
    # Flatten all confidence scores across all images, remembering the image index.
    scores, img_idx = flatten_scores_with_image_index(Yhat)
    sort (scores, img_idx) by ascending score
    # candidate thresholds: lambda_cnf = 1 - score; shift-left so each step lowers lambda_cnf
    scores = shift_left(scores); scores[last] = 1

    lambda_cnf = 1
    # init per-image loss arrays at lambda_cnf = 1 (keep everything)
    Lcnf = [ L_cnf_i(1)                          for each image i ]
    Lloc = [ L_loc_i(1, lambda_bar_loc)          for each image i ]   # monotonization base
    R = max( mean(Lcnf), mean(Lloc) )

    for (c, i) in zip(scores, img_idx):
        prev = lambda_cnf
        lambda_cnf = 1 - c
        Lcnf[i] = L_cnf_i(lambda_cnf)
        Lloc[i] = max( Lloc[i], L_loc_i(lambda_cnf, lambda_bar_loc) )   # running sup => monotone
        R = max( mean(Lcnf), mean(Lloc) )
        # stopping condition = first lambda_cnf violating the bound; infimum is the previous value
        if (n/(n+1)) * R + B/(n+1) > alpha_cnf:
            return prev
    return 0     # bound satisfied all the way down
```

### 7.3 Localization subroutine (paper Algorithm 4) — binary search + on-the-fly monotonization

```
function CALIBRATE_LOCALIZATION(Yhat, Y, alpha_loc, B, lambda_cnf_minus, l, u, S, tau_pix, margin_mode):
    result = None
    for step in 1..S:
        lambda_loc = (l + u) / 2

        # monotonize L_loc in lambda_cnf: sweep lambda' from 1 down to lambda_cnf_minus, running max
        scores, img_idx = flatten_scores_with_image_index(Yhat)
        sort (scores, img_idx) by ascending score
        scores = shift_left(scores); scores[last] = 1

        Lmono = [ L_loc_i(1, lambda_loc) for each image i ]
        lambda_prime = 1
        for (c, i) in zip(scores, img_idx):
            lambda_prime = 1 - c
            Lmono[i] = max( Lmono[i], L_loc_i(lambda_prime, lambda_loc) )
            if lambda_prime <= lambda_cnf_minus:    # only need sup over lambda' >= lambda_cnf_minus
                break
        R = mean(Lmono)

        if (n/(n+1)) * R + B/(n+1) <= alpha_loc:
            result = lambda_loc
            u = lambda_loc          # constraint holds => try a smaller margin
        else:
            l = lambda_loc          # constraint violated => need a larger margin

    if result is None: raise error("no feasible localization margin found")
    return result                   # slight upper-approximation of lambda_loc_plus
```

`L_loc` is non-increasing in `lambda_loc`, so the feasible set is an upper interval `[lambda_loc_plus, lambda_bar_loc]`; bisection narrows toward its left endpoint. Use `S` ~ 20-30 steps for sub-pixel precision.

---

## 8. Pseudocode — inference (paper Algorithm 2)

```
function INFER(X_test, f, lambda_cnf_plus, lambda_loc_plus, margin_mode):
    Yhat = f(X_test)                                 # NMS
    Yhat = [ p for p in Yhat if p.conf >= 1e-3 ]     # same prefilter as calibration

    # Step 1: confidence filtering
    Gamma_cnf = [ p for p in Yhat if p.conf >= 1 - lambda_cnf_plus ]

    # Step 2: localization margin
    Gamma_loc = [ expand(p.box, lambda_loc_plus, margin_mode) for p in Gamma_cnf ]

    return Gamma_loc     # list of conformally expanded boxes (single class, no class set)
```

Inference is cheap — just set construction — negligible vs. the `f` forward pass.

---

## 9. Guarantee and required parameter relationship

Under the i.i.d. + monotonicity assumptions, with the monotonization applied, SeqCRC gives:

```
E[ L_loc_test(lambda_cnf_plus, lambda_loc_plus) ] <= alpha_loc
```

provided the targets satisfy:

```
alpha_loc >= alpha_cnf + B_loc/(n+1) = alpha_cnf + 1/(n+1)
```

If additionally `L_cnf_i(1) <= alpha_cnf` almost surely for all i (i.e., keeping all boxes never violates the FNR budget), you also get the confidence guarantee:

```
E[ L_cnf_test(lambda_cnf_plus) ] <= alpha_cnf
```

The expectation is over the joint draw of calibration set + test point (marginal coverage), **not** conditional on a specific image. There is no separate `alpha_tot` corollary to apply here, because with no classification task there is only one second-step task — `alpha_loc` is the operative target. Validate the `alpha_loc >= alpha_cnf + 1/(n+1)` inequality at startup and fail loudly if violated.

---

## 10. Practical notes / gotchas

- **Data split.** If no held-out annotated set exists, split a validation set in half: one half calibration, one half evaluation (paper uses n = 2500 each). Calibration and test must be i.i.d.
- **Prefilter.** Drop `conf < 1e-3` before everything (calibration and inference identically). Data-independent, preserves guarantees, prevents huge margins.
- **`tau_pix` and margin_mode are fixed a priori.** Never tune them on calibration data, or the guarantee breaks.
- **Edge cases.** `|y| == 0 => loss 0`; empty `Gamma_cnf => loss B = 1`. Both must be handled in every loss call.
- **Box format.** `xyxy` (corner) vs `xywh` (center+size) give different correction geometry; the additive `(-1,-1,1,1)` margin above assumes `xyxy`. Convert consistently.
- **Recall-only semantics.** This setup guarantees you do not *miss* objects (low FNR) and that kept boxes *cover* their ground truth; it does **not** bound false positives / precision. Expect occasional spurious extra boxes — that is by design.
- **Performance.** The monotonizing sweeps dominate runtime. Sort confidence scores once per subroutine call and reuse.

---

## 11. Suggested module layout for implementation

```
geometry.py     : area(), intersect(), expand(box, lambda_loc, mode)
matching.py     : d_haus(), match(y, G)  -> dict {true_idx -> pred_idx}
losses.py       : L_cnf_i (FNR), L_loc_i (pixel-superposition-threshold), with edge-case handling
calibrate.py    : CALIBRATE, CALIBRATE_CONFIDENCE, CALIBRATE_LOCALIZATION
infer.py        : INFER
config.py       : alpha_cnf, alpha_loc, tau_pix, margin_mode, prefilter=1e-3, bisection_steps,
                  startup assert: alpha_loc >= alpha_cnf + 1/(n+1)
```

Reference implementation to cross-check against: https://github.com/leoandeol/cods
