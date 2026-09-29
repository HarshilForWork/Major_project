# Conformal Confidence — Design

**Status: designed, not built.** This records the design and the constraints this dataset puts
on it.

---

## What it's for

The classifier outputs a probability vector, e.g.

```
TD 0.52   PIGD 0.41   Indeterminate 0.07
```

Gradient-boosted trees aren't calibrated, so `0.52` is a score, not a probability a clinician
can act on. Conformal prediction turns it into a **prediction set** with a coverage guarantee:

```
{TD, PIGD}   at 90% coverage
```

**Guarantee (marginal):** across many patients, the true label falls inside the set 90% of the
time.

The useful signal is the **set size**:

| Set | Meaning |
|---|---|
| `{TD}` | confident — one label survives at 90% |
| `{TD, PIGD}` | can rule out Indeterminate, can't choose between these two |
| `{TD, PIGD, Ind}` | no information — say so rather than guess |

---

## How the threshold `q̂` is computed

### 1. Split by patient — three ways

| Split | Used for |
|---|---|
| Train | fit the classifier |
| **Calibration** | model **frozen** → compute scores → take `q̂` |
| Test | check coverage actually lands at 90% |

**`q̂` is never computed on training data.** The model has already fit those rows, so it gives
the true class an inflated probability. The scores come out too low, `q̂` too small, and the
sets too small. You'd report 90% coverage and deliver less.

### 2. Nonconformity score on the calibration set

How "strange" did the **true** label look to the model?

**APS (Adaptive Prediction Sets)** — the method specified. Sort the probabilities
descending, and sum up to and including the true class:

```
s(x, y_true) = Σ_{j=1..k} p̂_(j)        where the true class sits at rank k
```

- true label ranked first with p = 0.91 → s = 0.91
- true label ranked second behind 0.52 → s = 0.52 + 0.41 = 0.93 (worse, correctly)

### 3. Take the quantile

```
rank = ⌈(n + 1)(1 − α)⌉          α = 0.10 for 90% coverage
q̂    = the rank-th smallest calibration score
```

The `(n + 1)` is the finite-sample correction that makes the guarantee exact.
Example: n = 800 → rank = ⌈801 × 0.9⌉ = 721 → `q̂` = 721st smallest of 800.

### 4. At inference

Walk the classes in descending probability, accumulating, and include each until the
cumulative mass passes `q̂`. The class that crosses **is included**.

With `q̂ = 0.86`:

| Patient | Probabilities | Walk | Set |
|---|---|---|---|
| A | TD .52, PIGD .41, Ind .07 | .52 ≤ .86 → include; .93 > .86 → include, stop | `{TD, PIGD}` |
| B | TD .91, PIGD .07, Ind .02 | .91 > .86 → include, stop | `{TD}` |
| C | TD .45, PIGD .30, Ind .25 | .45 → .75 → 1.00 | `{TD, PIGD, Ind}` |

One threshold for the whole system; the set size does the per-patient work.

---

## APS vs LAC

The simpler **LAC** score is `s = 1 − p̂(y)`. Including y when `s ≤ q̂` is equivalent to
`p̂(y) ≥ 1 − q̂` — a plain **probability cutoff**, just one derived from calibration instead of
picked by hand.

APS is cumulative, so it **isn't** a per-class cutoff. A class with p = 0.05 can be included
if the ones above it haven't reached `q̂`; a class with p = 0.30 can be excluded if the top
class already crossed.

They diverge on flat outputs. For patient C above (TD .45 / PIGD .30 / Ind .25):

| | Result |
|---|---|
| LAC, cutoff 0.40 | `{TD}` — "confident TD" |
| APS | `{TD, PIGD, Ind}` — "no information" |

APS is the honest one here. Both hit 90% marginal coverage, but they place the misses
differently: **LAC misses disproportionately on hard cases**, which is the wrong place to miss
in a clinical tool. **Decision: APS.**

---

## Constraints this dataset imposes

### Exchangeability

Conformal assumes calibration and test examples are exchangeable. Our rows are ~17 highly
correlated visits per patient.

- Splitting by `PATNO` stops the same patient appearing in calibration and test.
- But many visits from one patient **inside** the calibration set still break row-level
  exchangeability.

**Options:** one row per patient in calibration, or a cluster / block conformal variant.

### Calibration size

439 patients, split 60 / 20 / 20 by patient → roughly **263 / 88 / 88**. With one row per
patient, n_cal ≈ 88:

```
rank = ⌈89 × 0.9⌉ = 81  →  q̂ = 81st smallest of 88
```

That works, but `q̂` will move noticeably depending on which patients land in calibration.

### Class-conditional coverage isn't feasible for Indeterminate

One global `q̂` gives **marginal** coverage: "90% overall" can hold while the rare class is
systematically under-covered. The usual fix is **Mondrian (class-conditional) conformal**, with a
separate `q̂` per class.

Indeterminate is 11.0% of the data → ~9 calibration patients:

```
rank = ⌈10 × 0.9⌉ = 9  →  q̂ = the LARGEST of 9 scores
```

Every Indeterminate set would contain all three classes. The minimum calibration size for
α = 0.10 is `1/α − 1 = 9`, and we'd be right at that floor.

**Realistic options:**

1. Report **marginal** coverage only, and state that Indeterminate is likely under-covered.
2. **CV+ / cross-conformal**: rotate the calibration role across folds to pool more
   calibration data.
3. Relax α for the rare class (e.g. 0.20).

Whichever we choose, any "90% coverage" in the paper must say **marginal or conditional**.

---

## Fitting it into the existing CV

Everything currently runs on `StratifiedGroupKFold` by patient. Conformal needs a **nested**
split: inside each training fold, carve out a calibration slice **by patient** before fitting.

- **Split conformal** — one calibration set; simple; that data isn't used for training.
- **CV+ / cross-conformal** — uses all data; slightly looser guarantee. Probably right at this
  sample size.

Planned library: `mapie` (APS supported).
