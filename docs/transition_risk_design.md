# Transition Risk — Design

**Status: target built and validated; model not trained.**

Transition risk is a **second supervised classifier**, not a rule. It estimates the probability
that a patient's subtype label **changes by their next visit**.

---

## How the `t+1` data is built

The table is **long format — one row per patient-visit**, not one row per patient. The next
visit isn't a separate table. It's the same patient's next row, pulled with a shift inside
each patient:

```python
df = df.sort_values(["PATNO", "VISIT_MONTH"])
df["NEXT_LABEL"]       = df.groupby("PATNO")["LABEL"].shift(-1)
df["NEXT_VISIT_MONTH"] = df.groupby("PATNO")["VISIT_MONTH"].shift(-1)

both = df["LABEL"].notna() & df["NEXT_LABEL"].notna()
df["LABEL_FLIPPED_NEXT"] = (df["LABEL"] != df["NEXT_LABEL"]).where(both)
```

The `groupby("PATNO")` is what stops leakage across patients. Without it, patient 3001's last
visit would get patient 3002's baseline as its "next".

One patient, as stored:

| PATNO | EVENT_ID | VISIT_MONTH | LABEL | NEXT_LABEL | NEXT_VISIT_MONTH | FLIPPED |
|---|---|---|---|---|---|---|
| 3001 | BL | 0 | TD | TD | 3 | 0 |
| 3001 | V01 | 3 | TD | TD | 6 | 0 |
| 3001 | V02 | 6 | TD | TD | 9 | 0 |

Every row is therefore both a **subtype** training example (its own X, its own label) and — if
it has a successor — a **transition** training example (its own X, `FLIPPED` as target).

### Where 4,918 pairs come from

```
5,742 labelled rows
 −824 labelled rows with no usable next label
=4,918 pairs
```

| Why no pair | Rows |
|---|---|
| patient's final visit — no next row | 223 |
| next visit exists but is unlabelled (not all 16 items completed) | 601 |

### Base rates

| Current label | Flip rate | Pairs |
|---|---|---|
| TD | 20.9% | 2,744 |
| PIGD | 27.1% | 1,629 |
| Indeterminate | **76.3%** | 545 |
| **Overall** | **29.1%** | **4,918** |

---

## Features

The **same permitted X** as the subtype model (all 19 banned columns still banned), plus
signals that are legitimately known at time *t*:

- deltas of permitted features since the previous visit
- months since baseline, visit index
- number of prior flips for this patient
- **outputs of the layers upstream:** predicted class probabilities, conformal set size, and
  the top-2 probability margin

That last group is why transition risk sits **downstream** of prediction, conformal and the
tracker in the architecture. A patient whose set has been size 2 for three visits, with a
narrowing TD–PIGD margin, is about to flip. That pattern only exists once there's history.

---

## Pitfalls

### Train on the *predicted* label, not the true one

At inference the true current label is unknown — only the model's prediction exists. Training
on the true label and serving on the predicted one is **train / serve skew**.

### `MONTHS_TO_NEXT_VISIT` is not a feature

It's only known once the next visit has happened. It's stored as a **target** column for that
reason.

### "Next visit" isn't a fixed horizon

Intervals vary: 3 months early in the study, 6 months later, 12 at annual visits. So
`LABEL_FLIPPED_NEXT` answers *"does it flip by the next appointment?"*, and that depends on
when the appointment happens to be.

**Recommended:** redefine the target at a **fixed horizon** — "flips within 12 months" — so
the number means the same thing for every patient. Alternatively, take the planned follow-up
interval as a user input at serving time.

### Much of the 29% is boundary noise

A large share of flips is Indeterminate churn: patients whose ratio sits in the 0.90–1.15 band
and crosses it back and forth. That's measurement wobble as much as progression.

Report flip risk **separately** for patients near the decision boundary and for those well
inside a class. It's also worth reporting TD ↔ PIGD flips separately from flips involving
Indeterminate. The ratio itself is Y-defining and can't be a feature, but the model's own
probability margin is allowed.

---

## Validation and output

- `StratifiedGroupKFold` by `PATNO`, same as everything else.
- Output: calibrated probability from a held-out fold → *"72% chance this patient's subtype
  label changes by the next visit."*

---

## Conformal vs transition risk — not the same question

| | Question | About |
|---|---|---|
| **Conformal** | How sure am I of the label *right now*? | the **model's** uncertainty |
| **Transition risk** | Will the true label *change* by next visit? | the **patient** changing |

They come apart in both directions:

| Set | Flip risk | Reading | Action |
|---|---|---|---|
| `{PIGD}` | 8% | confident and stable | plan around PIGD |
| `{PIGD}` | 61% | confident today, patient moving | plan around PIGD, shorten follow-up |
| `{TD, PIGD}` | 12% | model can't tell, patient stable | do the full exam once, settle it |
| `{TD, PIGD}` | 70% | worst case | commit to nothing, bring them back sooner |
