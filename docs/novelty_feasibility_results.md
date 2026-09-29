# Explainability Novelty — Feasibility Results (30 Sep 2026)

This follows `explainability_novelty_research.md`. It covers three things: a novelty search on
our "formula-aware conformal" idea, an independent ideation agent, and quick experiments on the
real data. Script: `notebooks/dia_feasibility.py`. All numbers are from the **corrected**
dataset (see §0).

---

## 0. Two data bugs found and fixed first

| Bug | Effect | Fix |
|---|---|---|
| PPMI codes *unable to rate* in Part III as **101** (not only the string `UR`) | 101 was averaged into the tremor / PIGD scores as a real score. **104 labelled visits (77 patients) were wrong, 98 of them forced to PIGD.** The "0 mismatches" check missed it: it recomputed from the same contaminated scores | 101 → NaN in the label items and `NHY`; those visits become unlabelled |
| SCOPA-AUT codes *not applicable* as **9** | 24,513 raw cells each added 9 points of fake autonomic burden to `SCOPA_AUT_TOTAL` (a feature) | 9 → NaN before the item sum |

| | Before | After |
|---|---|---|
| Labelled visits | 5,742 | **5,638** |
| Class balance | 52.6 / 36.6 / 10.8 | **53.5 / 35.5 / 11.0** |
| Transition pairs · flip rate | 4,918 · 29.1% | **4,818 · 28.7%** |
| Patients ever changing label | 80.9% | **78.4%** |
| TD vs PIGD balanced acc (logistic / HGB) | 0.696 / 0.690 | **0.694 / 0.689** |
| Sporadic first visit (logistic / HGB) | 0.584 / 0.530 | **0.553 / 0.530** |

The headline results barely move. Two new contract tests guard both codes.
**Rutu's conformal, transition, XGBoost and ablation outputs were built on the buggy data and
need re-running** on a machine with `xgboost`:
`make conformal transition xgboost ablation`.

---

## 1. "Formula-aware conformal" — ❌ already published

The idea was to put the conformal guarantee on the label's inputs (the item sums), then push
the region through the known Stebbins formula to get a label set. **This is published:**

- **Bortolotti et al. 2026**, *Concise and Logically Consistent Conformal Sets for
  Neuro-Symbolic Concept-Based Models* — [arXiv 2605.18202](https://arxiv.org/abs/2605.18202).
  "Concepts plus Deduction" is exactly this, with the coverage-inheritance proof. They treat it
  as a known baseline.
- **Ramalingam, Park, Bastani 2024**, *Uncertainty Quantification for Neurosymbolic Programs via
  Compositional Conformal Prediction* — [arXiv 2405.15912](https://arxiv.org/abs/2405.15912).
  Conformal sets pushed through any known program.
- **Braun et al.** ([2507.20941](https://arxiv.org/abs/2507.20941)) already handles refining
  the region as outputs are revealed. **COINS** ([2609.07112](https://arxiv.org/abs/2609.07112))
  handles sequential acquisition with validity at every stage.

The pure worst-case certificate (unrevealed items anywhere in 0–4) also **failed on the data**:
even a per-patient oracle needs a median of **16 of 16** items to certify the label.

**Verdict:** dropped as a methods contribution.

---

## 2. What the experiments actually found

### 2.1 Four in ten labels are one scoring point from a different label

Model a repeat full exam as rater noise: each item moves ±1 with probability *q*.

| | |
|---|---|
| Visits where **one** ±1 item change flips the label | **39.5%** |
| Chance a repeat exam gives a different label (fragility ρ ≥ 0.25), q = 0.2 | **30.8%** of visits |
| ρ ≥ 0.5 — a repeat exam is more likely than not to disagree | 10.3% |

### 2.2 Most subtype "transitions" are explained by fragility, not progression

| | Flip at next visit |
|---|---|
| One-point-fragile visits | **48.6%** |
| Robust visits | **15.0%** |
| Base rate | 28.7% |

**Fragility alone predicts the next-visit flip with AUROC 0.77.** Rutu's 40-feature transition
model scores **~0.65**. A single number describing how close today's scores sit to the
threshold beats the whole trained model.

**Interpretation:** a large share of what we've been calling "transition risk" is the label
wobbling across a threshold under ordinary rater noise. It isn't the disease changing subtype.
That matters clinically, because a "flip" of a fragile patient probably isn't a real event. It
also explains why the transition model plateaus.

> ρ is computed from the banned items, so it's an **oracle / explanatory** quantity, not a
> deployable feature.

### 2.3 Medication state flips a quarter of labels

In our cohort there are **2,116 same-day OFF / ON exam pairs** (400 patients). **25.6% get a
different label** depending only on whether the patient was on medication:

| OFF \ ON | IND | PIGD | TD |
|---|---|---|---|
| **IND** | 84 | 113 | 34 |
| **PIGD** | 40 | **776** | 31 |
| **TD** | 143 | **180** | **715** |

TD patients often read as PIGD when ON, as expected, because medication suppresses tremor.

Our "prefer OFF" rule only helps when an OFF exam exists. **Of the 5,638 labelled visits, 1,730
use an ON exam and 1,331 have unknown state.** Over time, the flip rate rises when the exam
state changes between visits: **OFF→ON 35.9%** vs **OFF→OFF 28.2%**, and those OFF→ON flips are
mostly TD→PIGD.

This is a **label-validity issue** to disclose. The options are to label OFF-state exams only,
or to stratify by state.

### 2.4 Conformal: the 90% guarantee hides a fragile subgroup

LAC on binary TD vs PIGD: **90.1%** coverage overall, **45.2%** singleton sets.

| Fragility tertile | Ambiguous sets | Coverage |
|---|---|---|
| low (ρ ≈ 0.00) | 47.0% | **94.2%** |
| mid | 58.1% | 91.3% |
| high (ρ ≈ 0.32) | 59.2% | **84.8%** |

- Among **confident** (single-label) calls, accuracy is **82.5%** for robust patients and
  **60.8%** for fragile ones.
- **76%** of ambiguous sets are *informational*: the patient isn't fragile, so a full exam would
  settle it. That supports the triage "do the exam" message.

### 2.5 The idea that failed

The ideation agent's top idea was **Definitional vs Informational Ambiguity**: tell the
clinician whether an abstention is irreducible (label fragility) or fixable (missing
information). It needs fragility to be predictable from the cheap features. **It isn't:
AUROC 0.546, near chance.** At deployment we can't tell which kind of ambiguity a patient has
without doing the exam. As a deployable explanation it fails. As an *analysis of what the model
is doing*, it's what produced §2.2–2.4.

---

## 3. What's left that could be genuinely new

### Candidate — "Flip anatomy": explaining subtype transitions

> **Update 30 Sep:** implemented and tested — see [`flip_anatomy.md`](flip_anatomy.md). The
> medication-state component validates; the noise-vs-genuine split does not.

Explain **each observed subtype transition** by decomposing it into causes that can be checked
exactly, because we hold the items, the formula and the exam's medication state:

1. **Threshold fragility.** Would the flip have happened under rater noise alone? Estimate from
   ρ at both visits.
2. **Exam-state artefact.** Did the medication state of the exam change (OFF↔ON)?
3. **Genuine change.** Item change beyond noise, split exactly into a **tremor path** and a
   **gait path**, since log ratio = log T − log P.

Then build a transition-risk model that predicts **genuine** flips only, and report how much of
the apparent 28.7% flip rate is artefact.

**Why this could be new:** we found no PD paper that decomposes subtype transitions into
artefactual vs genuine components. The explainability literature explains *predictions*;
this explains the **target's own changes**, using the known label-generating formula as ground
truth. It's also a direct explanation of why transition models plateau.

**Evidence it's worth doing:** §2.2 (fragility AUROC 0.77 vs 0.65) and §2.3 (25.6% OFF/ON
disagreement; OFF→ON drives TD→PIGD).

**Before claiming novelty:**

- [ ] novelty search: "subtype instability" + "measurement error" / "threshold" in PD
  (Simuni 2016 found starting therapy didn't explain changes — a related but different
  question); "label noise near decision boundary" + "longitudinal"; "flip decomposition"
- [ ] a published MDS-UPDRS inter-rater reliability value to set the noise level *q*, instead of
  our 0.1–0.3 sweep
- [ ] check the tremor/gait path split against the Stutz et al. line on ambiguous ground truth
  ([2307.09302](https://arxiv.org/abs/2307.09302))

**Confidence it's publishable as new:** moderate. It's untested for novelty, but the empirical
findings are strong and specific to this setting.

### Secondary — medication twin-label audit

The ideation agent's second idea. The 2,116 same-day OFF/ON pairs give an **observed
counterfactual label**. Train on each and compare attributions, to see whether LEDD's
importance reflects disease or exam-state artefact. Rated 7/10 by the agent after its search.
Not yet run.

---

## 4. Other notes

- **Name clash:** a June 2026 concept-bottleneck paper is already called **TRACE**
  ([arXiv 2606.30313](https://arxiv.org/abs/2606.30313)). Consider renaming before submission.
- The ideation agent's full list of 9 ideas, with novelty scores, is in the session record.
  Trait-vs-state attribution (5/10) is the safe fallback.
