# Flip Anatomy — Method, Results and Verdict (30 Sep 2026)

**Idea.** Explain each observed subtype transition (visit *t* → next visit) by its cause:

1. a change in the exam's **medication state**
2. **threshold wobble** — measurement noise near a cutoff
3. **genuine change**, split into a tremor path and a gait path

We can check causes because we hold the 16 items, the formula and the exam state. The pitch:
explainability usually explains a *model's predictions*; this explains the *label's own
changes*.

**Result in one line:** the medication-state component **validates**; the noise-vs-genuine
split **does not**. The negative result is itself informative.

Code: `notebooks/flip_anatomy.py`. Per-flip output: `data/interim/flip_anatomy.csv`
(git-ignored).

---

## 1. Method

For each consecutive labelled visit pair with a label change (**1,383 flips** out of 4,818
pairs, 28.7%), one primary cause is assigned, in this order:

| Cause | Rule |
|---|---|
| **Exam-state artefact (twin-confirmed)** | The Part III exam was OFF at one visit and ON at the other, **and** PPMI recorded a same-day exam at the next visit in the *same* state as visit *t*. Recomputing the next label from that twin exam **removes the flip**. The counterfactual is observed, not modelled. |
| **Exam-state change (unconfirmed)** | State changed but no twin exam exists to check |
| **Threshold wobble** | Not state-related, and the next label is reachable from visit *t*'s items by **≤ 1 item point** ("points-to-flip") |
| **Genuine — tremor / gait path** | Everything else. Exact decomposition Δ log r = Δ log T − Δ log P; the path with the larger share wins |

---

## 2. What the decomposition says

| Primary cause | Share of flips | Reverts at the following visit |
|---|---|---|
| Threshold wobble | **42.7%** | 28.2% |
| Exam-state change (unconfirmed) | 19.7% | 39.1% |
| Genuine — gait path | 17.1% | 40.8% |
| Genuine — tremor path | 16.5% | 46.4% |
| Exam-state artefact (twin-confirmed) | 3.9% | 47.8% |

Other facts:

- **54%** of flips needed only **one** item point to happen, and 79% needed two or fewer.
- The exam state changed between the two visits in **34.3%** of flips. Where a twin exam
  existed to check (201 flips), matching the state removed the flip in **27%** of cases. So most
  state-change flips are **not** purely medication artefacts.
- **Measurement noise is large relative to change.** Use baseline → 3 months in the same exam
  state as a test-retest proxy, since little genuine progression is expected in 3 months of
  early PD. There, median |Δ log r| is **0.47**; over 12 months it's **0.74**. About two-thirds
  of a year's ratio change is the size of 3-month noise. **69%** of TD ↔ PIGD flips fall inside
  the 3-month noise envelope.

---

## 3. Validation — the honest part

A decomposition is only useful if the causes behave differently. We ran three independent
checks. None of them uses the formula items to judge the formula items.

### 3.1 Artefacts should revert more often — ❌ failed

Wobble flips revert **least** (28%), and "genuine" flips revert **most** (41–46%). The
magnitude-based split points the wrong way.

### 3.2 Genuine gait-path flips should predict clinical worsening — ❌ failed

Change measured from the **post-flip** visit to ~24 months later, so the flip itself isn't
counted:

| TD → PIGD flip cause | Δ Hoehn & Yahr | Δ MoCA |
|---|---|---|
| genuine — gait path | +0.16 (n = 56) | −0.94 (n = 33) |
| genuine — tremor path | +0.34 (n = 59) | −0.50 (n = 34) |
| threshold wobble | +0.21 (n = 33) | −1.64 (n = 14) |
| exam-state (any) | +0.33 (n = 66) | +1.10 (n = 10) |
| TD stays TD (reference) | +0.20 (n = 478) | +0.01 (n = 259) |

No cause separates cleanly from the reference, and the samples are small.

> An earlier version measured H&Y change from visit *t* instead. That showed a strong gait-path
> signal (+0.30), but it was **mechanical**: a gait-path flip raises postural stability, and
> H&Y stage 3 is *defined* by postural instability. That's why the table measures from the
> post-flip visit.

### 3.3 Persistence — ✅ validates only the medication-state component

What share of TD → PIGD flips are **sustained** (still PIGD two visits later)? Overall **24%**.

| Cause | Sustained |
|---|---|
| Exam-state artefact (twin-confirmed) | **0%** (n = 4) |
| Exam-state change (unconfirmed) | **17%** (n = 101) |
| Genuine — tremor path | 24% |
| Genuine — gait path | 26% |
| **Threshold wobble** | **44%** — the *most* persistent |

Sustained TD → PIGD flips do show slightly worse outcomes than transient ones (Δ H&Y +0.26 vs
+0.14; reference +0.20). That's directionally right, but weak.

---

## 4. Why the noise-vs-genuine split fails

**Slow genuine progression and measurement noise look the same at a single step.** A patient
drifting toward PIGD first crosses the cutoff by a small margin, which is exactly what we
labelled "wobble". That drift then *persists*. Pure noise also crosses by a small margin, but
it reverts.

Magnitude can't tell them apart; only **what happens next** can. Large single-step jumps are
often tremor, which is intermittent, so they revert too.

**So the explanatory question "was this flip real?" can't be answered from two visits.** It
needs a trajectory: at least one more visit, or a model of the ratio over time with an explicit
noise term.

---

## 5. What survives

| Claim | Status |
|---|---|
| Per-transition **exam-state artefact detection** using same-day twin exams as an observed counterfactual | ✅ validates (state-related flips are the least persistent). Closest prior art is Luo et al. 2019 (*Parkinsonism Relat Disord*), which showed in aggregate that OFF/ON changes the subtype. Per-transition counterfactual use appears new — check doi:10.1002/mdc3.70740 first. |
| **Measurement noise dominates visit-to-visit change** (3-month noise ≈ ⅔ of 12-month change; 69% of TD↔PIGD flips inside the noise envelope) | ✅ robust finding. It supports Ren et al. 2021 (38% unstable at 1 month) with a quantitative decomposition. |
| Magnitude-based **noise vs genuine** classification of single flips | ❌ doesn't validate — report it as a negative result |
| Tremor-path / gait-path decomposition | exact and easy, but on its own it doesn't predict anything |

---

## 6. What this means for the project

1. **Per-visit "transition risk" is mostly predicting noise.** That explains why the transition
   model plateaus at AUC ~0.65, while fragility alone reaches 0.77.
2. **Better target:** a **sustained** transition (the new label holds for ≥ 2 more visits), or
   the smoothed trajectory of log r with an explicit noise term.
3. **Label hygiene:** handle the exam's medication state before modelling — OFF-only labels,
   or state as a covariate. It's the one component that clearly behaves like an artefact.
4. **Novelty positioning.** Honestly, flip anatomy as first framed isn't a strong standalone
   contribution. Two stronger options:
   - **A.** A state-space model of the log-ratio with an explicit noise term, which gives each
     patient a *probability that their subtype has genuinely changed*. The explanation would
     be: "given the noise we measured, a change this size, sustained this long, is 80% likely
     to be real". This directly uses the finding above.
   - **B.** Make **medication-state-aware relabelling** the contribution: twin-exam
     counterfactuals plus a model that's invariant to exam state. It's narrower, but it
     validates.

---

## 7. Prior art found (novelty search, 30 Sep)

- **Luo L et al.**, *Motor phenotype classification in moderate to advanced PD in BioFIND
  study*, Parkinsonism Relat Disord 2019 — OFF/ON changes subtype; tremor improvement drives it.
  **Closest.**
- **Simuni T et al.**, *How stable are PD subtypes in de novo patients*, Parkinsonism Relat
  Disord 2016 — shift "not affected by dopaminergic treatment" at cohort level (p = 0.59).
- **Eisinger RS et al.**, *Motor subtype changes in early PD*, Parkinsonism Relat Disord 2017 —
  45% hold ≥ 2 subtypes. [PMC5842811](https://pmc.ncbi.nlm.nih.gov/articles/PMC5842811)
- **Ren J et al.**, *Consistency and Stability of Motor Subtype Classifications in de novo PD*,
  Front Neurosci 2021 — 38% unstable after 1 month.
  [PMC7957002](https://pmc.ncbi.nlm.nih.gov/articles/PMC7957002)
- **Kohat AK et al.**, *Stability of MDS-UPDRS Motor Subtypes Over Three Years*, Front Neurol
  2021 — higher ratio predicts stability.
- **Lee JW et al.**, *Alteration of TD and PIGD Subtypes During Progression*, Front Neurol 2019.
- **Kotagal V**, *Is PIGD a legitimate motor subtype?*, Ann Clin Transl Neurol 2016.
  [PMC4892002](https://pmc.ncbi.nlm.nih.gov/articles/PMC4892002)
- **Analogues outside PD:** CKD stage reclassification on repeat testing
  ([PMC6035156](https://pmc.ncbi.nlm.nih.gov/articles/PMC6035156)); MCI "false reverters" after
  practice-effect adjustment (Alzheimer's Res Ther 2019, doi:10.1186/s13195-019-0480-5).
- **Not yet checked:** doi:10.1002/mdc3.70740 (a snippet suggests it compares PPMI OFF vs ON
  labels directly); the full text of Luo 2019.

The novelty agent's verdict on the full combination was "likely novel (~70%)". That was
**before** the validation failed. What's left that's novel is narrower (§5).
