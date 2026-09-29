# Explainability Framework Proposal — Formula-Coordinate Explanations (FCX)

*Draft, 30 Sep 2026. Name is a placeholder.*

> **Status 30 Sep:** built and evaluated the same day — see [`fcx.md`](fcx.md).

**Goal (from our guide):** our own explainability method that explains **our models'
predictions** — the subtype classifier, the conformal sets and the transition-risk model. Not
off-the-shelf SHAP, and not explanations of the data.

**Core idea.** Our label is a *known formula* over hidden exam items. The subtype is fixed by
where the tremor-to-gait log-ratio ℓ = log T − log P falls against two cutoffs: log 0.90 and
log 1.15. FCX explains every model **in that coordinate system**, measured in the clinic's own
units. Because we hold the true T and P, every explanation can be **scored against ground
truth**.

---

## 1. The three components — one coordinate system, three models

### C1 — Implied-Exam Explanation (Model 1: subtype classifier)

- From the 26 cheap features, estimate the classifier's **implied** tremor score T̂ and gait
  score P̂. Fidelity check: thresholding ℓ̂ = log T̂ − log P̂ must reproduce the classifier's
  decisions.
- Explanation: *"The model behaves as if this patient's gait score is 0.9 and tremor is 0.4
  → ratio 0.44, well past the 0.90 PIGD cutoff."*
- Each feature's contribution is split into a **tremor channel** and a **gait channel**, in
  units of *distance moved toward a cutoff*. This is exact, because ℓ̂ is additive in the two
  channels.

### C2 — Cutoff-straddle explanation (Model 2: conformal set)

- A set is `{TD, PIGD}` because the uncertainty interval on ℓ̂ **straddles a cutoff**.
- The explanation names **which channel's** uncertainty causes the straddle: *"uncertain
  because tremor can't be estimated from the questionnaire; gait alone points to PIGD"*.
- It also says how far the interval would have to shrink to resolve.

### C3 — Proximity–Drift–Noise (Model 3: transition risk)

A label flip in ℓ-coordinates can only come from three sources:

| Source | Meaning |
|---|---|
| **proximity** | the implied ratio sits close to a cutoff |
| **drift** | the ratio is actually moving over 12 months |
| **noise** | ordinary exam variability, including OFF/ON state |

- Fit a small first-passage surrogate to Model 3 and split each predicted risk exactly into
  those three terms, using Shapley over the 3 players.
- Output: *"risk 0.41 = mostly proximity (1 gait point from the cutoff) plus exam noise (ON
  state); little drift → re-examine OFF-state before re-subtyping."*

**Why this is one framework and not three tricks:** every explanation is expressed in ℓ, the
formula's own decision variable, with the same tremor/gait channel split and the same
cutoffs. The three models are explained consistently.

---

## 2. Feasibility — tested today

`notebooks/iee_fidelity.py`, binary TD vs PIGD, patient-grouped CV:

| Check | Result | Meaning |
|---|---|---|
| Implied-exam decision agrees with the classifier | **89.4%** | C1 is a faithful surrogate of Model 1 |
| Rank correlation, classifier P(PIGD) vs implied −ℓ̂ | **0.92** | it tracks the classifier's confidence, not only its label |
| Implied-exam balanced accuracy vs truth | 0.686 (classifier: 0.686) | no accuracy lost by explaining this way |
| Classifier tracks implied **gait** / **tremor** | **+0.80** / −0.58 | the model decides mostly from gait cues |
| Implied vs **true** gait score | r = **0.57** | the model's gait belief is partly right |
| Implied vs **true** tremor score | r = **0.25** | the model is **nearly blind to tremor** |

**This is already a result our method produces that SHAP can't:** *"The classifier separates
TD from PIGD almost entirely through gait-related cues. It effectively can't see tremor from
questionnaire data, so treat its TD calls with caution."* The finding is verified against
true scores the model never saw.

---

## 3. Prior art and positioning (adversarial search, 30 Sep)

| Closest work | What it does | How FCX differs |
|---|---|---|
| **Post-hoc CBM** — Yuksekgonul et al., ICLR 2023 ([2205.15480](https://arxiv.org/abs/2205.15480)) | turns a trained model into a concept model with a **learned** head | FCX uses a **fixed published clinical formula** as the head, on an unmodified XGBoost |
| **TRACE (glioblastoma CBM)** — Tarek et al., [2606.30313](https://arxiv.org/abs/2606.30313), Jun 2026 | trained CBM; deterministic RANO nodes; learned final head | not post-hoc; no conformal; no channel split. *Also a name clash with our project name.* |
| **COCOCO** — Bortolotti et al., [2605.18202](https://arxiv.org/abs/2605.18202) | conformal sets over concepts under known discrete logic | C2 is a continuous ratio straddling cutoffs, with channel blame |
| **Koh et al. CBM**, ICML 2020 | concept accuracy against real ground-truth concepts | so "verifiable against ground truth" alone is **not** new |
| **Shapley Flow / G-DeepSHAP** | attribution through composed functions | the channel maths is a special case; the clinical-cutoff units are the new part |
| **Uncertainty attribution via conformal** — [2505.13118](https://arxiv.org/abs/2505.13118) | Shapley on conformal interval width | feature-level only; no formula, no cutoffs |
| **Threshold regression** — Lee & Whitmore | first-passage models as predictors | C3 uses first-passage structure to **explain** a black box |

**Defensibly new, as a combination (novelty agent confidence ~0.65):**

1. Explaining an unmodified black box **in the coordinates of a fixed clinical scoring rule**
   (C1).
2. Conformal ambiguity explained as an **implied-score interval straddling a cutoff**, with
   channel blame (C2).
3. Transition risk decomposed into **proximity / drift / noise** in formula coordinates (C3).
   The design agent rated this component 7/10.
4. A **real-data evaluation protocol** with three separate scores (below). General "ground-truth
   concepts" work exists, so position this as a protocol, not as a first.

**Two objections reviewers will raise — handle them up front:**

- **Identifiability.** The formula only uses the ratio, and only through three zones. So "the
  classifier's output" alone can't pin down T̂ and P̂ separately. FCX therefore uses the true
  items as concept supervision, with fidelity *measured* rather than assumed. State this as a
  formal result.
- **Fidelity ≠ correctness.** A faithful explanation of a wrong model *should* disagree with
  the truth. Report three separate numbers:
  - **fidelity** — does the explanation reproduce the model?
  - **alignment** — do the implied scores match the true ones?
  - **error attribution** — when the model is wrong, does FCX blame the channel that is
    actually wrong?

---

## 4. Evaluation protocol

| Metric | Component | Baselines |
|---|---|---|
| Fidelity: agreement / R² with the model's output | C1, C3 | global surrogate tree, LIME |
| Alignment: corr(implied, true) for T, P, ℓ | C1 | — |
| **Error attribution**: when the model errs, AUROC of blamed channel = truly wrong channel | C1 | grouped SHAP (tremor-ish vs gait-ish features) |
| Straddle explanation vs actual set size; coverage by blamed channel | C2 | SHAP on conformal p-values (Johansson 2025) |
| Noise-vs-drift flip identification (ground truth from smoothed true ℓ) | C3 | grouped TreeSHAP, DeltaSHAP |
| Stability (bootstrap ICC) | all | SHAP |

---

## 5. Plan

| Week | Work |
|---|---|
| 1 | C1 fully: channel attribution in cutoff units, fidelity / alignment / error-attribution metrics, SHAP baseline |
| 2 | C2 on the fixed LAC conformal; straddle explanation; coverage stratified by channel |
| 3 | C3: smoothed true ℓ, drift and noise heads, first-passage surrogate, Shapley split, noise-vs-drift evaluation |
| 4 | Identifiability note, write-up, figures; rename the project |

**Depends on:** Rutu's fixes (APS, out-of-fold transition features) and a decision on exam
medication state.
