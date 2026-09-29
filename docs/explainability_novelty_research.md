# Explainability Novelty — Research and Proposal

**Question from our guide:** build an explainability algorithm or framework as the novel
contribution of TRACE-PD.

**Short answer:** the most defensible novelty is **Set-Resolving Acquisition**. When a patient's
conformal set is ambiguous (`{TD, PIGD}`), the method identifies the **smallest group of
specialist motor-exam items** that would resolve it. That turns *"do the full 30-minute
exam"* into *"check gait and postural stability — 2 items"*. Two companion explanations make
it a framework, not a single trick. Every sub-idea has close prior art, so the novelty is the
specific combination. Our confidence that it is unpublished is **medium (~60–65%)**.

Research date: 30 Sep 2026. Sources were checked by reading the arXiv / PMLR / publisher pages.
Some are abstract-only; these are flagged.

---

## 1. What already exists — and rules out the obvious ideas

### 1.1 "SHAP with the previous visit as baseline" — ❌ not novel

This was the idea in our own design doc (Δφ = φ(t) − φ(t−1)). **It is published.**

| Prior art | What it does |
|---|---|
| **DeltaSHAP** — Kim, Mun, Hahn, Yang. ICML 2025 Workshop on Actionable Interpretability. [arXiv 2507.02342](https://arxiv.org/abs/2507.02342) · [code](https://github.com/AITRICS/DeltaSHAP) | Shapley attribution of the **change** in predicted probability between consecutive time steps. Missing values are forward-filled from the previous step, so for a per-visit model it reduces exactly to f(xₜ) − f(xₜ₋₁). ICU data (MIMIC-III, PhysioNet). |
| **Delta-XAI** — Kim et al. ICLR 2026. [arXiv 2511.23036](https://arxiv.org/abs/2511.23036) · [code](https://github.com/AITRICS/Delta-XAI) | Wraps 14 attribution methods to explain prediction changes, and adds an evaluation suite. |
| **Baseline Shapley (BShap)** — Sundararajan & Najmi, ICML 2020. [arXiv 1908.08474](https://arxiv.org/abs/1908.08474) | Shapley values against any single reference point. Using the previous visit as reference is one instance. |

**Consequence:** cite DeltaSHAP as a method we *use*. Don't claim it.

### 1.2 Explaining conformal sets — exists, but only for a single input

| Prior art | What it does |
|---|---|
| **Explaining Set-Valued Predictions: SHAP Analysis for Conformal Classification** — Johansson, Sönströd, Maalej. COPA 2025, PMLR 266. [link](https://proceedings.mlr.press/v266/johansson25a.html) | SHAP on each class's conformal p-value. Static; set-level explanation is only outlined. |
| **Counterfactual Explanations for Conformal Prediction Sets** — Maalej, Sönströd, Johansson. COPA 2025, PMLR 266. [link](https://proceedings.mlr.press/v266/maalej25a.html) | Minimal change to *observed* features that changes the set. Hypothetical perturbations, static. |
| **ECCCo** — Altmeyer et al. AAAI 2024. [arXiv 2312.10648](https://arxiv.org/abs/2312.10648) | Counterfactuals with conformal set size as a penalty. Gradient-based, so it doesn't fit XGBoost. |
| **CONFEX** — Bilkhoo et al. [arXiv 2510.19754](https://arxiv.org/abs/2510.19754) | Counterfactuals with local conformal guarantees via mixed-integer programming. |
| **Conformal Selection of Counterfactual Explanations** — Johansson et al. COPA 2026, PMLR 329. [link](https://proceedings.mlr.press/v329/johansson26a.html) | Chooses the counterfactual with the highest conformal p-value. |

> ⚠️ The Johansson / Maalej / Sönströd group publishes in this exact area every year. Any
> claim we make must be checked against their newest papers first.

### 1.3 Attributing *uncertainty* to features — exists

- **Watson et al.**, *Explaining Predictive Uncertainty with Information Theoretic Shapley
  Values*, NeurIPS 2023 — [arXiv 2306.05724](https://arxiv.org/abs/2306.05724). Shapley values
  of predictive entropy. The authors themselves list feature acquisition as an application.
- **CLUE** — Antorán et al., ICLR 2021 — [arXiv 2006.06848](https://arxiv.org/abs/2006.06848).
  Counterfactuals that reduce BNN uncertainty. Needs a differentiable generative model.
- **Wood et al.** 2024, entropy-based model-agnostic importance —
  [arXiv 2310.12842](https://arxiv.org/abs/2310.12842).
- **Calibrated / Ensured Explanations** — Löfström et al. —
  [arXiv 2305.02305](https://arxiv.org/abs/2305.02305),
  [2410.05479](https://arxiv.org/abs/2410.05479). Explanations for reducing epistemic
  uncertainty. PyPI library.

### 1.4 "What to measure next" using conformal sets — the criterion exists

| Prior art | What it does | Why it isn't ours |
|---|---|---|
| **Conformal Information Pursuit (C-IP)** — Chan, Ge, Dobriban, Hassani, Vidal. NeurIPS 2025. [arXiv 2507.03279](https://arxiv.org/abs/2507.03279) · [code](https://github.com/ryanchankh/ConformalInformationPursuit) | Chooses the next question by minimising expected conformal set size. Tested on 20 Questions and the MediQ doctor–patient dataset. | LLM question-asking, not a tabular clinical model or exam items |
| **ALMA** — Hoarau et al. [arXiv 2501.18268](https://arxiv.org/abs/2501.18268) | Acquires the next modality while the conformal set isn't a singleton; MIMIC-IV. | fixed cost order — never chooses *which* test |
| **SepsisLab** — Zhang et al. CHI 2024. [arXiv 2309.12368](https://arxiv.org/abs/2309.12368) | Recommends lab tests that reduce uncertainty; 6 clinicians. | no conformal guarantee |
| **Dynamic feature selection (CMI)** — Covert et al. ICML 2023. [arXiv 2301.00557](https://arxiv.org/abs/2301.00557) · [code](https://github.com/iancovert/dynamic-selection) | Greedy mutual-information feature acquisition; supports groups. | entropy criterion — our natural **baseline** |
| **RouteCert** — Baghi. [arXiv 2608.15520](https://arxiv.org/abs/2608.15520) (Aug 2026) | Proves coverage **breaks** when an adaptive policy chooses which features get observed; gives valid fixes. | a **constraint we must satisfy** |

### 1.5 Parkinson's explainability — the gap is wide

PD papers with SHAP found: SCOPE-PD ([2601.22516](https://arxiv.org/abs/2601.22516)), STEP-PD
([2604.17611](https://arxiv.org/abs/2604.17611)), Yan et al. 2025 (*Front Aging Neurosci*),
Xu et al. 2025 (*Front Neurol*), [2003.09466](https://arxiv.org/abs/2003.09466). **All are static,
per-visit SHAP.**

**Nothing found in PD that:**

1. explains **why a patient's subtype changed** between visits
2. explains a **transition-risk** model
3. explains **conformal prediction sets**, static or over time
4. separates **medication** effects from **symptom** changes
5. recommends **which exam item** to collect to settle the subtype

The closest clinical analogue is Sreenivasan et al. (npj Digit Med 2025, MS —
[PMC12022056](https://pmc.ncbi.nlm.nih.gov/articles/PMC12022056/),
[code](https://github.com/caramba-uu/MSP-tracker)). It tracks conformal p-values visit by visit
with per-visit SHAP, but never explains the transition itself. It's structurally the most
similar paper to TRACE-PD.

---

## 2. The proposal — a three-question framework

A clinician looking at an ambiguous result asks three questions. Each maps to one explanation
component:

| Clinician's question | Component | Novelty |
|---|---|---|
| **Q1. Why is the model unsure?** | Set attribution — Shapley on each class's conformal margin (Johansson et al. 2025, applied) | ❌ applied, not new |
| **Q2. What changed since last visit?** | **Set-change attribution** — attribute the *change in the conformal set* between visits (DeltaSHAP + set-level game) | 🟡 new combination, medium |
| **Q3. What should I measure to be sure?** | **Set-Resolving Acquisition** — the smallest specialist-exam item group that resolves this patient's set | 🟢 **headline contribution** |

### 2.1 Q3 — Set-Resolving Acquisition (the core novelty)

**Idea.** For a patient whose set is `{TD, PIGD}`, estimate for each **group** of banned motor
items the probability that measuring that group alone would collapse the set to a single
label. Then recommend the smallest group that does it.

**Why it's strong for *this* project specifically:**

- **It fits the clinical-need argument exactly.** `docs/clinical_need.md` establishes that the
  full exam takes ~30 minutes and a certified rater. Today our triage output says *"uncertain
  → do the full exam"*. This makes it *"uncertain → check these 2 items"*: a targeted
  2-minute exam instead of a 30-minute one.
- **It turns our leakage constraint into an asset.** The 16 banned items are banned as
  *inputs*, but we hold their true values for every patient. That means we can **reveal them
  and check** whether the recommendation actually resolved the case. That's an oracle
  evaluation most acquisition papers can't do, because they don't have the answer.
- **It changes the output from a passive explanation to an actionable one.** Lee & Chew (CSCW
  2023, [2308.04375](https://arxiv.org/abs/2308.04375)) found that actionable counterfactual-style
  explanations cut over-reliance on wrong AI outputs by 21% compared with feature-importance
  explanations alone.

**Why groups, not single items.** Three reasons:

1. **Coverage has to be recalibrated for every combination of observed features** (RouteCert).
   With ~88 calibration patients, that's only feasible for a handful of combinations, not 2¹⁶.
2. Groups are clinically natural: *rest tremor* (3.17a–e, 3.18), *action tremor* (3.15–3.16),
   *gait / freezing* (3.10–3.11), *postural stability* (3.12).
3. Owen / group Shapley values are the standard tool for grouped features
   (`shap.PartitionExplainer`; Jullum et al., [2106.12228](https://arxiv.org/abs/2106.12228)).

**Method sketch:**

1. **Masked-input model.** Train the subtype model on the 26 cheap features **plus** the
   banned item groups, with each group randomly masked during training. One model then handles
   any subset of revealed groups.
   - The label remains a function of those items, so a fully revealed patient is trivially
     solved. The value is in the **partial** reveal.
2. **Per-state calibration.** Calibrate a separate conformal threshold for each acquisition
   state: none revealed, each single group, each pair. This follows RouteCert's valid
   construction.
3. **Imputation model.** For an unrevealed group g, model p(g | 26 cheap features) from
   training patients. Use quantile or bootstrap draws.
4. **Expected resolution score** for each group g:

   ```
   R(g) = P_{g ~ p(g | x_cheap)} [ |C_{cheap ∪ g}(x)| = 1 ]
   ```

   the probability that revealing g yields a singleton set. Recommend argmax R(g),
   optionally weighted by the time each item takes to administer.

**Evaluation — quantitative, on PPMI, no new data needed.** All splits are by patient.

| Metric | Measures |
|---|---|
| **Resolution curve** | % of ambiguous patients resolved to a singleton vs number of groups revealed; area under it |
| **Oracle agreement** | top-1 agreement with the group that *actually* resolved each case, using the true values |
| **Singleton accuracy** | when the set collapses, is it the right label? |
| **Per-state coverage** | coverage checked separately in every acquisition state, to show the guarantee survives |
| **Stability** | Kendall's τ of group rankings across bootstrap runs and adjacent visits |

**Baselines:** random order · fixed clinical order · global SHAP ranking · greedy mutual
information (Covert et al.) · entropy-Shapley (Watson et al.).

**Headline result this can produce:** *"For X% of ambiguous patients, revealing one item
group resolves the subtype with Y% singleton accuracy and coverage maintained at 90%. Median
items needed: k of 16."*

### 2.2 Q2 — Set-change attribution

**Idea.** Explain *why the prediction set changed* between visit t−1 and t — for example,
`{TD, PIGD}` → `{PIGD}`. Build a pairwise Shapley game where each coalition takes its features
from visit t and the rest from visit t−1. The value function is each class's **margin** to the
conformal threshold, so attributions sum exactly to the change in margin. Report the discrete
set-level version as well.

**What's new vs DeltaSHAP:** DeltaSHAP explains a probability change; this explains a
**conformal set change**, i.e. a change in *confidence status*.

**Pitfalls, stated honestly:**

- **For LAC, the margin game reduces to DeltaSHAP on probabilities** plus a constant. The
  threshold q̂ is fixed after calibration. We should **prove this equivalence** and say so.
  The genuine difference appears with **APS** (rank-dependent, nonlinear) and with the
  discrete set-level game.
- Shapley values of a thresholded output exist and sum exactly, but they're discontinuous and
  concentrate on whichever feature tips the threshold. That's why we report both the discrete
  and the margin versions.
- Set flips near the threshold may be calibration noise. Add a **bootstrap-q̂ stability test**
  and only explain flips that survive it.
- Time-driven features (`YRS_SINCE_*`, visit count) will absorb attribution. Report them
  separately.

**Clinically meaningful add-on — treatment vs disease split.** A group Shapley decomposition
of the change into **{LEDD, N_CONMEDS, PDTRTMNT}** vs **symptom features** answers *"did the
subtype drift because of medication or because of the disease?"*. No PD paper found does this.
Tremor is suppressed by dopaminergic medication (README §4.7), yet Simuni et al. (2016, PPMI)
reported that subtype changes were **not** explained by starting therapy at the cohort level.
A per-patient decomposition tests that question patient by patient.

### 2.3 Tying it together — evaluate explanations by what they predict

- Use Q2's set-change attributions as features for the **transition-risk** model. Do they
  anticipate the next flip better than raw deltas?
- That gives a *functional* evaluation of the explanations — they predict something — rather
  than only a plausibility check.

---

## 3. What this is **not**, and the risks

| Risk | Mitigation |
|---|---|
| **Novelty could already be taken.** Very active area; confidence ~60–65% | Before claiming: Google Scholar "cited by" pass on **C-IP**, **DeltaSHAP**, **Johansson COPA 2025**, **Maalej COPA 2025** |
| **Trivial solution.** All 16 items together determine the label exactly | Novelty is the **minimum** acquisition; report the curve, not the endpoint |
| **Coverage breaks under adaptive acquisition** (RouteCert) | calibrate per acquisition state; restrict to groups |
| **Small n.** ~88 calibration patients | ≤ 5 groups, few acquisition states; CV+ calibration |
| **Imputation quality.** R(g) is only as good as p(g \| x_cheap) | report calibration of the imputation; compare to the oracle |
| **Current code bugs.** APS mismatch, in-sample transition features, validator no-op (review of Rutu's commit) | fix first — this work builds directly on them |

---

## 4. Recommended plan

| Step | Work | Output |
|---|---|---|
| 0 | Fix APS, out-of-fold probabilities for transition, orchestrator validator | a correct base |
| 1 | Novelty check on C-IP / DeltaSHAP / COPA citing papers | go / no-go |
| 2 | Define 4–5 banned item groups with a clinician's input | group definitions |
| 3 | Masked-input model + per-state calibration | `models/acquisition.py` |
| 4 | Imputation model + resolution score R(g) | recommendations |
| 5 | Oracle-reveal evaluation vs 5 baselines | the paper's main table and figure |
| 6 | Q2 set-change attribution + treatment/disease split | second contribution |
| 7 | Optional: 2–3 neurologists rank groups on ~20 cases | clinician agreement (Kendall's W) |

**Suggested title angle:** *"Which exam item would settle it? Conformal set-resolving
acquisition for Parkinson's motor subtyping from low-cost data."*

---

## 5. Sources

**Temporal / change explanation:** DeltaSHAP (2507.02342) · Delta-XAI (2511.23036) · BShap,
Sundararajan & Najmi (1908.08474) · The Explanation Game, Merrick & Taly (1909.08128) ·
TimeSHAP, Bento et al. KDD 2021 (2012.00073) · FIT, Tonekaboni et al. NeurIPS 2020
(2003.02821) · WinIT, ICLR 2023 (2107.14317) · Dynamask, ICML 2021.

**Conformal explanation:** Johansson et al. COPA 2025 · Maalej et al. COPA 2025 · Johansson et
al. COPA 2026 · ECCCo, AAAI 2024 (2312.10648) · CONFEX (2510.19754).

**Uncertainty attribution:** Watson et al. NeurIPS 2023 (2306.05724) · CLUE, ICLR 2021
(2006.06848) · Wood et al. 2024 (2310.12842) · Löfström et al. (2305.02305, 2410.05479) · Bley
et al. (2401.17441).

**Acquisition:** C-IP, NeurIPS 2025 (2507.03279) · ALMA (2501.18268) · RouteCert
(2608.15520) · Covert et al. ICML 2023 (2301.00557) · SepsisLab, CHI 2024 (2309.12368) ·
active feature acquisition survey (2502.11067).

**Group / concept:** groupShapley (2106.12228) · `shap.PartitionExplainer` (Owen values) ·
TabCBM, TMLR 2023.

**Evaluation:** ROAR (1806.10758) · OpenXAI (2206.11104) · Quantus, JMLR 2023 · Nauta et al.
ACM CSUR 2023 (2201.08164) · The Disagreement Problem (2202.01602).

**Clinical:** Sreenivasan et al. npj Digit Med 2025 · Tonekaboni et al. MLHC 2019 · Lee & Chew
CSCW 2023 (2308.04375) · Simuni et al. Parkinsonism Relat Disord 2016.

**PD XAI:** SCOPE-PD (2601.22516) · STEP-PD (2604.17611) · Yan et al. 2025 · Xu et al. 2025 ·
2003.09466.
