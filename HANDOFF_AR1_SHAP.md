# Handoff — AR(1)-SHAP explainability framework (1 Oct 2026)

**Read this first.** It covers everything done on 1 Oct 2026: what was built, how it works, every
result, what didn't work, and what's left. It's written so a teammate, or their Claude Code
session, can pick up with full context.

> **Update 10 Oct 2026:** ADNI (Alzheimer's) was brought in as a second cohort, to support the
> "framework" claim — AR(1)-SHAP working on only one disease is a script, not a framework.
> **Preprocessing is done** (§13): one long table, a dictionary, leakage policy, 12 contract
> tests, all passing. **`explain/ar1.py` is now generic (§14)**: feature roles (fixed / clock /
> changing) are detected statistically, not hard-coded, verified to reproduce PPMI's exact known
> roles and to work correctly on ADNI with zero ADNI-specific code. **The full ADNI run is done
> too (§15)**: a classifier trained (0.980 balanced accuracy, binary CN-vs-Dementia), all five
> AR(1)-SHAP tests run, and PPMI vs ADNI compared side by side — the core findings replicate
> almost number-for-number on a disease with a different label type, different features, and a
> much stronger classifier. **The "framework" claim now has two-cohort evidence**, not just the
> capability to run on a second cohort.

> **Update 11 Oct 2026: read §16 first.** The core AR(1)-SHAP idea turned out to be already
> published (DynSHAP, Sep 2026). §16 records the novelty re-check, the rejected directions, and the
> new "expected vs new" SHAP direction. §16.7 assesses that direction as workshop-level as it
> stands, not journal-ready, so the novelty question is still **open**.

> **For Claude Code:** read this file, then `CLAUDE.md` (repo rules), then `docs/ar1_shap.md`
> (method + results). For the ADNI cohort, read §13 of this file and
> `src/trace_pd/data/preprocess_adni.py`'s docstring (it documents every data decision the way
> `preprocess.py` does for PPMI). The source of truth for numbers is `reports/metrics/*` and
> `data/processed/adni_dx_preprocess_runlog.txt`. The older `HANDOFF_README.md` is **stale**; do
> not quote it.

---

## 0. One-paragraph summary

TRACE-PD predicts Parkinson's motor subtype (TD vs PIGD) from cheap clinical data on the PPMI
cohort. The models are weak (~0.69 balanced accuracy) and can't be pushed meaningfully higher
(§7). So, as our guide directed, **the paper's contribution is an explainability method, not
the model.** We built **AR(1)-SHAP**: a version of SHAP for repeat-visit data that fills hidden
features with values realistic *for this patient given their last visit*, instead of copying them
from a random other patient. It is implemented, tested (12 new tests, 51/51 pass) and evaluated
on 88 held-out patients. The core hypothesis holds. Standard SHAP builds unrealistic patients
(a classifier spots them 97.6% of the time); AR(1)-SHAP's are much more realistic (67.9%). The
credit given to "years since diagnosis" falls 86.8%. Two parts of the original story did **not**
hold (§6.4).

---

## 1. Where things are

| What | Path |
|---|---|
| **This handoff** | `HANDOFF_AR1_SHAP.md` |
| Method + full results write-up | `docs/ar1_shap.md` |
| Codebase guide (models, quality, data facts, pre-build plan) | `CODEBASE_README.md` |
| Visit-to-visit model | `src/trace_pd/explain/ar1.py` |
| The explainer (Shapley + fillers) | `src/trace_pd/explain/conditional_shapley.py` |
| Evaluation (tests T1–T5) | `src/trace_pd/evaluation/evaluate_ar1_shap.py` |
| Raw results | `reports/metrics/ar1_shap_results.txt` |
| Run log (appended, step id `ar1`) | `reports/metrics/experiment_log.csv` |
| Unit tests | `tests/test_ar1_shap.py` |
| Model ceiling check | `notebooks/model_ceiling_check.py` |
| Saved models + per-visit attributions | `models/ar1_shap_artifacts.pkl` (**git-ignored**; regenerate with `make ar1shap`) |
| Original idea | `AR1-SHAP_Technical_Briefing.pdf` (one folder **above** the repo, not committed) |

**Files changed on 1 Oct:** `Makefile` (new `ar1shap` target), `README.md` (stale §9.3, §9.5, §9.6
numbers synced; status table updated), `reports/metrics/experiment_log.csv` (new rows).

> ⚠️ **Git state:** as of writing, none of this is committed. If you're reading this on another
> machine and the files above are missing, ask for the push.

---

## 2. Project context you need

- **Data:** PPMI, 439 Parkinson's patients, 6,922 visits, 5,638 with a label. DUA-restricted:
  keep the repo private and never copy patient rows out (`CLAUDE.md`).
- **Label:** Stebbins (2013) formula on 16 exam items. Tremor score ÷ gait score; ≥1.15 TD,
  ≤0.90 PIGD, between = Indeterminate.
- **Features:** 26 cheap features (questionnaires, demographics, medication). The 16 label items
  plus 3 indirect ones are **banned** as inputs. Features are chosen by the `CHEAP_FEATURE` bucket
  in `data/processed/ppmi_tdpigd_dictionary.csv`, never by hand.
- **Scope decision:** **longitudinal patients only.** First-visit prediction is out of scope, so
  every explained visit has a previous visit.
- **Validation rules:** everything is split by patient (`StratifiedGroupKFold` on `PATNO`). A
  locked 20% of patients (88, `data/interim/locked_test_patients.csv`) is never used for fitting
  or selection.
- **Existing explainability:** FCX (`docs/fcx.md`), built 30 Sep. AR(1)-SHAP is separate from it
  for now.

### Model quality before this session (unchanged)

| Model | Score | Note |
|---|---|---|
| Binary TD vs PIGD (XGBoost / logistic) | 0.69 balanced acc; locked test 0.693 [0.647, 0.738] | 12 setups, all statistically tied |
| 3-class | ~0.46 | Indeterminate can't be separated |
| Transition (will the label flip?), cheap features | AUROC 0.53 | no signal once a leak was removed |
| Transition with the exam + history | AUROC 0.80 | needs the exam, so not deployable |

---

## 3. The idea in plain words

SHAP measures how much a feature matters by **hiding it** and seeing how the prediction changes.
To hide it, it must fill the gap with some value. Standard SHAP copies the value from a
**random other visit**, usually another patient's. With repeat-visit data that builds impossible
patients: the gait score goes 4 → 4 → 3 and the filler is a stranger's 0.

**AR(1)-SHAP fills the gap with what is realistic for this patient**, given what we can see today
and what they looked like at their last visit.

```
standard     v(S) = E[ f(x_S, X_hidden) ],  X_hidden ~ P(X_hidden)                      (briefing Eq. 1)
AR(1)-SHAP   v(S) = E[ f(x_S, X_hidden) ],  X_hidden ~ P(X_hidden | x_S, x_{t-1}, Δt)     (briefing Eq. 3)
```

**A consequence to state in the paper:** the starting point becomes "what this patient's last
visit predicts for today", so the contributions add up to `f(x_t) − E[f(X_t) | x_{t-1}]`.
AR(1)-SHAP answers *"why is today's prediction different from what we expected for this
patient?"*, not *"why is this patient different from the average patient?"*.

---

## 4. What was built, technically

### 4.1 `explain/ar1.py`: the visit-to-visit model

The briefing's model `X_t | X_{t-1} ~ N(A·X_{t-1} + b, Σ)` breaks on our features in four places.
Each fix:

| Problem | Fix | Code |
|---|---|---|
| `SEX`, `HANDED` never change; `AGE_AT_VISIT`, `YRS_SINCE_DIAGNOSIS`, `YRS_SINCE_SYMPTOM_ONSET` advance exactly with the calendar → singular Σ / nothing to sample | **Feature roles.** Fixed + clock features are *known* given the previous visit and the date; never sampled. Only the **21 changing** features get a model. Today's fixed/clock values enter it as known inputs (`z_t`). `roles=False` declares nothing and makes the model learn it (robustness check) | `feature_roles()`, `STATIC`, `CLOCK`, `TransitionModel(roles=...)` |
| 20 of the 21 changing features are 0–4 items, counts or yes/no; a Gaussian would sample impossible values | **Gaussian copula.** Rank → normal score (mid-rank for ties), model there, map back through the empirical quantile. Every sampled value exists in the training data; the round trip is exact | `CopulaTransform` |
| Visits are 3 / 6 / 9+ months apart; Eq. 2 has no gap term | **Gap-aware.** Mean includes `Δt × previous values` and `Δt` terms. Leftover spread estimated per gap bucket (≤4.5, ≤7.5, >7.5 months), sharing one correlation matrix: `Σ(Δt) = D(Δt)·R·D(Δt)` | `GAP_EDGES`, `_fit_cov`, `sigma_for` |
| Only 25% of visit pairs have all 26 features | **Carry-forward** of the last recorded value, up to 12 months, with a `STALE_` flag; older/absent → population mean. Missing targets are dropped from the likelihood exactly (Gaussian marginalisation), never imputed | `build_pairs`, `LOCF_MAX_MONTHS`, `gauss_loglik_rows` |
| ~1,000 parameters from ~5,000 pairs | **Ridge** per target (alpha from {0.3, 3, 30, 300} by patient-grouped 5-fold CV on squared error); **Σ shrinkage** toward its diagonal (λ from {0, .05, .15, .3, .5} by held-out log-likelihood); nearest-PSD repair | `ALPHAS`, `LAMBDAS`, `nearest_psd`, `shrink` |

Mean model, in copula space (`u` = scores of the 21 changing features):

```
u_t = b + A·u_{t-1} + Δt·(A'·u_{t-1}) + c·Δt + C·z_t + (stale flags) + ε,    ε ~ N(0, Σ(Δt))
```

Four variants are kept for comparison: `population` (no history), `scalar` (each feature from its
own previous value, diagonal Σ), `full` (all previous values, fixed gap), `full_dt` (**used**).
Chosen on the real data: ridge alpha 30, shrinkage 0.15. `patients_seen_` records who was used in
fitting; the evaluation asserts no locked-test patient is in it.

### 4.2 `explain/conditional_shapley.py`: the explainer

**One estimator, pluggable "fillers"**, so differences between methods come only from how hidden
features are filled:

| Filler | Fills hidden features from |
|---|---|
| `InterventionalFiller` | a random complete development visit: **standard SHAP** |
| `PopulationFiller` | population conditional given what's shown, no history (Gaussian copula over all 26; the Aas et al. 2021 approach). **This is the ablation that isolates what history adds** |
| `AR1Filler` | this patient's next-visit model: **AR(1)-SHAP** |

Estimator details (`shapley_instance`, `explain_many`):
- **Permutation Shapley** with antithetic pairs. At **every** step of every ordering the hidden
  block is re-drawn (`n_mc` draws), because the conditioning set changes. Final run: 48
  orderings × 24 draws.
- **Exact add-up** by construction: `Σ φ = f(x) − v̂(∅)` (verified to ~1e-16).
- **Shared random numbers** along each ordering (`chain_state`). Each draw keeps the right
  distribution but step-to-step differences are less noisy.
- **Inert columns:** under `AR1Filler`, revealing a fixed/clock feature reuses the previous
  step's value, so they get exactly 0 credit, not 0 ± noise.
- **Features missing at the visit** are not players: they stay missing in every filled row and get
  0 credit, under all fillers.
- Gaussian conditioning: `gauss_conditional_draw` (closed form, Cholesky with PSD fallback).

### 4.3 `evaluation/evaluate_ar1_shap.py`: the five tests

- **Model explained:** binary TD-vs-PIGD XGBoost (`backend.make_classifier("xgb", 2)`, balanced
  weights), trained on development patients only. Output explained: **log-odds of PIGD**.
- **Explained visits:** the 940 labelled TD/PIGD visits of the 88 locked-test patients that have a
  previous visit.
- **Standard SHAP background:** 2,150 complete development visits (complete, so standard SHAP
  doesn't inject missing values, which would be unfair to it in the realism test).
- **Tests:** T1 visit-to-visit model fit · T2 synthetic exact-answer check · T3 realism of filled
  visits · T4 how explanations change · T5 seed stability. TreeSHAP (XGBoost `pred_contribs`) is
  also reported for reference.
- `make ar1shap` takes **~32 min**. `--quick` is a 2-minute smoke test that writes and logs nothing.

### 4.4 Tests (`tests/test_ar1_shap.py`, 12)

Copula round trip exact · Gaussian conditional matches closed form · pairs never cross patients and
carry-forward is flagged · roles split correctly · history beats population · exact add-up for
every filler · fixed/clock get 0 under AR(1) · missing features get no credit and stay missing ·
AR(1) fills stay on the data's support · AR(1) fills sit nearer the previous visit than standard
SHAP's · Σ(Δt) positive-definite for any gap · locked-test patients not in development data.
**Full suite: 51 passed, 2 skipped** (the 2 need `make conformal transition` first).

---

## 5. Decisions taken, and why

| # | Decision | Why |
|---|---|---|
| D1 | Explain the **binary subtype model** | The only model with usable signal. The cheap transition model is at chance (0.53), so explaining it would explain noise |
| D2 | Explain `f(x_t) − E[f \| x_{t-1}]` | Falls straight out of Eq. 3; matches "explain the shift" |
| D3 | Fixed/clock features known; copula for the rest | A literal Gaussian is singular and samples impossible values |
| D4 | Carry-forward ≤12 months + stale flag; missing-today = no credit | 75% of pairs are incomplete; imputing would invent data |
| D5 | **Not** merged into FCX yet | Kept separate to evaluate cleanly; the filler interface makes merging easy later |
| D6 | Gap-dependent mean and spread, not a full matrix exponential | Gaps are mostly 3 or 6 months; buckets are simpler, fit the data and are testable. The fixed-gap version is kept as the ablation |
| — | Use the locked test for **explaining** | Not model selection, so the "score once" rule isn't broken; gives an honest out-of-sample setting |

---

## 6. Results (all from `reports/metrics/ar1_shap_results.txt`)

The explained model scores **0.676** balanced accuracy on the locked test (project level ~0.69).

### 6.1 T1: Does a patient's past predict their next visit? ✅

| Visit-to-visit model | Held-out log-lik (higher = better) | Variance explained |
|---|---|---|
| population only | −1.123 | 0 |
| own previous value only | −0.918 | 0.469 |
| all previous values, fixed gap | −0.849 | 0.499 |
| **all previous values + gap (used)** | **−0.843** | 0.497 |

By gap: ~3 mo −0.789 → −0.786; ~6 mo −0.863 → −0.857; ≥9 mo −1.016 → −0.984 (gap-awareness helps
most for long gaps). Leftover spread grows with the gap: 0.554 / 0.611 / 0.645.

### 6.2 T2: Is the estimator correct? ✅

Simulated patients (known dynamics, linear model), so the exact value is computable over all 2⁵
coalitions.

| | Correlation with exact | Average error |
|---|---|---|
| AR(1)-SHAP with fitted parameters | **0.990** | 15.0% |
| same estimator with true parameters | 0.994 | 11.4% |

Most error is sampling noise (it shrinks with more samples); fitting costs ~3.6 points. **Known
property:** a feature the model doesn't use (weight 0) but which is correlated with used ones gets
credit 0.211 (true) / 0.208 (estimated); standard SHAP gives it 0. Conditional methods credit what
is *informative*, not only what the model *reads*. This must be stated in the paper.

### 6.3 T3: Are the filled-in patients realistic? ✅ (the "Frankenstein" claim)

Half the features of each test visit are hidden and filled; a classifier tries to tell real from
filled (0.5 = can't tell).

| Method | Detectable (26) | Detectable (21 changing) | Jump ratio (1 = realistic) |
|---|---|---|---|
| standard SHAP | **0.976** | 0.861 | 2.31 |
| population-conditional | 0.959 | 0.822 | 1.84 |
| **AR(1)-SHAP** | **0.679** | **0.681** | **1.12** |
| AR(1)-SHAP, roles learned | 0.845 | 0.659 | 1.12 |

Correlations alone barely help; **the patient's history is what fixes it.**

### 6.4 T4: How do the explanations change?

**Years since diagnosis** (rank 4 under standard SHAP, 9.5% of credit):

| Method | Raw credit | Share of credit | Rank |
|---|---|---|---|
| population-conditional | −24.8% | −32.5% | 4 |
| AR(1)-SHAP | −100% (**by design**: it's declared a clock) | −100% | 24 |
| **AR(1)-SHAP, roles learned** | **−86.8%** | **−74.0%** | 19 |

**Quote the −86.8%, not the −100%.** The −100% is built in. The −86.8% is learned from data and
matches an earlier prototype result (~83%).

| Credit by feature type | standard | pop-cond | AR(1) | AR(1) learned |
|---|---|---|---|---|
| fixed (sex, handedness) | 14.6% | 12.2% | 0% | 0.1% |
| clocks | 15.7% | 13.5% | 0% | 5.2% |
| changing (21) | 69.7% | 74.3% | 100% | 94.7% |

- **Top driver per visit:** standard → `SEX` 26%, `YRS_SINCE_DIAGNOSIS` 18%, `LEDD_TOTAL_MG` 16%.
  AR(1) → `NP2RISE` (rising from a chair) 27%, `LEDD_TOTAL_MG` 14%, `PDTRTMNT` 8%.
- The top driver differs on **72.0%** of visits (57.7% comparing changing features only).
- Per-visit ranking agreement with AR(1): standard 0.227 (0.573 on changing features);
  population 0.287 (0.564). AR(1) vs roles-learned: 0.867 (0.969 on changing features).
- **Credit goes to what moved unexpectedly** (rank correlation of credit vs surprise since last
  visit): standard +0.06, population +0.18, **AR(1) +0.41**. Partly true by definition, so treat
  it as a consistency check.
- Total explained amount halves (1.06 → 0.47 log-odds), because the reference is now the
  patient's own expected value.

**What did NOT hold:**
- ❌ *"History matters more for slowly-changing features":* Spearman −0.13. An earlier
  per-feature prototype gave r = 0.64; it doesn't reproduce with the full model.
- ❌ *"Sleep and mood become the top drivers"* (the briefing's example): they gain share (`STAI`
  3.3→6.0%, `GDS` 1.6→3.1%, `ESS` 2.8→3.8%) but `NP2RISE` and LEDD lead. **Update the
  example in the paper/slides.**
- ⚠️ *"Standard SHAP's top driver is years since diagnosis":* only partly; it's 2nd, after `SEX`.

### 6.5 T5: Stability ✅

Same 80 visits, new random seed, per-visit ranking agreement: standard 0.988, population 0.965,
**AR(1) 0.988**, AR(1) learned 0.978.

### 6.6 Claims table for the paper

| Claim | Evidence | Status |
|---|---|---|
| Standard SHAP explains longitudinal visits with unrealistic patients | T3: 97.6% detectable, jumps 2.3× | ✅ |
| AR(1)-SHAP's filled patients are realistic | T3: 67.9%, jumps 1.12× | ✅ (not perfect) |
| The estimator is correct | T2: r = 0.99 | ✅ |
| History, not just correlation, changes the answer | population-conditional moves little | ✅ |
| Time-since-diagnosis credit is a population-sampling artefact | −86.8% when learned | ✅ |
| Explanations point at what changed for this patient | +0.41 vs +0.06 | ✅ partly by definition |
| History matters more for slow features | −0.13 | ❌ |
| New top drivers are clinically actionable | no ground truth | ⚪ needs clinician review |

---

## 7. Model ceiling check (`notebooks/model_ceiling_check.py`)

Question: can the subtype model go past ~0.70? Development patients only, same 3,692 visits for
every row (visits that have a previous visit), grouped 5-fold CV, nothing logged.

| Approach | Balanced acc | AUROC |
|---|---|---|
| Today's visit, 26 features (baseline) | 0.686 | 0.744 |
| + previous visit values | 0.687 | 0.749 |
| + previous values + changes | 0.688 | 0.746 |
| + running mean and worst-so-far | 0.690 | 0.756 |
| Only OFF-medication exams | 0.626 | 0.687 |
| Smoothed label (avg of t-1, t, t+1) + history | **0.699** | 0.766 |
| **Previous visit's label alone** (needs last exam) | **0.803** | 0.820 |
| Previous exam ratio alone | 0.799 | 0.872 |
| History + previous exam ratio | 0.797 | 0.879 |
| **Ceiling:** last visit's real label used as today's prediction | **0.833** | — |

**Reading:** cheap features cap at ~0.70, and history doesn't help. The label is noisy: even the
real exam only predicts itself 0.83 at the next visit. The jump to 0.80 needs the previous exam
(it breaks the cheap-only rule) and still doesn't beat "copy last label" on balanced accuracy.
**Decision:** don't chase accuracy; the explainability framework leads, and 0.69 is fine if
reported openly.

---

## 8. Limitations to write in the paper

- The explained model is modest (0.68); attributions describe the model, not the disease.
- Conditional credit can land on unused features (§6.2).
- `NP2RISE`, the new top driver, is an axial item flagged as possible proxy leakage (`README.md` §9.3).
- Carried-forward values up to 12 months old are treated as "previous".
- The copula treats ordinal values as points when conditioning (approximation).
- Single cohort (PPMI), 88 explained patients, no external validation.
- **Novelty is unverified** against Gaussian conditional Shapley (Aas, Jullum & Løland 2021),
  Frye et al. 2021 (manifold Shapley), TimeSHAP, Dynamask. Our defence: the temporal anchor, the
  gap model, tree models on sparse visits, and the realism evidence. The population-conditional
  ablation is what shows history matters. **Do a proper literature search before submitting.**

---

## 9. Future scope (prioritised)

### A. Turn it into a general framework (main next step)
1. ✅ **Auto-detect feature roles** — **done, 10 Oct 2026** (§14). Constant within patient → fixed;
   linear in time within each patient → clock; manual override still available.
2. **Simple API:** `fit(table, id_col, time_col, features)` → `explain(model, visit, previous
   visit)` → `check()` (runs the T2/T3 health checks on any dataset). Package it under one name.
   The pieces exist (`TransitionModel.fit`, the fillers, `evaluate_ar1_shap.py`'s T1–T5); this is
   about wrapping them in one clean interface, not building anything new.
3. **Show it works with any model:** XGBoost, random forest, logistic regression.
4. **Second dataset** (what makes it a framework): ✅ **data ready** — ADNI (Alzheimer's) is
   downloaded, preprocessed, and tested (§13). ⏳ **Still to do:** run AR(1)-SHAP itself on it,
   which needs item 1 done first. Fallback if that stalls: a strong simulated benchmark.
5. Extensions: use the last 2 visits (AR(2)); grouped features; a full continuous-time gap model
   (matrix exponential) as an ablation against the bucket version.
6. Open-source the code (no PPMI data) with a worked example.

### B. Strengthen the evidence
7. **Clinician review:** show ~20 visits with standard vs AR(1) explanations, ask which is more
   useful. This is the only way to support "actionable".
8. Faithfulness test: deletion/insertion curves with AR(1) fills.
9. Report the T2 "unused feature gets credit" property with a real-data dummy-feature test.
10. Run with larger sampling (e.g. 96 orderings) to cut per-visit noise.
11. Optional: merge into FCX as its background sampler (decision D5).

### C. Paper and slides
12. Replace the briefing's "sleep and mood" example with real findings (`NP2RISE`, LEDD).
13. Use −86.8% (learned), never −100%.
14. Add the 0.83 ceiling result to explain capped accuracy.
15. The paper draft (`reports/paper/TRACE-PD_paper_draft_v7.docx`) still has 13/15 incomplete
    references and no Results section (`README.md` §16).

### D. Pre-existing repo issues (not touched on 1 Oct)
16. The binary subtype model isn't persisted by `models/conformal.py` (only 3-class is saved).
    `evaluate_ar1_shap.py` saves its own copy to `models/ar1_shap_artifacts.pkl`.
17. Orchestrator validator is a no-op; tracker first-visit deltas 0.0 vs NaN; `inference.py
    --patient-id` not wired (`docs/NEXT_STEPS_RUTU.md`).
18. Medication-state issue (25.6% of same-day OFF/ON exams disagree) and the 845 zero-PIGD
    edge-case rows need a sensitivity analysis.

---

## 10. How to run

```bash
cd Major_project
pip install -r requirements.txt          # Python 3.10; xgboost 3.1.2, sklearn 1.7.2, numpy 2.2.6, pandas 2.3.2
export PYTHONPATH=src                    # PowerShell: $env:PYTHONPATH="src"   (package isn't installed)

python -m pytest tests -q                                   # 51 passed, 2 skipped
python -m trace_pd.evaluation.evaluate_ar1_shap --quick     # 2-min smoke test, writes nothing
make ar1shap                                                # full run, ~32 min, writes results + log
python notebooks/model_ceiling_check.py                     # ~1 min
```

**Gotchas:**
- `shap` is **not** installed and not needed; the Shapley code is our own.
- `models/` is git-ignored; artifacts must be regenerated on a new machine.
- Windows console is cp1252; scripts call `sys.stdout.reconfigure(encoding="utf-8")`. If you
  print Unicode elsewhere, do the same.
- Each full run **appends** to `experiment_log.csv`; repeated rows are repeated runs.
- Don't edit `configs/config.yaml` expecting changes; constants live in the scripts.

---

## 11. Prompt to give your Claude Code

> I'm continuing the TRACE-PD project. Read `HANDOFF_AR1_SHAP.md`, `CLAUDE.md` and
> `docs/ar1_shap.md` first. AR(1)-SHAP is implemented in `src/trace_pd/explain/ar1.py` and
> `src/trace_pd/explain/conditional_shapley.py` and evaluated by
> `src/trace_pd/evaluation/evaluate_ar1_shap.py`. Keep the repo rules: features by dictionary
> bucket only, everything split by patient, never fit on the locked-test patients, never copy
> PPMI rows out of the repo. Next task: [pick from §9, e.g. "make feature roles auto-detected
> and add a fit/explain/check API so the method runs on any longitudinal table"]. Run
> `python -m pytest tests -q` before and after. Report results plainly, including what doesn't
> work.

---

## 12. Plain-language glossary

| Term | Meaning here |
|---|---|
| SHAP / Shapley value | how much each feature pushed this prediction up or down |
| filler | how a hidden feature's value is filled in when measuring its importance |
| standard SHAP | fills from a random other visit |
| population-conditional | fills with values typical of similar people, ignoring history |
| AR(1)-SHAP | fills with values realistic for this patient, given their last visit |
| fixed / clock / changing | never changes / ticks with the calendar / genuinely varies |
| copula | the trick that lets a bell-curve model handle 0–4 scores and counts |
| detectable (T3) | how easily a classifier spots filled-in patients as fake (0.5 = can't) |
| jump ratio (T3) | how big filled-in visit-to-visit changes are vs real ones (1 = realistic) |
| locked test | 88 patients never used for fitting, used once for honest evaluation |
| balanced accuracy | accuracy averaged over TD and PIGD so the bigger class doesn't dominate (0.5 = chance) |

---

## 13. ADNI second cohort — preprocessing (done, 10 Oct 2026)

**Why:** AR(1)-SHAP proven on PPMI alone is a result for Parkinson's data, not a framework. To
call it a framework (the explicit decision taken this session), it has to work on a second,
different disease. ADNI (Alzheimer's) was chosen and access obtained. This section covers getting
the raw ADNI download into the same shape PPMI is in. **AR(1)-SHAP has not been run on it yet** —
that needs the code generalised first (§9.A.1).

### 13.1 What the download actually was

ADNI's current data package (`ADNIMERGE2`, the ATRI Biostatistics R package) is **not** one
pre-merged `ADNIMERGE.csv` the way older guidance describes. It's an R package shipping **~235
separate raw tables**, one `.rda` file each. There is no single file to convert — the same kind of
multi-table join PPMI needed had to be redone for ADNI.

`notebooks/convert_adnimerge.py` converts the **15 tables needed** (not all 235 — the rest are
imaging/PET/biomarker tables out of scope for a cheap-features project) from `.rda` to CSV using
`pyreadr` (pure Python, no R install needed). One table, `ADSL`, fails to load — it uses a newer R
storage format `pyreadr` can't parse — but it isn't needed: `PTDEMOG` + `DXSUM` + `REGISTRY` cover
the same ground. Output: `data/raw/adni_csv/*.csv`.

### 13.2 The pipeline: `src/trace_pd/data/preprocess_adni.py`

Built to the same discipline as `preprocess.py`, so the full design rationale is **in the
script's own docstring** (read it before changing anything) — this section is the summary.

**Run:** `make preprocess-adni` (or `PYTHONPATH=src python -m trace_pd.data.preprocess_adni`).
**Output:** `data/processed/adni_dx_long.csv`, `adni_dx_dictionary.csv`,
`adni_dx_preprocess_runlog.txt`.

**The label is structurally different from PPMI's, and that's good news for the project.**
PPMI's label is a deterministic formula (the Stebbins ratio) — the whole project's framing is
"can cheap data reproduce what the formula would say". ADNI's label (`DIAGNOSIS`: CN / MCI /
Dementia, from `DXSUM`) is a **clinician's consensus call**, informed by but not computed from a
full work-up. That reframes the task as "can cheap data + history estimate the diagnosis a full
work-up would reach, between visits where that work-up isn't repeated" — arguably a more natural
clinical story. It also means patients genuinely **convert** over time (CN → MCI → Dementia) —
closer to what a "transition risk" / "time-to-confidence" paper wants than PPMI's label wobble
was.

**Mapping onto PPMI's structure:**

| PPMI | ADNI | Role |
|---|---|---|
| `PATNO` / `EVENT_ID` / `VISIT_MONTH` | `RID` / `VISCODE2` / `VISIT_MONTH` | identifiers, timeline |
| Stebbins formula (16 items) | `DXSUM.DIAGNOSIS` | the label |
| `NP3TOT`, `NHY`, 16 formula items (`BANNED_LABEL_DEFINING`) | the full CDR exam, 8 columns (`BANNED_LABEL_DEFINING`) | banned — see 13.3 |
| `GENETIC_COHORT` (`ADMIN`, held out) | `APOE_GENOTYPE` (`RESOURCE_DEPENDENT_FEATURE`) | genetic test, not routinely available |
| 26 `CHEAP_FEATURE` columns | 23 `CHEAP_FEATURE` columns | demographics, cognition, function, mood, vitals |
| `LABEL_FLIPPED_NEXT` | `LABEL_FLIPPED_NEXT` | same construction: does the label differ at the next visit |

**The 23 cheap features:** `SEX`, `HANDED`, `EDUCATION_YEARS`, `AGE_AT_VISIT` (demographics) ·
`MMSE_TOTAL`, `MOCA_TOTAL`, `LOGMEM_IMMEDIATE`, `LOGMEM_DELAYED`, `BOSTON_NAMING`,
`CATEGORY_FLUENCY_ANIMALS`, `TRAILS_A_SEC`, `TRAILS_B_SEC`, `DIGIT_SPAN` (cognition) ·
`FAQ_TOTAL`, `ECOG_PATIENT_TOTAL`, `ECOG_PARTNER_TOTAL` (function) · `GDS_TOTAL`, `NPI_TOTAL`
(mood/behaviour) · `WEIGHT_KG`, `HEIGHT_CM`, `BP_SYSTOLIC`, `BP_DIASTOLIC`, `PULSE` (vitals).

### 13.3 Decisions that needed real checking, not assumption (full detail in the script's docstring)

1. **The entire CDR exam is banned** (`CDMEMORY`, `CDORIENT`, `CDJUDGE`, `CDCOMMUN`, `CDHOME`,
   `CDCARE`, `CDGLOBAL`, `CDRSB`) — **verified empirically, not assumed**: on 10,755 visits with
   both a diagnosis and a CDR score, `CDGLOBAL=0` → CN 89.7% of the time, `CDGLOBAL≥1` →
   Dementia 93.4% of the time, with `CDGLOBAL=0.5` the ambiguous band split between MCI and mild
   Dementia — the ADNI analogue of PPMI's Indeterminate zone around its two cutoffs. The
   preprocessing script re-runs this check and **asserts** >85% on both, so a changed extract
   can't silently un-verify the ban.
2. **MMSE and Logical Memory delayed recall (`LOGMEM_DELAYED`) are kept as features, not
   banned**, despite also informing the clinical diagnosis — checked and found to be a *softer*
   case than CDR: group means separate (CN 13.6, MCI 8.0, Dementia 1.7 for `LOGMEM_DELAYED`) but
   with wide, overlapping spread (sd 3–5), unlike CDR's near-complete separation. Disclosed, not
   banned — the ADNI analogue of PPMI's `NP2RISE`/`NP2TURN` axial-item disclosure.
3. **A sentinel-code bug, the same shape as PPMI's**: `VSWEIGHT`/`VSHEIGHT` use `-1` for "not
   recorded", mixed in with real measurements (137 rows). Fixed before use.
4. **Unit mixing + a second real bug found by checking, not assuming, the output.** Weight is
   recorded in pounds or kilograms, height in inches or centimetres, inconsistently. After
   converting to kg/cm, a **plausibility clamp** (30–250 kg, 130–220 cm) caught 4 weights and 27
   heights that converted to impossible figures (one row would have been a 490 cm patient) — a
   wrong unit tag on the source form, not a real measurement. Set to missing rather than guessed.
5. **742+ patients would have been silently dropped** if "screening" visits were simply discarded
   the way PPMI discards its unscheduled codes: of 2,752 patients with a screening-visit
   diagnosis, 2,010 also have a redundant baseline diagnosis (screening correctly dropped as
   noise), but a further set have **no separate baseline visit at all** — their screening visit
   is their only anchor in time. Checked directly (not assumed) before deciding: screening is
   promoted to `VISIT_MONTH=0` only for a patient with no other month-0 entry. Net effect:
   1,397 extra patient-visits recovered, 4,470 patients in the final table (up from 3,073 before
   this check).
6. **The visit timeline uses `VISCODE2`, not `EXAMDATE`.** ADNI re-codes every visit's raw,
   phase-specific code into a `VISCODE2` that's already a consistent "months since this patient's
   own baseline" number, stable across a patient moving between ADNI1 → ADNIGO → ADNI2 → ADNI3 →
   ADNI4. Verified on an actual patient who crossed phases: `EXAMDATE` keeps pace with
   `VISCODE2`'s number the whole way through. This also sidesteps `EXAMDATE` being missing on
   ~15% of rows.
7. **Height is carried forward per patient.** It's recorded at a minority of visits (adults don't
   get re-measured each time); forward/backward-filling within a patient raised usable coverage
   from 29% to 99%, the same reasoning PPMI uses for its static demographic columns.
8. **No imputation outside a fold** — same policy as PPMI. `NaN` stays `NaN`.

### 13.4 Results

| | PPMI | ADNI |
|---|---|---|
| Patient-visits | 6,922 | **24,302** |
| Patients | 439 | **4,470** |
| Labelled rows | 5,638 (81.5%) | **14,336 (59.0%)** |
| Label classes | TD 53.5% · PIGD 35.5% · Indeterminate 11.0% | **CN 39.4% · MCI 41.2% · Dementia 19.4%** |
| Cheap features | 26 | **23** |
| Banned (label-defining) | 19 | **8** |
| Labelled consecutive-visit pairs | 4,818 | **5,615** |
| Overall next-visit flip rate | 28.7% | **8.7%** |

**The much lower ADNI flip rate (8.7% vs PPMI's 28.7%) is expected, not a bug**: Alzheimer's
diagnostic conversion is a slower, rarer event per visit than PD subtype re-classification. It
also means the class-conditional story is different — flip rate by current label: CN 5.2%, MCI
13.4%, Dementia 2.4% (MCI, the boundary state, is by far the least stable, which nicely parallels
PPMI's Indeterminate-visits-flip-76%-of-the-time finding).

**Structural missingness mirrors PPMI exactly**, for the same reason (instrument schedule, not
randomness): `MOCA_TOTAL` 16.8% present (MOCA replaced MMSE only in later ADNI phases),
`ECOG_PATIENT_TOTAL`/`ECOG_PARTNER_TOTAL` 11.4% (introduced later), `DIGIT_SPAN` 15.6%, versus
`AGE_AT_VISIT`/`EDUCATION_YEARS` at 96.2%. Same policy as PPMI: leave as `NaN`, never impute here.

### 13.5 Tests

`tests/test_adni_dataset_contract.py` (12 tests, all passing), mirroring
`test_dataset_contract.py`: exact shape, patient count, no duplicate `(RID, VISCODE2)` keys,
valid label values, labelled-row count, **CDR columns never in a feature bucket**, **APOE
genotype never a cheap feature**, the CDR↔diagnosis association re-verified at >85% (so a changed
extract can't silently invalidate the ban), no sentinel values reach `WEIGHT_KG`/`HEIGHT_CM`,
age within a plausible range, transition-pair count, `VISIT_MONTH` fully populated and
non-negative. **Full repo suite: 63 passed, 2 skipped** (unchanged skips — need `make conformal
transition`).

### 13.6 What's next for ADNI specifically — ✅ all done, 10 Oct 2026, see §15

1. ✅ ~~Generalise `explain/ar1.py`~~ — see §14.
2. ✅ ~~Train a cheap-feature classifier on ADNI~~ — `train_adni.py`, binary CN-vs-Dementia
   chosen as the parallel to PPMI's TD-vs-PIGD (see §15 for why), 0.980 balanced accuracy.
3. ✅ ~~Run the five AR(1)-SHAP tests on it~~ — `evaluate_ar1_shap_adni.py`.
4. ✅ ~~Report PPMI and ADNI side by side~~ — §15.
5. ✅ Decided: binary CN-vs-Dementia, dropping MCI — MCI is ADNI's unstable middle diagnosis
   (13.4% flip to something else at the next visit vs CN 5.2%, Dementia 2.4%), the same role
   Indeterminate plays in PPMI. Exactly mirrors PPMI's own design decision.

---

## 14. Generalising `explain/ar1.py` — auto-detected feature roles (done, 10 Oct 2026)

**Why:** `ar1.py` had PPMI's column names written into it (`STATIC = ["SEX", "HANDED"]`,
`CLOCK = ["AGE_AT_VISIT", ...]`). That made it a PPMI script, not a framework — it couldn't run on
ADNI (or anything else) without hand-editing. This section makes it detect those roles from the
data itself, with no column names assumed, and was the blocking task for everything in §13.6.

### 14.1 What changed

| | Before | After |
|---|---|---|
| Feature roles | Hard-coded `STATIC`/`CLOCK` lists, looked up by name | `detect_feature_roles(visits, features, id_col, time_col)` — statistical, works on any table |
| Patient/time columns | Hard-coded `"PATNO"`/`"VISIT_MONTH"` throughout `build_pairs` and `TransitionModel` | `id_col=`/`time_col=` parameters everywhere, defaulting to PPMI's names |
| Text-valued features | `build_pairs` silently coerced every feature to numeric — would turn ADNI's `SEX="Male"/"Female"` into all-`NaN` | Non-numeric columns pass through untouched; a stored, reused category→code map (`fit_category_maps`) feeds them into the copula correctly |
| Manual override | Not possible | `TransitionModel(..., static=[...], clock=[...])` still works, for an explicit forced role |

**`detect_feature_roles`, the core addition:**

- **STATIC**: among patients with ≥2 recorded values, the value is unchanged ≥98% of the time.
  Works on numbers, text or categories (exact equality).
- **CLOCK**: among patients with ≥3 recorded values, fit *value ~ time* separately **within each
  patient**; pool the leftover variance across patients. A clock feature's value is explained
  almost perfectly (R² ≥ 0.97) by each patient's own straight line in time, because it's
  deterministic given the calendar.
- **DYNAMIC**: everything else. Ties are broken toward dynamic on purpose — it costs a little
  model quality, whereas wrongly calling a feature static/clock means it's never sampled at all,
  corrupting every explanation that touches it.

**A real bug the first version of this hit, and the fix.** The obvious design — test whether the
*change between consecutive visits* is explained by the gap length — **failed on PPMI's own
known clock feature**, `AGE_AT_VISIT` (R² = 0.27, not ~1.0). Cause: age is rounded to one decimal
year, and over a 3-month gap the true change (0.25 years) is the same size as the rounding step
(0.1), so the signal drowns in rounding noise. Fixed by testing each patient's **full trajectory**
instead of consecutive pairs — rounding noise is negligible next to years of real drift, giving
R² = 0.986–0.989 on PPMI's actual clock features. Lesson for any future detector: short-gap,
pairwise tests are fragile against realistic measurement rounding; whole-trajectory tests aren't.

**A second bug, found while building the end-to-end ADNI test, not assumed away.** The initial fix
for text features called `pd.factorize` fresh on every call. That's wrong: a single row containing
only `"Right"` would be coded `0`, even though training (which saw both `"Left"` and `"Right"`)
coded `"Right"` as `1` — a silent, hard-to-notice misalignment between training and inference.
Fixed with `fit_category_maps`: the category→code mapping is built **once**, from the full
training data, stored on the fitted model, and reused for every later call.

### 14.2 Validation

- **Reproduces PPMI's exact known roles**, verified against the hard-coded reference
  (`PPMI_STATIC`/`PPMI_CLOCK`, kept only as a labelled answer key now):
  static `{SEX, HANDED}`, clock `{AGE_AT_VISIT, YRS_SINCE_DIAGNOSIS, YRS_SINCE_SYMPTOM_ONSET}`,
  21 dynamic — exact match, with R² 0.986–0.989 on the real clock features.
- **Re-running the full evaluation pipeline on PPMI reproduces the documented numbers exactly**
  (same T1 log-likelihoods to 4 decimal places, same gap-by-gap breakdown) — the generalisation
  changed *how* roles are found, not the roles themselves, so every number in §6 still stands.
- **Runs correctly on ADNI with zero ADNI-specific code**: detects `SEX`, `HANDED` as static
  (same pattern as PPMI) and `AGE_AT_VISIT` as clock (same pattern as PPMI) — and also finds
  `EDUCATION_YEARS` is static, a feature that doesn't even exist in PPMI, entirely on its own.
- **Tested on a synthetic dataset with different column names and a string-valued static
  feature** (`SUBJECT_ID`/`MONTHS`/`GROUP="A"/"B"`), including the same coarse-rounding trap that
  broke the first version — correctly classified every feature.
- 8 new tests in `tests/test_ar1_shap.py` (20 total in that file; **71 passed, 2 skipped** across
  the whole repo).

### 14.3 A documented limitation, not fixed (narrow, no real case hits it yet)

The Shapley/filler layer (`conditional_shapley.py`) still expects **numeric** feature vectors —
the same precondition any SHAP-style explainer has. A text-valued feature that is **dynamic**
(changes per patient, not static) would need to be numerically encoded before reaching that layer;
`numeric_matrix` raises a clear `ValueError` naming the column rather than crashing deep in pandas
if this is ever hit. Neither PPMI's nor ADNI's current dynamic features are text-valued, so this
hasn't come up in practice — noted here so it doesn't surprise a future dataset.

### 14.4 Files touched

`src/trace_pd/explain/ar1.py` (the generalisation itself) ·
`src/trace_pd/evaluation/evaluate_ar1_shap.py` (updated to call `detect_feature_roles` and log
*why* each feature got its role) · `tests/test_ar1_shap.py` (+8 tests).

---

## 15. AR(1)-SHAP on ADNI — the framework claim, with evidence (done, 10 Oct 2026)

**This is the actual payoff of §13 and §14.** The same method, same code, same five tests, run on
a second disease with zero dataset-specific code in `explain/ar1.py` — the result below is what
makes "framework" a supportable claim rather than an aspiration.

### 15.1 What was built to get here

- **`src/trace_pd/evaluation/splits_adni.py`** — ADNI's locked 20% test split (894 patients),
  same methodology as PPMI's `splits.py` (stratified on baseline label, seed-fixed, patient-
  grouped CV folds). `log_run` and `patient_bootstrap_ci` are reused directly (already generic).
- **`src/trace_pd/models/train_adni.py`** — the model AR(1)-SHAP explains. Bake-off (majority,
  logistic, XGBoost, HGB) on two tasks, mirroring `compare_subtype_models.py`'s design exactly:

  | Task | Majority | Winner | Locked-test balanced acc | 95% CI | AUROC |
  |---|---|---|---|---|---|
  | 3-class CN/MCI/Dementia | 0.333 | XGBoost | **0.796** | [0.778, 0.816] | — |
  | **Binary CN vs Dementia** | 0.500 | XGBoost | **0.980** | [0.970, 0.988] | 0.996 |

  **Binary CN-vs-Dementia was chosen as the explained model** — the direct parallel to PPMI's
  decision to drop Indeterminate and report binary TD-vs-PIGD, because **MCI plays the same
  "unstable middle" role as Indeterminate**: MCI visits flip to a different diagnosis at the next
  visit 13.4% of the time, against 5.2% (CN) and 2.4% (Dementia) — see §13.4.
- **`src/trace_pd/evaluation/evaluate_ar1_shap_adni.py`** — ADNI's version of
  `evaluate_ar1_shap.py`, same five tests, same reasons for being a separate file as
  `preprocess_adni.py` is from `preprocess.py` (§13.2): lower risk while both are in active use,
  matching the project's own convention (`train_baseline.py`/`train_xgboost.py`,
  `splits.py`/`splits_adni.py`).

### 15.2 A real bug this run found, and the fix

Running the actual pipeline — not just the generalised library code — on ADNI immediately broke
something the PPMI pipeline never hit: **the standard-SHAP background pool was empty.** PPMI's
background was built from visits with *every* feature present (`.dropna()`); on ADNI, **not one
visit, out of 6,748, has all 23 features present at once** (max observed: 21/23) — its instruments
rotate across visit types far more than PPMI's do. Fixed by falling back to the full visit pool
(missing values included, never imputed) whenever the complete-row pool has fewer than 50 rows —
applied to **both** evaluation scripts for consistency. Verified this changes **nothing** for
PPMI (its complete-row pool, 2,150 rows, is comfortably above the threshold, so it still uses
exactly the same background as before — re-ran PPMI's quick check to confirm the exact same
`2150 complete visits` line reappears). A second, smaller fix in the same spot made the "jump
ratio" statistic NaN-robust for the same reason.

### 15.3 Results: PPMI vs ADNI, side by side

| | PPMI (TD vs PIGD) | ADNI (CN vs Dementia) |
|---|---|---|
| Explained model's locked-test accuracy | 0.676 (weak) | **0.980 (very strong)** |
| T1: history explains next-visit variance | R² 0.497 | R² 0.578 |
| T2: estimator vs exact truth | r = 0.990 | r = 0.990 *(identical — same simulation code/seed, not an independent re-confirmation)* |
| **T3: standard SHAP detectable as fake** | 97.6% | **97.7%** |
| **T3: AR(1)-SHAP detectable as fake** | 67.9% | **75.1%** |
| T3: standard SHAP jump ratio | 2.31 | **2.61** |
| T3: AR(1)-SHAP jump ratio | 1.12 | **1.18** |
| T4: clock feature's credit drop (roles learned) | −86.8% (years since diagnosis) | **−96.8%** (age) |
| T4: top driver flips between standard and AR(1) | 72.0% of visits | **72.4%** of visits |
| T4: credit tracks "surprise" (AR(1) vs standard) | +0.41 vs +0.06 | **+0.41 vs +0.00** |
| T4: history-matters-more-for-slow-features link | −0.13 (not supported) | **−0.52 (still not supported, same direction)** |
| T5: seed-to-seed stability, standard / AR(1) | 0.988 / 0.988 | **0.992 / 0.985** |

**Reading this:** the core claims replicate almost number-for-number on a disease with a
completely different label (clinician diagnosis, not a formula), a completely different feature
set (cognitive tests, not motor items), and a classifier at the opposite end of the accuracy
scale (0.98 vs 0.68). That combination — same effect size, on an unrelated disease, regardless of
whether the underlying model is weak or strong — is what "framework" is supposed to mean, not
just "worked once."

**One real difference, not a replication:** ADNI's standard SHAP is far more concentrated on one
feature — `FAQ_TOTAL` (an informant-reported functional questionnaire) is the top driver of **86%**
of visits under standard SHAP, dropping to 23% under AR(1)-SHAP, which instead spreads credit more
evenly across `FAQ_TOTAL`, `MMSE_TOTAL` and `CATEGORY_FLUENCY_ANIMALS`. PPMI never showed a single
feature dominating that strongly. This is a real, disease-specific finding — functional decline is
close to definitional for a dementia diagnosis — not an artefact, and it's worth keeping in the
paper as a point of genuine difference between the two cohorts rather than smoothing it into "same
as PPMI."

### 15.4 What this does and doesn't prove

| Claim | Status |
|---|---|
| The method runs on a second disease with zero code changes to `explain/ar1.py` | ✅ proven |
| The realism gap (standard SHAP's fakes are detectable, AR(1)'s aren't) replicates | ✅ proven, same magnitude |
| The clock-feature credit collapse replicates | ✅ proven, even larger on ADNI |
| It works regardless of whether the underlying model is weak or strong | ✅ proven (0.68 vs 0.98) |
| "History matters more for slow-changing features" | ❌ still not supported, on either cohort |
| The new top drivers are more clinically useful | ⚪ still untested — needs a clinician/domain expert on each cohort, not just PPMI's |
| A third, unrelated disease would also replicate | ⚪ two cohorts is evidence, not proof; say "two cohorts" in the paper, not "general" |

### 15.5 Files and how to reproduce

`make split-adni` → `make train-adni` → `make ar1shap-adni` (full run ~57 min; `--quick` for a
~5 min smoke test). Results: `reports/metrics/adni_baseline_results.txt`,
`reports/metrics/ar1_shap_results_adni.txt`, logged to `experiment_log.csv` under step ids
`adni-1` and `ar1-adni`. Artifacts: `models/ar1_shap_artifacts_adni.pkl` (git-ignored).

---

## 16. Novelty re-check and the new direction (10–11 Oct 2026)

**Read this before writing any part of the paper.** The novelty claim in the original briefing
(`AR1-SHAP_Technical_Briefing.pdf`) no longer holds as stated. This section records what was
checked, what was dropped and why, and the idea the project has now landed on. Nothing in this
section is implemented yet.

### 16.1 The core AR(1)-SHAP idea is already published

**DynSHAP: Towards Explainable Dynamic Survival Analysis** (Anokhina, Jürß, Liò; arXiv:2609.13042,
submitted 11 Sep 2026, about three weeks before our briefing) includes a variant, *Temporal
DynSHAP*, that fills hidden features from a linear-Gaussian model of the previous time step:
x_f^(t) ≈ w_{t,f}ᵀ x^(t−1) + b_{t,f} + ε, ε ~ N(0, σ²_{t,f}). It targets irregular longitudinal
clinical data and argues, as we did, that population (marginal) sampling builds unrealistic
patients. Its method section was verified directly on 11 Oct. The earlier search most likely
missed it because it was brand new.

**Other related work, less close than first reported:**

- **FIT** (Tonekaboni et al., arXiv:2003.02821, 2020) conditions on the patient's history, but it
  is a KL-divergence importance score, not a Shapley method.
- **DeltaSHAP** (Kim et al., arXiv:2507.02342, ICML 2025 workshop) explains the change between
  consecutive predictions using only observed feature coalitions, not a model of the past.
- **Aas, Jullum & Løland** (arXiv:1903.10464; the `shapr` package) do Gaussian/copula conditional
  SHAP. Our method is their approach with the previous visit added to the conditioning set.

**What our implementation does that DynSHAP does not** (a stronger version of the same idea, not
a new idea):

| | Temporal DynSHAP | Our AR(1)-SHAP |
|---|---|---|
| Conditions on other features seen at the same visit | No (Eq. 7 uses t−1 only) | Yes |
| Uses the length of the visit gap | No (fixed time bins) | Yes (gap buckets) |
| Noise model | Independent per feature | Full covariance |
| Ordinal / discrete features | Raw Gaussian | Gaussian copula |
| Static and clock features | Not treated separately | Detected and held fixed |
| Models explained | Deep survival models | Tree ensembles (XGBoost) |
| Stability | Self-reported ~3× worse than KernelSHAP | Equal to standard SHAP (0.985 vs 0.988) |
| Cohorts | PBC (312), MS (500) | PPMI (439), ADNI (4,470) |

**Implication:** the paper must say "we build on and improve DynSHAP", cite it, and compare against
it head-to-head. Writing "nobody has done this" would be rejected.

### 16.2 Directions explored and rejected (so nobody re-proposes them)

Five literature passes ran on 10–11 Oct. Each candidate was checked for prior work.

| Direction | Verdict | Why dropped |
|---|---|---|
| One model powering both SHAP and conformal intervals | Partly new | Moves away from explainability into prediction/uncertainty |
| Trajectory-level counterfactuals ("what trend change flips the next diagnosis") | Partly new | Counterfactual recourse, closer to prediction than a SHAP framework |
| Concept-group drift over a patient's visits | Partly new | Cosmetic without heavy extra validation |
| Visit-level Shapley ("which past visit mattered") | Already published | TimeSHAP event-level, WindowSHAP, groupShapley |
| Loss/error SHAP over time | Already published | LossSHAP (Lundberg et al., Nat. Mach. Intell. 2020), SAGE |
| SHAP on conformal set size / uncertainty | Mostly published | Johansson et al. COPA 2025; Watson et al. NeurIPS 2023; Il Idrissi et al. 2025 |
| Label noise as a Shapley player | New | Centred on model errors, which the team ruled out |
| Temporal-consistency axiom | Covered | Trivial for fixed backgrounds; Asymmetric Shapley (Frye et al. 2020) |
| Causal/Granger temporal Shapley | Covered | Asymmetric Shapley, Causal Shapley, Shapley Flow |
| Trend-component players (level/speed/acceleration/wobble) | New for irregular visits | Closest is C-SHAP (arXiv:2504.11159, regular univariate). Judged to answer one question, not a general framework |
| Frequency/wavelet attribution | Exists for dense regular signals | Not meaningful with 3–17 irregular visits |

**Team decisions that shaped the final idea:**

- Stay purely in explainability methods. The contribution is new mathematics in the SHAP value
  function, not better prediction, conformal sets, errors or transition risk.
- The models are weak (PPMI ~0.68), so the framework must not depend on model quality. An
  explanation method is judged on describing the model faithfully.
- **No synthetic or simulated data in the paper.** All validation must use real PPMI and ADNI
  data. The synthetic T2 check (§6.2) should be replaced or moved to a supplement.

### 16.3 The idea we have landed on: "expected vs new" SHAP

**One-line idea:** split each feature's SHAP credit into the part the patient's own history
predicted, and the part that is genuinely new at this visit.

> "Memory score pushed the prediction up by 0.30: **0.10 expected** from this patient's history,
> **0.20 new** this visit."

**Mathematics:**

- Work in Gaussian-copula space, z_j = Φ⁻¹(F̂_j(x_j)). Static and clock features are held fixed.
- Split each dynamic feature at visit t into a predictable part and an innovation:
  M_j = E[z_{j,t} | patient history] and E_j = z_{j,t} − M_j. This is the time-series
  "innovations" idea (Wold), the same split economists use for nowcast "news" (Bańbura & Modugno).
- Each feature becomes two Shapley players: P_j (predictable) and I_j (innovation).
- Value function, with M and E conditioned on the revealed players:
  v(𝒫 ∪ ℐ) = E[ f( T⁻¹(M + E) ) | M_𝒫 = m_𝒫, E_ℐ = e_ℐ ]
- Attribution is the Owen value with groups {P_j, I_j}, so **φ_j = φ_j^expected + φ_j^new
  exactly**, and the totals add up to the prediction.
- Cost is about twice standard SHAP.

**The history model behind "expected"** (each of these is new as a SHAP background):

1. **All past visits with measurement noise**, via a Kalman filter. One odd visit does not
   dominate the expected part.
2. **Exact time gaps** via a continuous-time Ornstein–Uhlenbeck process. A 12-month gap allows
   more change than a 3-month one, with no bins. Parameterise by drift A and diffusion Q_c so the
   covariance is always valid.
3. **A trust dial** from a random-effects (patient-level) model. The history weight
   Λ_n = Σ_b(Σ_b + Σ_w/n)⁻¹ grows with the number of prior visits n.

**Provable properties:**

1. **Efficiency:** expected parts plus new parts sum to f(x) − E[f].
2. **News consistency:** for a linear model, the new parts sum exactly to f(x) − E[f | history].
   This links the method to the economics news decomposition.
3. **Reductions:**
   - with no memory, it becomes Aas-style conditional SHAP;
   - with all patients identical, it becomes population SHAP;
   - with bins instead of exact gaps and no same-visit conditioning, it is Temporal-DynSHAP-like.
4. **Stability certificate:** |Δφ_k| ≤ 2·max TV ≤ 2·max √(KL/2), with closed-form Gaussian KL.
   This directly answers DynSHAP's self-reported instability.

**What is new versus the closest work:**

- The expected/new player split for black-box SHAP. The nearest relatives are Shapley over causal
  noise terms (Budhathoki et al., ICML 2022, for outlier scores in a known causal graph) and linear
  news decompositions; neither does this.
- Continuous-time, all-visit, patient-level history models used as the SHAP background.
- The parameter-stability certificate.

A reviewer may call the history model "just a better sampler", so **the paper must lead with the
expected/new split**.

### 16.4 Validation plan (real PPMI and ADNI data only)

1. **History model check:** on held-out patients, does p(z_t | history, gap) predict the next
   visit well? Compare against population-only, last-visit-only and a DynSHAP-style binned model.
2. **Visit thinning:** drop 20–50% of real visits. Explanations of fixed visits should stay
   stable. Compare against DynSHAP-style and standard SHAP.
3. **Bootstrap refits:** measure attribution spread, and check whether the stability certificate
   holds.
4. **Faithfulness:** deletion and insertion curves.
5. **Model-randomisation sanity check** (Adebayo et al., NeurIPS 2018).
6. **Clinical agreement:** the "new" share should jump at real ADNI MCI→Dementia conversion visits
   and PPMI subtype changes, and the "expected" share should dominate in stable patients.
7. **History shuffle:** permute one feature's real history across patients. Its expected credit
   should fall to about zero.

### 16.5 Known risks

- **The trust dial is weak in our data.** With 47% within-patient variance, κ ≈ 0.89. The dial
  already trusts the patient's history about 53% after one prior visit, 69% after two and 95%
  after sixteen, so it mostly matters early on.
- **Disease progression:** an OU process returns to a fixed level, but Parkinson's and Alzheimer's
  worsen. The mean must depend on age and disease duration, and random slopes (integrated OU) may
  be needed.
- **Linearity:** dependence is still linear-Gaussian after the copula transform.
- **The split depends on the history model.** If that model is poor, credit leaks between
  expected and new, which is why check 1 matters.
- **Lag features:** if the XGBoost model also takes past-visit features, they must be derived
  consistently from the same z.

### 16.6 Must-read before writing

- DynSHAP (arXiv:2609.13042), Section 3 in full
- Aas et al. (arXiv:1903.10464)
- shapr `explain_forecast` (arXiv:2504.01842)
- Budhathoki et al., ICML 2022
- Bańbura & Modugno 2014
- Diggle 1988, *Biometrics*
- Taylor, Cumberland & Sy 1994, *JASA*
- Owen 1977
- Lin, Covert & Lee, NeurIPS 2023 (arXiv:2306.07462)
- Sundararajan & Najmi, ICML 2020
- Chen et al. 2020 ("true to the model or true to the data")

Not opened during the search: Michiels et al., *DSS* 2024, and the full text of Amoukou et al.,
AISTATS 2022.

### 16.7 Publishability assessment of "expected vs new" SHAP (11 Oct 2026)

A dedicated search judged whether the §16.3 idea is journal-worthy. **Verdict: not journal-ready
as it stands; workshop-level now.** The team's position is that workshop-level is not good
enough, so treat this idea as **unconfirmed** until the gaps below are closed or a better idea is
found.

**Novelty:** partial, leaning new. No paper does a per-feature, per-visit split of SHAP credit
into "expected from the patient's own history" and "new at this visit". Close precedents a
reviewer will cite:

- **Janzing et al., "Quantifying intrinsic causal contributions"** (AISTATS 2024,
  arXiv:2007.00714). An inherited-vs-new split shared out with Shapley, but global (variance or
  entropy) and causal-graph based. **This is the main threat.**
- **Budhathoki et al.** (ICML 2022). Local Shapley over noise terms, for outlier scores.
- **Asymmetric Shapley Values** (Frye et al., NeurIPS 2020, arXiv:1910.06358). With a
  "history first" order, the current feature gets only its innovation. The paper must argue why
  keeping the expected part attached to feature j is better.
- **Causal Shapley Values** (Heskes et al., NeurIPS 2020). Direct/indirect split.
- **Michiels, De Vos & Suykens** (arXiv:2306.10880). Interventional + dependent split of
  conditional SHAP.
- **Nowcasting:** no Shapley/ML version of the Bańbura–Modugno news decomposition was found
  (closest: Hopp, arXiv:2203.11872). This is a possible selling point.

**Technical issue:** the symmetric Owen split inside {P_j, I_j} is not the same as ordered
"news" (history first) for non-linear models such as XGBoost. Either justify the symmetric
choice or report both.

**Rough acceptance odds (judgement, not measured):**

| Venue | As is | With the package below |
|---|---|---|
| Workshop / ML4H findings | 50–65% | — |
| JBI / AI in Medicine / IEEE JBHI | 10–20% | 40–55% |
| NeurIPS / ICML main track | <5% | <5% |

For calibration, DynSHAP (Liò's Cambridge group) itself landed as a NeurIPS 2026 *workshop*
poster, and DeltaSHAP as an ICML 2025 workshop paper.

**Three biggest rejection risks:**

1. **"Incremental, with no comparison to close prior work."** Asymmetric Shapley, Causal
   Shapley, `shapr` conditional SHAP and a DynSHAP-style baseline must be *run*, not just cited.
2. **No evidence the method recovers the correct answer.** The "no synthetic data" rule is
   largely self-imposed. Every close competitor (Aas et al., DynSHAP, FIT, Budhathoki) validates
   on simulated ground truth, and journals do not frown on it for a new attribution method.
   - If the rule stays, the minimum substitute is a **"real data, known model" test**: explain a
     linear or shallow-tree model fitted on real PPMI/ADNI data, where the true expected/new
     values can be computed exactly, and show the estimator recovers them.
   - Also report held-out calibration of the history model.
   - Deletion/insertion faithfulness metrics will probably favour plain TreeSHAP (as in DynSHAP;
     see Chen et al., arXiv:2006.16234). Say so up front.
3. **The explained models.** At PPMI 0.68, frame results strictly as explaining the model, not
   the disease. At ADNI 0.98, reviewers will suspect leakage, because MMSE and FAQ inform the
   clinical diagnosis. Justify the CDR-only ban or tighten it.

**Minimum package for a journal submission:**

- Formal proofs: exactness, the linear news identity, the reduction cases, the stability bound.
- Head-to-head baselines: TreeSHAP (interventional and path-dependent), `shapr`
  Gaussian/copula, Asymmetric Shapley with time order, DynSHAP-style, and TimeSHAP/WindowSHAP
  where applicable.
- The known-model recovery test (above).
- The §16.4 real-data checks, reported with confidence intervals.
- Improve or justify the PPMI model, and audit ADNI for leakage.
- Optional but valuable: 3–5 neurologists rating a sample of explanations.

**Venue fit if pursued:** JBI (best fit; see WindowSHAP, JBI 2023, which was real-data-only but
had strong baselines), AI in Medicine, IEEE JBHI/TBME (see IT-SHAP, TBME 2025), ML4H/CHIL.
npj Digital Medicine and JAMIA would expect clinician evidence.

**Status: open.** The team intends to keep searching for a stronger novelty angle with other
research tools. Anyone continuing should start from §16.1–16.2 (what is already published or
rejected) and this section, so they don't repeat the same searches.

### 16.8 Next steps

1. Write a one-page proposal from §16.3–16.4 for the guide.
2. Add a DynSHAP-style baseline to the existing evaluation scripts.
3. Build the history model (Kalman + OU + random effects) and run its check (§16.4 step 1) on PPMI.
4. Implement the expected/new Owen split.
5. Run the full §16.4 plan on PPMI, then ADNI.
