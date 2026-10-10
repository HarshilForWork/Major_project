# TRACE-PD — Codebase Guide and AR(1)-SHAP Implementation Plan

*State of the repo at commit `d760d6d` (30 Sep 2026), written 1 Oct 2026.*

> **Update 1 Oct 2026: AR(1)-SHAP is now built.** Decisions D1–D6 (§8) were taken as
> recommended, except D5 (FCX integration), which is left for later. Code, results and what did
> and did not hold: [`docs/ar1_shap.md`](docs/ar1_shap.md). Sections 6–10 below are the
> pre-build plan, kept for the record. The stale numbers in `README.md` §9.3, §9.5 and §9.6 were
> also synced.

This file sits next to the project document [`README.md`](README.md) and does not replace it.
`README.md` explains the clinical problem, the label and the decision log. **This file answers
three narrower questions:** what is in the code, how good each model actually is, and what has
to be built to add the AR(1)-SHAP method from `AR1-SHAP_Technical_Briefing.pdf`.

Numbers marked **[recomputed]** were recalculated on 1 Oct 2026 from
`data/processed/ppmi_tdpigd_long.csv`. All other numbers come from `reports/metrics/*` and
`docs/results_session_final.md`. No patient-level data appears in this file.

---

## 1. Summary

- **The pipeline is complete and clean.** Raw PPMI → labelled long table → subtype classifier →
  conformal sets → transition model → FCX explanations → inference demo. Leakage controls are
  enforced in code and tests (41 tests: 39 pass, 2 skip while `models/` is empty).
- **The models are weak, and the repo says so.** Binary TD-vs-PIGD sits at ~0.69 balanced
  accuracy, 3-class at ~0.46. Twelve configurations are statistically indistinguishable, so
  tuning is exhausted.
- **The cheap-feature transition model has no signal** (AUROC 0.53 next visit, 0.56 at 12
  months; Brier no better than predicting the base rate). Only a model that is given the exam
  and the patient's history is predictive (AUROC 0.80), and that model is not deployable.
- **The explainability layer (FCX) already exists** and uses its own interventional Shapley
  estimator. AR(1)-SHAP would be a second, separate contribution, and the relationship between
  the two has not been decided (§5).
- **AR(1)-SHAP is built and evaluated** (`docs/ar1_shap.md`, `make ar1shap`). Six decisions had
  to be taken first (§8), the biggest being *which model is being explained*.
- **Two claims in the briefing did not reproduce** (§7): the "Years-since-diagnosis is the top
  SHAP driver" example, and the 0.737 / 47.7% autocorrelation figures.

---

## 2. Repository map

```
src/trace_pd/                           6,439 lines
├── config.py                           every path resolves from here — never hardcode
├── data/preprocess.py        (660)     19 PPMI CSVs → ppmi_tdpigd_long.csv + dictionary
├── models/
│   ├── backend.py            (60)      make_classifier("xgb"|"hgb"), fit_classifier
│   ├── train_baseline.py     (248)     Exp 1–4: majority / logistic / HGB (+ leakage control)
│   ├── train_xgboost.py      (217)     Exp 1–3 with XGBoost
│   ├── conformal.py          (335)     LAC + APS sets; saves conformal_model_production.pkl
│   └── train_transition.py   (307)     next-visit and 12-month flip models (cheap features)
├── evaluation/
│   ├── splits.py             (183)     locked 20% test set, patient folds, experiment log, bootstrap
│   ├── compare_subtype_models.py       12-config subtype bake-off
│   ├── evaluate_c1_rounds.py (367)     per-round wide models (R1–R4, A3)
│   ├── validate_c3.py, improve_c3.py   transition models with signal (E2/E3/E5/E6) + FCX C3
│   ├── evaluate_fcx.py       (341)     FCX C1/C2/C3 evaluation
│   └── ablation.py, sporadic_baseline.py
├── explain/                            FCX — see §5
│   ├── formula.py            (40)      Stebbins rule in log-ratio coordinates
│   ├── shapley.py            (58)      OWN interventional Shapley (no `shap` dependency)
│   ├── implied_exam.py (93) C1 · straddle.py (70) C2 · pdn.py (173) C3
│   └── trajectory.py         (51)      per-visit history built on the FULL scheduled sequence
├── tracking/tracker.py, explanations/orchestrator.py, inference.py   serving path
└── viz/                                figures
tests/                                  7 test files, 41 tests
docs/                                   design docs; fcx.md and results_session_final.md matter most
reports/metrics/experiment_log.csv      331 rows: every run, including the losers (append-only)
data/interim/locked_test_patients.csv   88 held-out patients (seed 2026) — scored once per step
models/                                 empty on a fresh clone (git-ignored)
```

**Invariants you must not break** (full list in `CLAUDE.md`): the 26 features are chosen by
dictionary bucket (`CHEAP_FEATURE`), never by hand; no label-defining column in X; all CV
grouped by `PATNO`; `NaN` left as `NaN` outside a fold; locked-test patients never used for
selection. **Any new code that fits AR parameters must respect the same patient folds and the
locked split.**

---

## 3. Data facts that decide how AR(1)-SHAP can be built

Table: 6,922 visits × 75 columns, 439 patients, 5,638 labelled visits. Consecutive visit pairs
within a patient: **6,483**. All figures in this section are **[recomputed]**.

### 3.1 The 26 features are not Gaussian-friendly

| Type | Features | Count | Problem for `N(AX + b, Σ)` |
|---|---|---|---|
| Static | `SEX`, `HANDED` | 2 | zero within-patient variance → singular Σ |
| Deterministic clocks | `AGE_AT_VISIT`, `YRS_SINCE_DIAGNOSIS`, `YRS_SINCE_SYMPTOM_ONSET` | 3 | lag-1 r ≥ 0.997; they are `x_{t-1} + Δt`, so there is nothing to sample |
| Binary | `PDTRTMNT` | 1 | Gaussian draws fall between 0 and 1 |
| 5-level ordinal (0–4) | ten Part II items: `NP2SPCH`, `NP2SALV`, `NP2SWAL`, `NP2EAT`, `NP2DRES`, `NP2HYGN`, `NP2HWRT`, `NP2HOBB`, `NP2TURN`, `NP2RISE` | 10 | discrete and bounded; a continuous draw is off-support |
| Integer totals | `NP1RTOT`, `NP1PTOT`, `GDS_TOTAL`, `STAI_TOTAL`, `ESS_TOTAL`, `RBDSQ_TOTAL`, `SCOPA_AUT_TOTAL`, `MCATOT`, `N_CONMEDS` | 9 | bounded counts, right-skewed |
| Heavy-tailed continuous | `LEDD_TOTAL_MG` | 1 | needs a log transform |

So only about **21 features are genuinely stochastic**, and 20 of those are discrete or bounded.
A plain multivariate Gaussian puts probability mass on values no patient can have. That is the
same off-manifold problem the method is meant to remove, in a different place.

### 3.2 Temporal behaviour

| Quantity | Value | Note |
|---|---|---|
| Mean lag-1 autocorrelation (26 features, within patient) | **0.754** | briefing says 0.737, see §7 |
| Mean within-patient share of variance | **46.9%** | briefing says 47.7% |
| Highest autocorrelation | `N_CONMEDS` 0.935, `LEDD_TOTAL_MG` 0.845, `MCATOT` 0.813 | |
| Lowest autocorrelation | `STAI_TOTAL` 0.505, `NP2SWAL` 0.530, `GDS_TOTAL` 0.537 | nearly memoryless; AR conditioning adds little here |

### 3.3 Visit gaps are not "3, 6 or 12 months"

| Gap between consecutive visits (months) | 3 | 6 | 9 | 12 | 15 or more |
|---|---|---|---|---|---|
| Pairs | 2,130 | 3,897 | 64 | 350 | 42 |

Gaps are mostly 3 or 6 months, so a single fixed transition matrix `A` mixes two different
carry-over strengths. The briefing's Eq. (2) has no `Δt` term, although its own figure shows one.

### 3.4 Missingness follows the instrument schedule

Questionnaires (`SCOPA_AUT`, `GDS`, `STAI`, `ESS`, `RBDSQ`) are ~33% missing, `MCATOT` 47.7%
(annual visits only), Part II / Part I totals / LEDD ~8%. Consequences:

- Pairs where **all 26 features are observed at both visits: 1,645** of 6,483 (25%).
- Labelled visits with all 26 features at that visit: 3,024 of 5,638.
- Many `X_{t-1}` values are really "last observed 6–12 months ago", so irregular spacing exists
  at the **feature level**, not only the visit level.
- Fitting 676 entries of `A` plus 351 of `Σ` from 1,645 complete pairs is not safe.
  Regularisation and a missing-data strategy are required, not optional.

---

## 4. Models: inventory and quality

Metrics are balanced accuracy unless stated (3-class chance 0.333, binary chance 0.500).
All models use the 26 cheap features and patient-grouped CV unless stated.

### 4.1 Subtype classifiers (one visit in, subtype out)

| # | Model | 3-class | Binary TD vs PIGD | Verdict |
|---|---|---|---|---|
| 1 | Majority class | 0.333 | 0.500 | floor |
| 2 | Logistic (L2) | 0.461 | 0.694 | best CV score, but see row 5 |
| 3 | HistGradientBoosting | 0.452 | 0.689 | about the same as logistic |
| 4 | **XGBoost** (the one persisted) | 0.450 baseline visit / 0.462 all visits | 0.695 | about the same as logistic |
| 5 | 12-config bake-off (xgb depth/lr grid, missingness flags, tuned threshold, LightGBM, formula-aware two-head) | n/a | 0.669 – 0.693 | **indistinguishable**: spread 0.024 is smaller than the per-fold sd of 0.031 |
| 6 | Sporadic PD, first visit only | n/a | logistic 0.553 / HGB 0.530 | chance; **out of scope** (longitudinal-only decision) |

Locked test (88 patients, scored once): **binary balanced accuracy 0.693, 95% CI [0.647, 0.738];
AUROC 0.753, CI [0.698, 0.804]**. Significantly above chance, modestly so. The 3-class model
cannot separate Indeterminate (a 0.90–1.15 ratio band whose label flips 76% of the time at the
next visit).

Where the signal comes from (ablation, binary, `reports/metrics/ablation_results.txt`): full 26
features 0.683; `GENETIC_COHORT` alone 0.630 (now removed from X); only the two axial items
0.636; dropping medication features costs 0.019. `NP2RISE` and `NP2TURN` are plausible proxy
leakage, and the model reads LEDD dose as tremor information (FCX C1).

**Persistence gap:** `models/conformal_model_production.pkl` stores only the **3-class** XGBoost
(`model_3c`). The binary model, the only one with usable signal, is evaluated but **not saved**.

### 4.2 Conformal layer

| Task | Method | Coverage (target 90%) | Singleton sets |
|---|---|---|---|
| Binary | LAC | 89.4% | 47.4% |
| Binary | APS (bug fixed) | 89.4% | 39.3% |
| 3-class | LAC | 89.9% | **10.6%** |
| 3-class | APS (bug fixed) | 90.0% | 2.1% |

Calibrated correctly after the 30 Sep APS fix. Caveats: coverage is marginal only (84.8% on
label-fragile visits against 94.2% on robust ones); ~17 correlated visits per patient strain
exchangeability; the 3-class sets are uninformative (half are size 3).

### 4.3 Transition models

| Model | Inputs | Target | AUROC | Brier vs base-rate Brier | Deployable? |
|---|---|---|---|---|---|
| Cheap transition (production) | 26 + 8 deltas + out-of-fold upstream probabilities | next-visit flip | **0.533** | 0.2083 vs 0.2047 (**worse**) | yes, and useless |
| same | same | flip within 12 months | **0.564** | 0.2444 vs 0.2433 (**worse**) | yes, and useless |
| E2 | + exam at visit *t* | any flip | 0.775 | 0.163 vs 0.205 | **no** (needs the exam) |
| **E5** | + exam + previous-visit history | any flip | **0.802** (locked 0.803, CI [0.771, 0.833]) | 0.155 vs 0.205 | **no** |
| E6 | E5 features | flip within 12 months | 0.795 (locked 0.764) | 0.180 vs 0.243 | **no** |

The two leaks removed on 30 Sep (the APS bug and in-sample upstream probabilities) took the
cheap model from AUROC 0.65 to 0.53. Its top features are now `PDTRTMNT`, `SEX`, `HANDED`,
which is noise. **Anything "explaining" this model is explaining nothing.** Also, 80–85% of
modelled flip risk in E5/E6 is proximity to a cutoff, i.e. label wobble, not disease change
(FCX C3).

### 4.4 Per-round wide models (C1 rounds)

Logistic on `26 × visits` features, target TD/PIGD at month 24.

| Round | Visits | n | CV bal. acc | Locked bal. acc |
|---|---|---|---|---|
| R1 | BL | 328 | 0.683 | 0.664 |
| R2 | BL+6 | 267 | 0.608 | 0.694 |
| R3 | BL+6+12 | 248 | 0.631 | 0.634 |
| R4 | BL+6+12+24 | 248 | 0.615 | 0.644 |
| A3 | BL+12+24 | 307 | 0.628 | 0.574 |

**More visits did not help; the best round is the one with the fewest visits.** No "confidence
curve" emerged. R4 and A3 include the month-24 intake, so they are not forecasts.

### 4.5 FCX explainer models (small heads, not black boxes)

| Component | Explains | Result | What failed |
|---|---|---|---|
| C1 implied exam | subtype model | fidelity 0.70–0.90 (by setting); gait alignment r 0.57, tremor 0.25 | per-case error attribution is at chance |
| C2 cutoff straddle | conformal sets | fidelity κ 0.61; a tremor exam collapses 95% of ambiguous sets | per-patient blame loses to the rule "always examine tremor" (90% vs 94%) |
| C3 proximity-drift-noise | transition risk | R² 0.51–0.62 | **never beat grouped Shapley** at noise-vs-drift; went below chance on the stronger E5 model |

### 4.6 Count

Distinct learners trained and scored: **majority, logistic, HGB, XGBoost, LightGBM, the
formula-aware two-head model, four transition variants (cheap, E2, E5, E6), per-round logistic
×5, and the FCX heads.** Persisted artefacts: **one 3-class XGBoost with two LAC/APS
calibrators, and two cheap transition XGBoosts** (`models/*.pkl`, regenerate with
`make conformal transition`). Everything else is evaluated and discarded.

---

## 5. Explainability already in the repo — FCX

FCX explains the three models in the Stebbins formula's own coordinates (`ℓ = log T − log P`,
two cutoffs, tremor channel vs gait channel). Its Shapley code is
[`explain/shapley.py`](src/trace_pd/explain/shapley.py):

```python
shapley_mc(f, X, B, n_perm=32, groups=None, rng=None)
# per permutation: draw ONE background row b per instance, then overwrite columns of b with x
# as features are revealed. Interventional value function; efficiency holds exactly.
```

**How AR(1)-SHAP relates to this is a decision for the team.** Three options:

1. **Replace FCX as the paper's contribution.** Cleanest story, discards the C1–C3 work.
2. **Add AR(1)-SHAP as a background sampler inside FCX** (recommended). FCX's C1 channel split
   `φ_j(ℓ̂) = φ_j(ĥ_T) − φ_j(ĥ_P)` stays exact for any value function that is linear in `f`, and
   `v_AR(1)` is linear in `f`. FCX then becomes the application framework and AR(1)-SHAP the
   method.
3. **Two separate papers.**

**`shapley_mc` cannot be reused as-is.** It fixes one background row per permutation and keeps
those hidden values while features are revealed. A *conditional* value function needs the hidden
block re-drawn from `P(X_S̄ | X_S, X_{t-1})` **at every coalition step**, because the conditioning
set changes. The permutation loop needs a per-step sampler hook.

---

## 6. What the briefing specifies, against what the repo has

| | Briefing (`AR1-SHAP_Technical_Briefing.pdf`) | In the repo today |
|---|---|---|
| Transition model | `X_t \| X_{t-1} ~ N(A·X_{t-1} + b, Σ)`, full `A`, full `Σ` | nothing |
| Value function | `v(S) = E_{X_S̄ ~ P(X_S̄ \| X_S, X_{t-1})}[f(X_S, X_S̄)]` | interventional only |
| Irregular spacing | named as the gap being filled, but `Δt` is absent from Eq. (2) | `MONTHS_SINCE_PREV` exists in `trajectory.py` |
| Pair table `(X_{t-1}, X_t, Δt)` | implied | `trajectory.build()` shifts a few quantities, not all 26 features |
| Baseline | standard SHAP | `shapley_mc` (interventional) and XGBoost `pred_contribs` (TreeSHAP) |
| Explained model | "model predicts a TD → PIGD shift at visit *t*" | undefined, see §8 |
| Comparison library | none | no `shap` package installed |

**A semantic change the briefing does not state.** With the conditional value function,
`v(∅) = E[f(X_t) | X_{t-1}]`, so the efficiency identity becomes
`Σφ = f(x_t) − E[f(X_t) | X_{t-1}]`. The attributions explain how the prediction **differs from
what the last visit predicted**, not how it differs from the cohort average. That is exactly
why "acute change" features rise and "clock" features fall. It must be stated in the paper:
part of the difference from standard SHAP is a changed *question*, not a corrected *answer*.

---

## 7. Checks on the briefing's claims

| Claim in the briefing | Result |
|---|---|
| Standard SHAP's top driver is "Years-since-diagnosis" | **Not reproduced globally.** Mean \|TreeSHAP\| on the repo's XGBoost config. Binary: `LEDD_TOTAL_MG` 11.1%, `NP2RISE` 10.6%, `SEX` 10.5%, then `YRS_SINCE_DIAGNOSIS` 8.0% (rank 4). 3-class: `LEDD` 9.9%, `SEX` 8.4%, `YRS_SINCE_DIAGNOSIS` 8.0% (rank 3). Clocks are 3rd–5th, not first. This is a global in-sample ranking, so a **per-visit** check is still open. **[recomputed]** |
| Mean autocorrelation 0.737; 47.7% within-patient variance | Recomputed as **0.754** and **46.9%** (mean over 26 features, lag-1 on consecutive rows). Close, but the paper should cite one script. |
| "Four models stopped at ~0.46" | True for **3-class**. Binary is ~0.69; say which. |
| 6,922 visits | Only **5,638** are labelled, and **fewer** form usable pairs. |
| "Sleep and mood drop" is the acute driver | An illustration, not a result. Nothing in the repo shows it. |
| "No method explains tree models on sparse irregular visits with same-patient structure" | **Unverified.** Nearest prior art to check before submission: Aas, Jullum & Løland 2021 (Gaussian conditional Shapley), Frye et al. 2021 (Shapley on the data manifold), TimeSHAP, Dynamask. A needed ablation is plain Gaussian conditioning *without* the lag term. |
| Attribution shift as an early warning of the next flip | Tried in an earlier pilot (not committed): summed SHAP shift gave AUROC **0.52** on 4,481 pairs, against 0.60 for the top-2 probability margin. Do not re-propose without a refinement. |

---

## 8. Open decisions

| # | Decision | Why it blocks | Recommendation |
|---|---|---|---|
| D1 | **Which model is explained?** | The cheap transition model is at chance, so explaining "shift drivers" there explains noise. E5 has signal (0.80) but is not cheap-feature and not persisted. The persisted 3-class model scores 0.46. | Primary: the **binary subtype classifier** (0.69, significant on the locked test), which needs persisting. Secondary stress test: **E5**, stating plainly that it uses the exam. |
| D2 | **What is a "shift" being explained?** | Needs a definition: change in predicted probability between `t-1` and `t`, or the prediction at `t` against `E[f \| X_{t-1}]`. | Use `f(x_t) − E[f(X_t) \| X_{t-1}]`; it falls out of Eq. (3) with no extra definition. |
| D3 | **Feature treatment** | §3.1: static, clock, binary, ordinal, count, heavy-tailed. | Hold static features fixed and clock features deterministic (`x_{t-1} + Δt`); keep both out of the stochastic block; model the rest with a latent Gaussian plus threshold/rounding or a Gaussian copula; log-transform LEDD. |
| D4 | **Missing data in `X_{t-1}` and `X_S`** | Only 25% of pairs are fully observed. | Gaussian marginalisation over the observed dimensions (exact), plus last-observation carry-forward with a feature-level age that widens the variance. |
| D5 | **Relationship to FCX** | §5. | Option 2: a sampler inside FCX. |
| D6 | **Δt model** | Eq. (2) is gap-blind. | `A(Δt) = exp(M·Δt)` with the matching Ornstein–Uhlenbeck `Σ(Δt)`; fit `M` by pairwise likelihood over the observed gaps. Build the fixed-`A` version first as the ablation. |

---

## 9. Work packages

Staged so each step produces something testable. File names are proposals.

| WP | Deliverable | Files | Acceptance test |
|---|---|---|---|
| **0** | Resolve D1–D6 and record them in one design note | `docs/ar1_shap_design.md` | team sign-off |
| **1** | Pair table: one row per consecutive scheduled-visit pair with `X_{t-1}`, `X_t`, `Δt`, observed-masks and the per-feature age of the last observation, for all 26 features | `src/trace_pd/data/pairs.py` (or extend `explain/trajectory.py`) | counts match §3 (6,483 pairs, gap histogram); no pair silently spans an unlabelled gap |
| **2** | Persist the binary subtype XGBoost and the exact patient-fold assignment | extend `models/conformal.py` | pickle loads and reproduces the binary CV score |
| **3** | Transition model `A, b, Σ` with ridge / Ledoit-Wolf shrinkage and missing-data handling, **Δt-free first** | `src/trace_pd/explain/ar1.py` | patient-grouped held-out log-likelihood beats (i) the population marginal and (ii) the per-feature scalar AR; locked-test patients excluded from fitting |
| **4** | `Δt`-aware version (D6) | same | held-out log-likelihood improves on WP3, checked separately on 3-month and 6-month pairs |
| **5** | Conditional sampler: Gaussian conditioning of the hidden block given `X_S` and `μ = A(Δt)x_{t-1} + b`; discrete and bounded handling; static features never perturbed | `src/trace_pd/explain/conditional_shapley.py` | samples stay inside feature bounds and integer support; static columns unchanged |
| **6** | Permutation Shapley with a per-coalition resampling hook; interventional, Gaussian-conditional (no lag) and AR(1)-conditional samplers behind one interface | same | efficiency `Σφ = f(x) − v(∅)` to tolerance for all three; lag coefficient 0 reduces AR(1) to the plain conditional; diagonal `Σ` reduces it to interventional |
| **7** | Evaluation harness (§10) | `src/trace_pd/evaluation/evaluate_ar1_shap.py` writing `reports/metrics/ar1_shap_results.txt`, logging every run to `experiment_log.csv` | reproducible from `make ar1shap` |
| **8** | Tests | `tests/test_ar1_shap.py` | the WP6 reductions, bounds, Δt monotonicity (a wider gap gives a wider conditional variance), no use of locked-test patients |
| **9** | FCX integration (option 2) | `explain/implied_exam.py` calls the new sampler | the C1 channel split stays additive (existing test in `tests/test_fcx.py` still passes) |
| **10** | Claims table for the paper: validated / not validated | `docs/` | every claim traces to a metrics file |

Add a `make ar1shap` target and a new `step` id in `experiment_log.csv` so results follow the
repo's append-only convention.

---

## 10. What would count as evidence

The briefing's central claim, "points to what a clinician can act on", has **no ground truth**
in this data. A reviewer will ask how it is validated. Proposed tests, strongest first:

1. **Synthetic ground truth.** Simulate data with known AR dynamics and a known model, where
   the true conditional Shapley values are computable. The sampler must recover them. **Do not
   plant the answer using the method's own components.** That flaw is what withdrew FCX C3's
   "3/3 recovered" (`docs/results_session_final.md` §1, bug 5).
2. **Off-manifold evidence.** Measure how far the coalition inputs sit from real data
   (Mahalanobis distance under the AR model, nearest real-visit distance, and the model's output
   variance) for interventional versus AR(1) draws. This tests the "Frankenstein patient" claim
   directly.
3. **Faithfulness under realistic perturbation.** Deletion and insertion curves where removed
   features are replaced by AR(1) draws rather than population draws.
4. **Stability.** Attribution consistency across seeds and across adjacent visits of one patient.
5. **Divergence from standard SHAP**, reported as top-1 flips and rank correlation, **split by
   per-feature autocorrelation** (§3.2). An earlier prototype (per-feature scalar AR, not
   committed) showed 65% top-1 flips and r = 0.64 between the AR coefficient and the attribution
   change. Treat that as a lower-bound sanity check only.

Required ablations: no lag term (plain Gaussian conditional); per-feature scalar AR versus full
multivariate; fixed `A` versus `Δt`-aware `A`; sensitivity to the shrinkage strength.

---

## 11. Risks and known flaws

1. **Conditional Shapley answers a different question.** It can give credit to a feature the
   model never uses if that feature is correlated with one it does (Janzing et al. 2020;
   Sundararajan & Najmi 2020; Chen et al. 2020, "true to the model or true to the data").
   Conditioning on `X_{t-1}` reduces this and does not remove it. Include a dummy-feature test:
   a feature the model ignores must get about 0, or the deviation must be reported.
2. **The model being explained is weak.** At 0.69 binary, explanations describe a model that is
   only modestly right. Say so, and do not present attributions as disease mechanisms.
3. **Label wobble.** 39.5% of labelled visits are one item-point from a different label, and
   80–85% of modelled flip risk is proximity to a cutoff. A "driver of the subtype shift" may
   be a driver of scoring noise.
4. **Medication state.** LEDD is a top driver, and 25.6% of same-day OFF/ON exams give a
   different label. An explanation that surfaces LEDD may be surfacing the medication artefact.
5. **`SEX` is a top-3 feature in both subtype models** (10.5% binary). It is static and possibly
   a recruitment artefact. AR(1) holds it fixed, so it is a natural showcase case and also a
   confound to explain.
6. **Parameter count against data.** 1,053 parameters (26 features) from 1,645 complete pairs.
   Drop to the ~21 stochastic features and shrink, or the transition matrix is noise.
7. **Gaussian misspecification** (§3.1) on 20 of the 21 stochastic features.
8. **Single cohort.** PPMI only: early-stage patients from specialist centres. No external
   validation.
9. **Novelty is unverified** (§7). "Conditioning on the lag" is close to existing Gaussian
   conditional Shapley. The defensible contribution is the temporal anchor, irregular gaps, tree
   ensembles on sparse clinical visits and a validation protocol, and only if the ablations show
   the lag term matters.

---

## 12. Which documents to trust

| Document | Status |
|---|---|
| `reports/metrics/*` | source of truth for numbers |
| `docs/results_session_final.md` | current (30 Sep). It cites 321 and 311 runs; the log has **331** rows. |
| `docs/fcx.md` | current, post-audit; read §5 before quoting |
| `README.md` §9.3, §9.5, §9.6 | **stale**, still marked ⏳ as pre-fix. It says ablation full = 0.690 and transition AUROC 0.647; the current files say 0.683 and 0.533. |
| `HANDOFF_README.md` | **stale** (documented in `CLAUDE.md`) |
| `configs/config.yaml` | documentation only; editing it changes nothing |

---

## 13. Running things

`trace_pd` is not installed; use `PYTHONPATH=src` or `pip install -e .`.

```bash
make preprocess      # raw PPMI → data/processed/ (deterministic)
make conformal       # trains and saves models/conformal_model_production.pkl
make transition      # run AFTER conformal
make fcx c3          # FCX evaluation
make test            # PYTHONPATH=src python -m pytest tests -q  →  39 passed, 2 skipped (1 Oct)
```

Environment seen on 1 Oct: Python 3.10.11, xgboost 3.1.2, scikit-learn 1.7.2, numpy 2.2.6,
pandas 2.3.2. `shap` and `lightgbm` are **not** installed; `shapley_mc` is the in-repo baseline.
Set `git config core.autocrlf true` to avoid CRLF-only diffs. The `data/` folder is
DUA-restricted PPMI data: keep the repository private and copy nothing out of it.
