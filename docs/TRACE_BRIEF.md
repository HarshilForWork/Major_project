# CLAUDE.md: TRACE implementation brief

> Stored in the repo on 14 Oct 2026 as the governing brief for the TRACE implementation
> (the user supplied it as a `CLAUDE.md`; the repo's own `CLAUDE.md` is kept and points here).
> Progress log: `results/PROGRESS.md`. Discovery: `results/DISCOVERY.md`.

Read this whole file before writing any code. Work stage by stage. Stop at every gate. If something is unclear (file locations, feature groups, existing code interfaces), ask me. Do not guess.

Keep a short running log in `results/PROGRESS.md` (what you did, what passed, what failed, what you need from me).

---

## 1. What this project is, in plain words

We have an XGBoost model that scores **one clinic visit at a time** and outputs a risk score (raw log-odds). The same model scores every visit of a patient, so each patient has a risk score that moves over time.

We want to explain **why the score changed between two visits**: which input features pushed it up or down.

The standard way to do this is **Baseline Shapley**: take visit r's values, swap features one at a time to visit t's values, and split the change fairly among the features by averaging over all swap orders. Call the credit of feature j for going from visit r to visit t `phi_j(r->t)`.

**The problem:** these pairwise explanations contradict each other. For one patient, `phi_j(1->3)` is usually not equal to `phi_j(1->2) + phi_j(2->3)`. This happens because features interact, and every tree model of depth 2 or more has interactions.

**TRACE** fixes this. For each feature, average its pairwise credits over every visit used as the starting point. This gives one "tally" per feature per visit, and the explanation of any change is the difference of two tallies. The result:

- every route between two visits gives the same answer (no contradictions);
- credits still add up exactly to the risk change;
- it is the least-squares closest consistent answer to all the pairwise explanations;
- it is exactly equal to interventional TreeSHAP with the patient's own visits as the reference set, so it is computed exactly with existing tools.

**Data:**
- PPMI (Parkinson's): 439 patients, 6,922 visits, median 17 visits each. Task: motor subtype TD vs PIGD. Model balanced accuracy about 0.69.
- ADNI (Alzheimer's): 4,470 patients, 24,302 visits, median 3 visits each. Task: cognitively normal vs dementia. The main model is **without MMSE and FAQ** (balanced accuracy about 0.93). The full model (about 0.98) is a sensitivity analysis only.

**Pilot already done (on locked-test patients with 3 or more visits):** PPMI 88 patients, ADNI 475 patients (median 8 visits). Add-up error about 1e-5. Step-by-step vs direct explanations disagree by 40-52% and name a different top cause for 19-31% of patients. Direct pairwise comparisons themselves have only 1.5-3% contradiction. So the real problem is step-by-step accumulation.

---

## 2. Hard rules (never break these)

1. **No synthetic or simulated data anywhere.** No toy fixtures, no simulated patients, no injecting artificial signal into real visits. All tests and experiments run on the real preprocessed cohorts. Do not create toy test cases.
2. **Never modify the locked preprocessed data, the locked splits, or the locked test set.** Read only.
3. **Patient-level splits only.** All explanation results are computed on **held-out test patients**.
4. **ADNI main results use the model without MMSE and FAQ.** The full model appears only as a labelled sensitivity analysis. Check whether any other ADNI feature is directly used to define the diagnosis and tell me; do not remove anything without asking.
5. **Missing values stay missing.** No imputation, no LOCF, no mean filling. NaN is treated as a value and routed by XGBoost's learned default direction.
6. **Work on the raw margin (log-odds) scale**, not probabilities.
7. **Use interventional TreeSHAP only.** Do not use XGBoost's built-in `pred_contribs` or path-dependent TreeSHAP for TRACE. Path-dependent TreeSHAP is allowed only as a labelled baseline.
8. **No patient identifiers in any output.** Use anonymous integer indices. Figures and reports must not show patient IDs, dates, ages or raw feature values. Per-patient arrays stay in local gitignored files.
9. **Do not promise or hard-code success thresholds for the comparison metrics** (faithfulness, stability, baselines). Report what you measure, with confidence intervals. Only the exactness checks in Stage 3 have fixed pass/fail tolerances.
10. **Never loosen a tolerance to make a check pass.** If a check fails, stop, debug, and tell me.
11. **Claim wording:** TRACE explains what the *model* is reacting to in each patient's data. It does not explain disease causes or mechanisms. Reports must not say it explains disease progression. Medication features can look important simply because sicker patients get more medication.

---

## 3. The maths you must implement

Notation: features `j = 1..d`; one patient has visits `x^1..x^T`; `f(x)` is the XGBoost raw margin (NaN allowed in x).

**Pairwise change explanation (Baseline Shapley).** For visits r, t and a feature group S:

```
v_{r->t}(S) = f( x^t on S , x^r off S ) - f(x^r)
phi_j(r->t) = sum over S not containing j of  |S|! (d-|S|-1)! / d!  *  [ v(S + {j}) - v(S) ]
```

- Players are only the features whose values differ between r and t (treat NaN == NaN as equal; NaN vs a number is a difference). Features with equal values get exactly 0.
- Properties: sum_j phi_j(r->t) = f(x^t) - f(x^r); phi_j(t->r) = -phi_j(r->t); equal value at both visits gives 0.

**How to compute it.** For each reference visit r, build `shap.TreeExplainer(model, data=<row x^r only>, feature_perturbation="interventional", model_output="raw")` and explain all T visits of the patient. Store the result as `F[r, t, j] = phi_j(r->t)`, with `F[t, t, :] = 0`. Use float64 and store per-patient `.npz`.

**TRACE.**

```
u[t, j]       = (1/T) * sum over r of F[r, t, j]          (the "tally", includes r = t which is 0)
psi[s->t, j]  = u[t, j] - u[s, j]                          (the consistent explanation)
R[r, t, j]    = F[r, t, j] - psi[r->t, j]                  (the contradiction removed)
I             = sum_j sum_{r<t} R^2  /  sum_j sum_{r<t} F^2        (trust score, between 0 and 1)
```

**Identities that must hold:**
- `sum_j psi[s->t, j] = f(x^t) - f(x^s)` (add up)
- `psi[s->t] + psi[t->q] = psi[s->q]` (no contradictions)
- `f(x^t) = fbar + sum_j u[t, j]` where `fbar = mean over r of f(x^r)` (full trajectory decomposition)
- `u[t, :]` equals interventional TreeSHAP of visit t with the **whole set of the patient's T visits as the background data** (use this as an independent cross-check of your `F`-based computation)
- `u` is the least-squares projection: minimise `sum_{r<t} ( F[r,t,j] - (u(t) - u(r)) )^2` for each feature j (gauge: u sums to 0 over visits).
- A feature with the same value at all visits of this patient (for example sex) has `u = 0` exactly. A feature the model never splits on has `u = 0` exactly.
- With T = 2, TRACE equals Baseline Shapley.

**What is NOT guaranteed (never claim these):**
- A feature unchanged between visits s and t can still get nonzero `psi[s->t]` (it may have changed at other visits). This is a proven impossibility (no method can be exact, contradiction-free and always give 0 to an unchanged feature). Measure and report how often this happens.
- `psi[s->t]` depends on the patient's other visits, so adding a visit revises earlier explanations by a small amount.
- Hybrid inputs can be clinically odd combinations, even though every value is one the patient really had.

---

## 4. Stages

### Stage 0: Discovery and setup (no analysis yet)
1. Explore the repo. Find: trained models, preprocessed data, locked splits, feature lists, existing pilot code, existing AR(1)-SHAP and Asymmetric Shapley code, and any existing feature groupings.
2. Write `results/DISCOVERY.md` listing what you found and what is missing.
3. Create `config.yaml`: random seeds, tolerances, `k_values: [1,2,3,4,5]`, `n_bootstrap: 1000`, `min_visits: 3`, `n_background_population: 500`, XGBoost margin decision threshold.
4. Log package versions (shap, xgboost, numpy). Fix all seeds. Results must be reproducible.
5. **Ask me** for anything missing before moving on. In particular I will need to confirm: the feature groups (see Stage 9) and the decision threshold used for model-predicted subtype labels.

### Stage 1: Models and cohorts
1. Load the locked PPMI model. For ADNI, load the model without MMSE and FAQ (`M_minus`). If it is not saved, retrain it on the **same locked train split with the same hyperparameters and no tuning on test**, then save it. Also load the full ADNI model (`M_full`) for the sensitivity analysis.
2. Check how XGBoost handles `best_iteration` (use it consistently in `predict`) and confirm `base_score` handling.
3. Select analysis patients: test patients with at least 3 visits. Print counts per cohort. Check against the pilot (PPMI 88, ADNI 475). If counts differ, report it.
4. Report model balanced accuracy on the locked test set for each model (PPMI, M_minus, M_full) as a sanity line in the report.

### Stage 2: Core TRACE engine
Implement as a small, clean module (for example `src/trace/engine.py`):
1. `pairwise_F(model, X_patient)` returns `F[T,T,d]` using interventional TreeSHAP as described in Section 3.
2. `trace_from_F(F)` returns `u`, `psi`, `R`, `I`.
3. `trace_via_background(model, X_patient)` computes `u` by passing all T visits as background (independent cross-check).
4. `trace_via_lstsq(F)` computes the projection with a numpy least-squares solve on the visit graph (independent check of the closed form).
5. Store per-patient results in local gitignored files. Process patients in parallel if easy.

### Stage 3: GATE. Exactness checks (must pass at all costs)
Write these as `pytest` tests (`test_gate_*`) that run on real locked-test patients, for PPMI, ADNI `M_minus` and ADNI `M_full`. **If any of them fails, stop everything, debug, and report to me. Do not continue to Stage 4.**

| # | Check | Definition | Tolerance |
|---|---|---|---|
| G1 | Add-up error | max over all patients and pairs of `abs(sum_j psi[s->t,j] - (f(x^t) - f(x^s)))`; also max of `abs(f(x^t) - fbar - sum_j u[t,j])` | <= 1e-5 |
| G2 | Consistency error | max over triples of `abs(psi[s->t] + psi[t->q] - psi[s->q])`; also max difference between the closed form `u` and `trace_via_lstsq`, and between `trace_from_F` and `trace_via_background` | <= 1e-5 |
| G3 | Zero checks | (a) features whose value (NaN-aware) is identical at all visits of a patient have `u = 0`; (b) features with zero splits in the model have `u = 0` | max abs <= 1e-6 (theory says exactly 0; tolerance covers float rounding). Report the actual max. |
| G4 | Brute-force match | For a random stratified sample of at least 2,000 pairs per cohort where the number of differing features `k <= 12`: enumerate all `2^k` hybrid visits, get margins via `booster.predict(DMatrix(..., missing=np.nan), output_margin=True)`, compute exact Shapley values, compare with `F[r,t,:]` | <= 1e-5 (expect about 1e-6); report actual max |

For G4:
- Make sure the sample includes pairs with NaN in the reference visit, in the explicand visit, and in both.
- Also check antisymmetry `F[r,t] = -F[t,r]` on the same sample.
- If G4 fails only on NaN pairs, the installed `shap` version probably has a missing-value routing bug. Record the version, report it to me, and do not work around it silently.

### Stage 4: Size of the problem (metrics 5 and 6)
First, **reproduce the pilot numbers** (step-by-step vs direct disagreement, top-cause change rates). If the pilot code defines "total disagreement" and "clear winners", reuse those definitions exactly. If not, ask me. Report any difference from the pilot.

For each patient with T >= 3, for the pair (first visit, last visit), and also over all pairs (s,t) with at least one visit between them:
- `direct = phi(s->t)`, `chain = sum_{k=s}^{t-1} phi(k->k+1)`.
- **Top-cause disagreement rate:** share of patients where the top feature (largest absolute credit) differs between `chain` and `direct`.
- **Total disagreement:** `sum_j abs(chain_j - direct_j) / sum_j abs(direct_j)`.
- **TRACE change rate:** share of pairs where the top feature of `psi[s->t]` differs from the top feature of `direct`.
- **Seed noise baseline:** retrain the model with N different seeds (use 10). First check whether the seed actually changes the model (with `subsample=1` and `colsample=1`, XGBoost can be deterministic). If it does not, use patient-level bootstrap resamples of the **training** set instead. Compute how often the top feature of `direct` itself changes between two retrained models. The disagreement rate from chaining should be compared against this noise.
- **Trust score `I`:** distribution (median, quartiles, histogram) per cohort and by number of visits T. Also check whether `I` is higher for patients with larger chain-vs-direct disagreement.
- Also report how often a feature **unchanged between s and t** gets large `abs(psi[s->t])` (the price of the impossibility result), in both cohorts.

### Stage 5: Baselines
Implement each as a function that returns a change explanation `A(s->t)` (vector over features) for any pair of visits of a patient:

| Baseline | Definition |
|---|---|
| Direct pairwise | `phi(s->t)` (Baseline Shapley; wins on its own pair by construction, so it is the upper reference) |
| Chaining (DeltaSHAP-style) | `sum_{k=s}^{t-1} phi(k->k+1)` (call it "DeltaSHAP-style"; the original DeltaSHAP is for LSTMs with a forward-fill reference) |
| **Fixed first-visit reference** | `phi(1->t) - phi(1->s)` (consistent and exact; the simplest competing fix). Also build the fixed last-visit version `phi(T->t) - phi(T->s)` |
| Population TreeSHAP difference | interventional TreeSHAP with a fixed random background of `n_background_population` training visits (fixed seed): `phi_pop(t) - phi_pop(s)` |
| Path-dependent TreeSHAP difference | default `shap.TreeExplainer`: `phi(t) - phi(s)` |
| Conditional AR(1) SHAP | use the team's existing implementation, as a difference of per-visit attributions. Does not sum exactly to the risk change, so report its efficiency gap |
| Asymmetric Shapley | use the team's existing implementation if there is one (static features before dynamic). If not, tell me; do not invent one |
| FA-SHAP (optional, internal) | only if forecasts already exist in the repo |
| DynSHAP, C-SHAP | out of scope (they explain sequence models, not this object). Do not implement |

For every baseline, report its **efficiency gap**: `abs(sum_j A_j(s->t) - (f(x^t) - f(x^s)))`.

### Stage 6: Faithfulness (metric 7)
Question: does the explanation point at the features that really move the model's score?

For every held-out pair `(s,t)` with `s < t` and every method:
1. Rank features by `abs(A_j(s->t))`. Break ties by feature index.
2. For `k = 1..5`: revert the top-k features of visit t to their values from visit s (NaN copied too, **real values only**, nothing interpolated) and compute `f` of the reverted visit.
3. `recovered_k = (f(x^t) - f(reverted)) / (f(x^t) - f(x^s))`.
4. Exclude pairs with `abs(f(x^t) - f(x^s)) < eps` (propose `eps = 0.05` log-odds; confirm with me) and report how many pairs were excluded.
5. Average over pairs within each patient first, then over patients (so long-history patients do not dominate). Report a patient-level bootstrap 95% CI.
6. Report paired differences against each baseline with bootstrap CIs.

Notes:
- Plain pairwise Baseline Shapley will win on its own single pair by construction. The claim for TRACE is **good on all pairs of a patient at once**, not best on one pair. State this in the report.
- Reverting a feature that did not change does nothing, so methods that give credit to unchanged features are correctly penalised.

### Stage 7: Stability (metrics 8 and 9)
Question: how much does an explanation move when one visit is removed?

For each patient with T >= 4, each pair `(s,t)`, and each other visit `q` (not s or t): delete visit q, recompute every method, and measure `max_j abs(Delta A_j(s->t))` and the mean over j. **Use deletion only** (it uses real records only). Do not fabricate replacement visits.

Report **separately** for three cases of the deleted visit: (a) the first visit, (b) an intermediate visit, (c) the last visit.

- **TRACE:** check the bound `max_j abs(Delta psi_j) <= 4R/(T-1)`. Use `R = sum over trees of (max leaf - min leaf)`, a valid upper bound on the model's output range. A violation means a bug: stop and tell me.
- **Chaining:** compare. One noisy visit can change it by up to `4R` (no shrinking with T).
- **Fixed first-visit reference:** read this carefully. This baseline is **completely unaffected** by deleting an intermediate visit (it only depends on visit 1 and the pair), so TRACE cannot beat it in case (b). It is fully exposed when the **first visit itself** is the deleted or noisy one (case a). Report all three cases honestly. This is the most important comparison for the paper, so do not summarise it as a single number.
- **Retraining stability:** across the N retrained models from Stage 4, compute the Spearman correlation of explanations for the same pairs, per method.
- Report bootstrap CIs and paired differences. Do not state a winner without the CI.

### Stage 8: Sanity check (metric 10)
- Retrain the model on the locked train split with **patient-level shuffled labels** (same hyperparameters). It uses real visits with shuffled labels, so it is not synthetic data; I will flag this to my guide.
- Report its test balanced accuracy (expect about 0.5).
- Recompute TRACE tallies for the same test patients and report the Spearman correlation of per-feature mean `abs(u_j)` with the real model's. The explanations should fall apart.

### Stage 9: Clinical and descriptive analyses (no pass/fail values)
These are descriptive only. Do not interpret them causally.

1. **Feature groups.** Create `feature_groups.yaml` with groups such as medication, questionnaire, demographic/static. **Do not invent the grouping.** Propose a draft from the feature names and ask me to confirm it.
2. **PPMI model-predicted subtype flips.** A flip is a crossing of the model's decision threshold (converted to margin units) between consecutive visits. For each flip, take `psi[k->k+1]` and aggregate by feature and by group: mean share of credit, and sign. Question: do medication features dominate the flips? (Clinically they should not, per Simuni et al. 2016.) Also compare model-predicted flips with flips in the true labels (labels are not model inputs).
3. **Measurement accounting (optional).** Split each `u_j(t)` into the part from comparisons where the feature's missing/observed status differs and the part where both are observed. Report the share of prediction movement due to the first part, overall and by ADNI study phase if the phase column exists.
4. **ADNI within-person vs between-person.** With `phi_pop_j(t)` = interventional SHAP against the population background, define `b_j(t) = phi_pop_j(t) - u_j(t)`. Check numerically that `sum_j b_j(t) = fbar_patient - fbar_background` is the same at every visit of a patient (extra identity check, tolerance 1e-5). Report per-feature mean `abs(u_j)`, mean `abs(b_j)`, and the within-patient SD of `b_j(t)` over visits. Static features must have `u = 0`, so all of their population credit is between-patient.
5. **Headline patient.** Pick one real patient (anonymised) where chaining and direct name different top causes. Make a bar chart with the two routes side by side, then TRACE's single answer. Add a stacked "feature curves" plot of `u_j(t)` across visits that reproduces the patient's risk curve. No IDs, dates, ages or raw values on the figures.

### Stage 10: Reporting
Produce `results/REPORT.md` and CSV/JSON files for every table. The report must contain:
1. A **gate table** (G1-G4) with PASS or FAIL and the actual maximum errors, for PPMI, ADNI `M_minus` and ADNI `M_full`.
2. The **size-of-problem tables** and the trust score distribution.
3. **Faithfulness** curves for k = 1..5, all methods, with CIs.
4. **Stability** tables for the three deletion cases, all methods, plus the bound check and the retraining correlations.
5. The **sanity check** result.
6. The **descriptive clinical analyses** and the headline figure.
7. A **Decision checklist** (Section 6) filled in honestly.
8. A **Deviations and open questions** section: anything you changed from this brief, every number that differs from the pilot, and everything you need from me.

Rules for the report: show all numbers with CIs where they are estimates; do not write sentences stronger than the numbers support; do not describe the findings as explaining the disease.

---

## 5. Metrics: what each one is for and what good looks like

Rows 1-4 are hard gates and **must pass at all costs**. Rows 5-10 have no promised numeric thresholds. The "good" column is a direction I will agree with my guide before the runs.

| # | Metric | Why it is required | What good looks like |
|---|---|---|---|
| 1 | **Add-up error** (tallies vs risk change) | Shows the maths is implemented correctly and no credit is lost | <= 1e-5 (pilot already reached this) |
| 2 | **Consistency error** (1->2 plus 2->3 vs 1->3; closed form vs least squares vs background form) | Shows the contradictions are really gone and the formula is the least-squares fix | <= 1e-5 |
| 3 | **Zero checks** (never-changing features, unused features) | Shows the guarantees hold on real data | Zero (numerically <= 1e-6) |
| 4 | **Brute-force match** on small cases, including NaNs | Shows the TreeSHAP usage is correct, especially the missing-value handling | About 1e-6, hard limit 1e-5 |
| 5 | **Size of the problem** (chain vs direct top-cause and total disagreement; TRACE change rate) | Shows there is a real problem to fix | Clearly above the seed/bootstrap noise, with CI excluding the noise level |
| 6 | **Trust score `I` spread** | Shows the score is informative, not constant | Varies across patients; higher where chain-vs-direct disagreement is larger |
| 7 | **Faithfulness** (reverting top-k features recovers the change) | Shows the explanation points at what really moves the model | Not clearly worse than the best baseline when averaged over all pairs (CI of paired difference not entirely below zero) |
| 8 | **Stability** (delete one visit; retrain) | Tests the robustness claim and the 4R/T bound | TRACE moves less than chaining; bound never violated; reported separately for first, intermediate and last deleted visit |
| 9 | **Fixed first-visit reference baseline** | Tells us whether TRACE adds anything over the simplest consistent fix | An honest, case-by-case answer. Expect the fixed reference to be unaffected by intermediate deletions and exposed when the first visit is noisy. If it matches TRACE everywhere, the paper rests on the theory alone, and I will frame it that way |
| 10 | **Sanity check** (shuffled labels) | Shows explanations depend on real signal | Explanations fall apart (low correlation with the real model) |

**Descriptive only, no ideal value:** PPMI subtype-flip drivers (medication should not dominate), measurement-status share, ADNI within vs between-person split.

---

## 6. Decision checklist (am I ready to start writing?)

Start writing the paper only when all of these hold. Fill this in at the end of the report.

1. G1-G4 pass on PPMI, ADNI `M_minus` (and `M_full`).
2. The size-of-problem metric (row 5) is clearly above seed/bootstrap noise.
3. TRACE is more stable than chaining (row 8), with the paired CI excluding zero.
4. TRACE is not clearly worse than the best baseline on faithfulness (row 7).
5. We have an honest, written answer to the fixed first-visit reference question (row 9).
6. The sanity check (row 10) shows explanations collapsing under shuffled labels.

These decision rules are proposals. I will confirm them with my guide before the runs, so if they change I will tell you.

---

## 7. Pitfalls checklist

- `numpy ==` does not treat NaN as equal to NaN. Use a NaN-aware comparison everywhere, or Lemma "unchanged features get 0" will break.
- shap's handling of XGBoost missing-value routing has had bugs in some versions. G4 exists to catch this.
- Check `base_score` and `best_iteration` handling: `sum_j u[t,j]` must equal `f(x^t) - fbar` exactly.
- Do not use the model's probability output for explanations.
- The number of single-reference TreeSHAP calls per patient is `T^2`; keep results on disk, do not recompute.
- Do not report only the favourable comparison. Report all baselines, all deletion cases, all ks.
- If anything looks too good (for example an exact 0 where a small value is expected, or a perfect match with a baseline), assume a bug first.
