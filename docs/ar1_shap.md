# AR(1)-SHAP — patient-anchored Shapley values

*Built 1 Oct 2026 from `AR1-SHAP_Technical_Briefing.pdf`.*

| | |
|---|---|
| Code | `src/trace_pd/explain/ar1.py` (visit-to-visit model) · `src/trace_pd/explain/conditional_shapley.py` (explainer) |
| Evaluation | `src/trace_pd/evaluation/evaluate_ar1_shap.py` → `reports/metrics/ar1_shap_results.txt` |
| Tests | `tests/test_ar1_shap.py` (12) |
| Run | `make ar1shap` (≈40 min) · `PYTHONPATH=src python -m trace_pd.evaluation.evaluate_ar1_shap --quick` (≈2 min smoke test, writes nothing) |

---

## 1. The idea

Standard SHAP measures a feature's importance by hiding it and filling the gap with a value
copied from a random other visit, usually another patient's. For longitudinal data that builds
impossible patients: the gait score was 4 → 4 → 3 and the filler is a stranger's 0.

AR(1)-SHAP fills the gap with what is realistic **for this patient**, given what is shown at
this visit and what the patient looked like at the previous visit:

```
standard     v(S) = E[ f(x_S, X_hidden) ],   X_hidden ~ P(X_hidden)                     (Eq. 1)
AR(1)-SHAP   v(S) = E[ f(x_S, X_hidden) ],   X_hidden ~ P(X_hidden | x_S, x_{t-1}, Δt)    (Eq. 3)
```

**What changes in meaning.** The starting point `v(∅)` becomes `E[f(X_t) | x_{t-1}]`, the
prediction the patient's last visit implies for today. So the contributions add up to

```
Σ φ_j = f(x_t) − E[f(X_t) | x_{t-1}]
```

AR(1)-SHAP answers **"why does today's prediction differ from what we expected for this
patient?"**. Standard SHAP answers "why does this patient differ from the average patient?".
They are different questions. Some of the difference in the results below comes from that change
of question, not from standard SHAP being "wrong".

---

## 2. Design decisions

The briefing's model `X_t | X_{t-1} ~ N(A X_{t-1} + b, Σ)` does not fit the 26 features as
written. Each problem and the decision taken:

| # | Problem | Decision |
|---|---|---|
| D1 | Which model to explain? The cheap transition model has no signal (AUROC 0.53) | The **binary TD-vs-PIGD XGBoost**, the only subtype model with usable signal. Output: log-odds of PIGD. Trained on development patients only |
| D2 | What is being explained? | `f(x_t) − E[f(X_t) \| x_{t-1}]`, which falls straight out of Eq. 3 |
| D3a | `SEX`, `HANDED` never change; age and the two "years since" columns advance exactly with the calendar | **Fixed** and **clock** features are known exactly given the previous visit and the visit date, so they are never re-sampled. They receive zero credit by construction. A second version (`roles=False`) declares nothing and makes the model learn this, as a robustness check |
| D3b | 20 of the 21 changing features are 0–4 items, counts or yes/no | **Gaussian copula**: each feature is turned into a normal score by its rank, modelled there, and turned back through the empirical quantile. Every filled value is one a real patient has had |
| D4 | Missing data: only 25% of visit pairs have all 26 features | Previous values are **carried forward** from the last visit that recorded them (up to 12 months), with a "stale" flag; older values fall back to the population mean. Features missing *today* are not players: they stay missing and get no credit, under every method. Missing targets are left out of the fit exactly, never imputed |
| D6 | Visits are 3, 6 or 9+ months apart; Eq. 2 has no gap term | The carry-over and drift **depend on the gap** (interaction terms). Leftover spread is estimated separately for ~3, ~6 and ≥9-month gaps, sharing one correlation pattern: `Σ(Δt) = D(Δt) R D(Δt)` |
| — | ~1,000 parameters from ~5,000 pairs | **Ridge** on the mean model (strength picked by patient-grouped CV) and **shrinkage** of Σ toward its diagonal (picked by held-out likelihood) |
| D5 | Relation to FCX | Kept separate for now. The explainer takes any "filler", so FCX's C1 heads can use the AR(1) filler later (not done) |
| — | Estimator noise | The same Shapley loop for every method, so differences come from the filler, not the code. The same random numbers are reused along each feature ordering, which lowers noise |

The visit-to-visit model's mean is

```
u_t = b + A u_{t-1} + Δt·(A' u_{t-1}) + c·Δt + C z_t + (stale flags) + ε,   ε ~ N(0, Σ(Δt))
```

with `u` the copula scores of the 21 changing features and `z_t` today's fixed and clock values.

**Methods compared**, all with the same estimator (48 orderings × 24 draws per step):

| Name | Hidden features filled from |
|---|---|
| standard SHAP | a random complete development visit (Eq. 1) |
| population-conditional | the population's conditional given what is shown, with no history (Gaussian copula over all 26; the Aas et al. 2021 approach). **This isolates what the patient's history adds** |
| AR(1)-SHAP | this patient's next-visit model (Eq. 3) |
| AR(1)-SHAP, roles learned | same, with no feature declared fixed or clock |
| TreeSHAP | the default `shap` TreeExplainer algorithm, for reference |

All fitting uses the 351 development patients. Explanations are computed for the **88 locked-test
patients** (940 labelled TD/PIGD visits that have a previous visit), whom no fitted part has seen.

---

## 3. Results

Full output: `reports/metrics/ar1_shap_results.txt` (48 orderings × 24 draws, 940 visits, 88
patients, 32 min). Every run is also appended to `reports/metrics/experiment_log.csv` (step `ar1`).
The explained model scores 0.676 balanced accuracy on the locked test, the same level as the
project's 0.69.

### T1 — Does the patient's own past predict the next visit? ✅

Held-out log-likelihood on locked-test visit pairs (higher is better):

| Visit-to-visit model | Log-lik | Variance explained |
|---|---|---|
| population only (no history) | −1.123 | 0 |
| each feature's own previous value | −0.918 | 0.469 |
| all previous values, fixed gap | −0.849 | 0.499 |
| **all previous values + gap length (used)** | **−0.843** | 0.497 |

History explains about half of each next-visit value. Using all features beats one feature at a
time. Gap-awareness adds a little, most for ≥9-month gaps (−1.016 → −0.984). The leftover spread
grows with the gap (0.554 → 0.611 → 0.645), as it should.

### T2 — Is the estimator correct? ✅

Simulated patients with known dynamics and a linear model, so the exact AR(1)-SHAP value is
computable by enumerating every coalition.

| | Correlation with exact | Average error |
|---|---|---|
| AR(1)-SHAP, fitted A and Σ | **0.990** | 15.0% |
| same estimator, true A and Σ | 0.994 | 11.4% |

Most of the error is random sampling and shrinks with more samples; fitting the model costs about
3.6 points. **A known property, shown openly:** a feature the model never uses (weight 0) but that
is correlated with used ones gets true AR(1)-SHAP credit 0.211 (estimated 0.208). Standard SHAP
gives it 0. Conditional methods credit what is *informative*, not only what the model *reads*.

### T3 — Are the filled-in visits realistic? ✅ (the "Frankenstein patient" claim)

For each test visit, half the features are hidden and filled by each method. A classifier is then
trained to tell real visits from filled ones (0.5 = cannot tell).

| Method | Detectable, all 26 | Detectable, 21 changing | Jump ratio (1 = realistic) |
|---|---|---|---|
| standard SHAP | **0.976** | 0.861 | 2.31 |
| population-conditional | 0.959 | 0.822 | 1.84 |
| **AR(1)-SHAP** | **0.679** | **0.681** | **1.12** |
| AR(1)-SHAP, roles learned | 0.845 | 0.659 | 1.12 |

Standard SHAP's filled visits are almost always recognisable as fake, and their visit-to-visit
jumps are 2.3× real ones. Using correlations without history (population-conditional) helps a
little; **adding the patient's history is what fixes it**. The roles-learned version is easy to
spot on all 26 because its sampled ages wobble; on the changing features it is as realistic as
AR(1)-SHAP.

### T4 — How do the explanations change?

**Years since diagnosis** (rank 4 under standard SHAP, 9.5% of credit):

| Method | Raw credit vs standard | Share of credit vs standard | Rank |
|---|---|---|---|
| population-conditional | −24.8% | −32.5% | 4 |
| AR(1)-SHAP | −100% *(by design)* | −100% | 24 |
| **AR(1)-SHAP, roles learned** | **−86.8%** | **−74.0%** | 19 |

The −100% is built in: the method is *told* the clock advances with the calendar. The **−86.8%**
is the real finding. When the model has to learn on its own that age and disease duration just
tick forward, their credit still collapses. History removes most of the clock's credit; the
no-history conditional removes only a quarter.

**Credit by feature type:**

| | standard | pop-cond | AR(1) | AR(1) learned |
|---|---|---|---|---|
| fixed (sex, handedness) | 14.6% | 12.2% | 0% | 0.1% |
| clocks | 15.7% | 13.5% | 0% | 5.2% |
| changing features | 69.7% | 74.3% | 100% | 94.7% |

**Most common top driver per visit:** standard SHAP → `SEX` 26%, `YRS_SINCE_DIAGNOSIS` 18%.
AR(1)-SHAP → `NP2RISE` (getting out of a chair) 27%, `LEDD_TOTAL_MG` 14%. The top driver changes
on **72%** of visits, and on 57.7% even when only changing features are compared.

**Credit goes to what moved unexpectedly.** Per visit, the rank correlation between a feature's
credit and how surprising its value was given the last visit: standard +0.06, population +0.18,
**AR(1) +0.41**. This partly follows from the definition, so it is a consistency check, not
independent proof.

**What did not hold.**

- *"History matters more for features that carry over more"*: **not supported** (Spearman
  −0.13). The earlier prototype's r = 0.64 does not reproduce with the full multivariate model.
- *"Standard SHAP's top driver is years since diagnosis"*: partly. It is the top driver on 18% of
  visits, second after `SEX` (26%).
- *"Sleep and mood drop"* from the briefing's example: sleep and mood gain share under AR(1)
  (`STAI` 3.3 → 6.0%, `GDS` 1.6 → 3.1%, `ESS` 2.8 → 3.8%) but are **not** the leading drivers;
  `NP2RISE` and LEDD are.
- The total amount explained halves (mean |prediction − start| 1.06 → 0.47), because AR(1)
  explains the change from this patient's expected value, not from the cohort average.

### T5 — Are the explanations stable? ✅

Same 80 visits, different random seed, per-visit ranking agreement: standard 0.988, population
0.965, **AR(1) 0.988**, AR(1) learned 0.978.

---

## 4. Claims the paper can make

| Claim | Evidence | Status |
|---|---|---|
| Standard SHAP explains longitudinal visits with unrealistic patients | T3: 97.6% detectable, jumps 2.3× real | ✅ |
| AR(1)-SHAP's filled patients are realistic | T3: 67.9% detectable, jumps 1.12× | ✅ (not perfect) |
| The estimator is correct | T2: r = 0.99 against the exact answer | ✅ |
| The patient's history, not just correlations, is what changes the answer | T3/T4: population-conditional moves little | ✅ |
| Time-since-diagnosis credit is an artefact of population sampling | T4: −86.8% when learned | ✅ |
| Explanations point at what changed for this patient | T4: +0.41 vs +0.06 | ✅ partly by definition |
| History matters more for slowly changing features | T4: −0.13 | ❌ |
| The new top drivers are clinically actionable | no ground truth in the data | ⚪ untested — needs clinician review |

## 5. Limitations

- **The model being explained is modest** (0.68 balanced accuracy). Attributions describe the
  model, not the disease.
- **Conditional credit can land on unused features** (T2). Report this alongside every result.
- `NP2RISE`, the new top driver, is an axial item flagged as possible proxy leakage
  (`README.md` §9.3).
- Feature-level gaps: a carried-forward value up to 12 months old is treated as "previous".
- The copula treats ordinal values as points when conditioning; it is an approximation.
- Single cohort (PPMI), 88 explained patients, no external validation.
- Novelty against Gaussian conditional Shapley (Aas et al. 2021) rests on the temporal anchor,
  the gap model and the realism evidence. The population-conditional row is the ablation that
  shows the history term matters.
