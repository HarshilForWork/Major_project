# TRACE progress log

Brief: `docs/TRACE_BRIEF.md`. Newest entries at the bottom.

## 14 Oct 2026: Stage 0 (discovery and setup)

**Done**
- Explored the repo; wrote `results/DISCOVERY.md`.
- Created `configs/trace.yaml` (seeds, tolerances, k values, bootstrap count, background size, proposed eps and threshold).
- Copied the pilot into `notebooks/trace_pilot.py`.
- Stored the brief as `docs/TRACE_BRIEF.md` and added a pointer to it in the repo's `CLAUDE.md`.
- Added `results/cache/` to `.gitignore` (per-patient arrays). Added `shap` and `pyyaml` to `requirements.txt`.
- Installed `shap 0.49.1`.
- Logged versions: python 3.10.11; shap 0.49.1; xgboost 3.1.2; numpy 2.2.6; pandas 2.3.2; scikit-learn 1.7.2; scipy 1.15.3; pyyaml 6.0.3; matplotlib 3.10.7.
- Verified that the saved PPMI model equals a fresh retrain (max margin difference 0.0), so the models are reproducible from seed 42.

**Failed / found**
- `shap 0.49.1` cannot parse XGBoost 3.1.2's base-score format (`"[4.9999997E-1]"`). Planned fix: a documented shim that converts a one-element vector to a scalar (see DISCOVERY.md). Gates G1 and G4 will verify it independently.
- There is no Asymmetric Shapley implementation in the repo.
- `LOGMEM_DELAYED` is very likely a diagnosis-defining feature in ADNI (needs a decision).

**Decisions (user, 14 Oct 2026)**
- ADNI main model = no MMSE, FAQ or LOGMEM_DELAYED (`adni_main`). Brief's M_minus (`adni_minus_mf`) and `adni_full` are sensitivity models.
- Asymmetric Shapley: skipped (listed under Deviations).
- Feature groups, decision threshold and faithfulness eps: to be confirmed before Stages 6 and 9.

## 14 Oct 2026: Stage 1 (models and cohorts)

- Models saved to `models/trace/*.json` (gitignored). Same recipe as before (`backend.make_classifier("xgb", 2)`, balanced weights, development split only, no tuning). No early stopping (`best_iteration` None; all 300 trees). base_score 0.5, so margin offset is 0.
- `ppmi` and `adni_full` are identical to the classifiers in the AR(1)-SHAP artifacts (max margin difference 0.0).
- Locked-test balanced accuracy:
  - ppmi 0.676 (1,018 TD/PIGD visits). Below the project's headline ~0.69, which came from a different (CV-selected) model; this is the same model the pilot used.
  - adni_full 0.980; adni_minus_mf 0.930; adni_main 0.916 (1,685 CN/Dementia visits).
- Analysis patients (locked test, >= 3 visits): PPMI 88 (median 17 visits, range 4-21); ADNI 475 (median 8, range 3-27). **Matches the pilot.**

## 14 Oct 2026: Stage 2 (engine)

- `src/trace_pd/explain/trace_engine.py`: `pairwise_F`, `trace_from_F`, `trace_via_background`, `trace_via_lstsq`, the shap/XGBoost-3 base-score shim, `nan_equal`, `tree_output_range`, `used_feature_mask`.
- `src/trace_pd/evaluation/trace_run.py`: models, cohorts and the per-patient cache in `results/cache/<model>/p<idx>.npz` (gitignored; anonymous index = position among sorted eligible test patients).
- First real patient (T = 19): add-up error 1.4e-6. The background route matches the closed form exactly (0.0). That is expected (shap averages the same per-reference sums) but is not an independent correctness check; G4 is.

## 14 Oct 2026: Stage 3 (gate). PPMI

`results/gate/ppmi.json`:
- **G1 PASS.** Pair add-up max error 3.3e-6; trajectory decomposition 2.2e-6.
- **G2 PASS.** Triples 2.2e-16; closed form vs least squares 1.3e-15; closed form vs background route 0.0.
- **G3 PASS.** Never-changing features: max |u| = 0.0 over 221 (patient, feature) cases. Unused features: n/a, because the PPMI model splits on all 26 features.
- **G4 FAIL, but not on accuracy.** All 1,488 brute-forced pairs match shap to max 1.3e-6, in every NaN category (NaN only in the reference: 269 pairs; only in the explicand: 194; in both: 828; none: 345). Antisymmetry 0.0. **The failure is the sample-size rule:** PPMI has only 1,488 pairs with k <= 12 differing features (median k is 17, since PPMI has 26 features), so ">= 2,000 pairs with k <= 12" cannot be met.
  - Supplementary run, not the official gate (`results/gate/ppmi_G4_supplementary_k13.json`): with k <= 13 there are 2,192 eligible pairs. 2,000 were sampled; max error 1.5e-6 in every NaN category.
  - Config left unchanged (max_k 12). **Needs your decision.**
- "Too good" exact zeros checked:
  - Antisymmetry 0.0, background route 0.0 and constant-feature u 0.0 are bit-exact. TreeSHAP's single-reference algorithm treats a feature with equal values as contributing nothing, and the F(r,t) and F(t,r) sums are the same terms with opposite sign.
  - These were computed by independent calls (no code copies F[r,t] into F[t,r]).
  - G4, which never calls shap, agrees to 1e-6.

## 14 Oct 2026: user instructions, round 2 (these supersede the Stage 0 decisions above)

**Accepted:** code/config locations, threshold 0.0, all-visits analysis (plus a labelled-only sensitivity run), Asymmetric Shapley skipped. Eps stays open.

**ADNI variants:**

| Variant | Code name | Dropped | Role |
|---|---|---|---|
| M_minus | `adni_minus` | MMSE, FAQ | **Primary** (matches pilot and proposal) |
| M_minus_LM | `adni_minus_lm` | MMSE, FAQ, LOGMEM_DELAYED | Variant |
| M_minus2 | `adni_minus2` | MMSE, FAQ, LOGMEM_DELAYED, LOGMEM_IMMEDIATE | Variant |
| M_full | `adni_full` | none | Sensitivity |

- Gates run on all four ADNI models plus PPMI. Stages 4-9 run on PPMI and M_minus; Stages 4 and 7 also run on M_minus_LM and M_minus2.
- `adni_main` (= M_minus_LM) was briefly deleted when it looked dropped; it was retrained (deterministic, seed 42). Same BA (0.916) and output-range bound (26.363) as before.
- Locked-test BA: M_minus 0.930, M_minus_LM 0.916, M_minus2 0.887, M_full 0.980.
- Analysis patients, labelled-only sensitivity: PPMI 88 (median 14 visits); ADNI 426 (median 5).

**shap fix, reworked:**
- **Newer release?** No: 0.49.1 is the newest on PyPI. shap's GitHub master has fixed it (parses `"[5E-1]"` with `ast.literal_eval`), but the fix isn't released yet. Drop our fix once a release includes it.
- **What changed:** the global monkey-patch is gone. shap now gets a verified **copy** of the model (`trace_engine.shap_model_copy`). It is a `Booster` subclass whose UBJSON reports the base score as a scalar string. The copy is checked to decode to exactly the original model apart from that one string.
- **Single route:** every shap explainer (engine and baselines) goes through `trace_engine.explainer`.
- **Checked:** the new route reproduces the cached F bit-for-bit (max diff 0.0 on 25 patients each for ppmi, adni_minus, adni_full).
- **Model hashes:** recorded before the rework in `results/gate/model_hashes_before.txt`; compared again at the end.

**Static-feature report (asked for before Stage 2; done now):**
- **ADNI encoding:** SEX and HANDED are text, encoded by sorted `pd.factorize` on the full table: SEX {Female: 0, Male: 1}, HANDED {Left: 0, Right: 1}, missing -> NaN.
- **PPMI encoding:** SEX and HANDED are numeric as stored (SEX 0/1; HANDED 1/2/3), no missing values.
- **Variation across visits (NaN-aware):** **0 patients** in either cohort have any static feature (PPMI: SEX, HANDED; ADNI: SEX, HANDED, EDUCATION_YEARS) changing across visits, among all patients or analysis patients.
- **ADNI missingness is whole-patient:** SEX is missing for 193 patients at every visit, HANDED for 208, EDUCATION_YEARS for 212, and is never missing for only some visits.

**Pilot cross-check (new, supplementary gate column):** F is compared with the pilot's own exact tree-Shapley code on 2,000 random ordered pairs of any size k. PPMI: k up to 24 (median 17), max diff 2.8e-8, PASS. This covers the large-k pairs that G4 cannot reach.

**Feature groups:** `configs/feature_groups.yaml`. Draft, per model, time features in their own group; checked to cover each model's features exactly. Awaiting confirmation.

## 14 Oct 2026: Stage 3 (gate). All models. **GATE NOT PASSED; stopped before Stage 4.**

Full table: `results/gate/GATE.md`. Per model: `results/gate/<model>.json`. pytest: 23 passed, 2 failed.

| Model | G1 | G2 | G3 | G4 | Pilot code |
|---|---|---|---|---|---|
| ppmi | PASS (3.3e-6) | PASS | PASS | **FAIL**: only 1,488 eligible pairs (all match, max 1.3e-6) | PASS (2.8e-8, k up to 24) |
| adni_minus (primary) | PASS (7.4e-6) | PASS | PASS (2,920 cases) | PASS (3.9e-6, 2,000 pairs) | PASS (4.1e-8) |
| adni_minus_lm | PASS (9.0e-6) | PASS | PASS | PASS (4.3e-6) | PASS (3.1e-8) |
| adni_minus2 | PASS (9.0e-6) | PASS | PASS | PASS (4.1e-6) | PASS (3.4e-8) |
| adni_full | **FAIL (1.03e-5)** | PASS | PASS | PASS (5.2e-6) | PASS (6.5e-8) |

- **G1 adni_full failure, diagnosed (`results/gate/G1_diagnosis.md`):** XGBoost's `predict` sums 300 trees in float32. Its margins are off from the exact float64 tree sums by up to 8.8e-6, so a difference of two margins can exceed 1e-5. Against exact float64 margins, G1 is 5.6e-8 to 9.1e-8 for every model. The same rounding explains why G4 errors are around 1e-6 rather than 1e-8. Tolerance not loosened.
- **G3 unused-feature check:** vacuous for every model (each model splits on all its features). Reported, not hidden.
- **Model hashes:**
  - Before: `results/gate/model_hashes_before.txt`. After: `results/gate/model_hashes_after.txt`.
  - All 5 files that existed before (ppmi, adni_minus and adni_full json; both AR(1)-SHAP pkl) are byte-identical. The only new entries are adni_minus_lm and adni_minus2, trained after the snapshot.

## 14 Oct 2026: Stage 3 gate. **PASSED (25/25 pytest) after the user's decisions**

**User decisions:**
- G1 is measured against the exact float64 tree sum (`trace_engine.exact_margin`; XGBoost routing, float64 leaf sums). The limit stays 1e-5, and both numbers are reported.
- PPMI G4 uses k <= 13 (2,000 pairs).

**Results:**
- G1: 3.6e-8 to 9.1e-8 on every model. Against XGBoost float32 it is 3.3e-6 to 1.0e-5, reported in brackets in `results/gate/GATE.md`.
- G2, G3 and G4 pass on all five models; the pilot-code column passes too. PPMI G4: 2,000 pairs (k <= 13), max 1.5e-6.

**New in the engine:** `baseline_shapley_batch` / `pairwise_F_fast`, the pilot's exact vectorised leaf-path algorithm. It is checked against the cached shap F on **every pair of every patient**: ppmi max diff 2.8e-8, adni_minus 4.3e-8. It is used only where shap would be too slow: Stage 4 seed models and Stage 7 deletions and retraining.

## 14 Oct 2026: Stage 4 (size of the problem). DONE; full tables in `results/stage4/STAGE4.md`

- **Pilot reproduced exactly** on all 8 numbers for both PPMI and ADNI M_minus.
- **Step-by-step vs direct, compared with seed noise** (top cause differs; patient means with 95% CI):

  | Run | Comparison | Step-by-step vs direct | Seed noise | Difference |
  |---|---|---|---|---|
  | ADNI M_minus | first->last | 22.9% | 11.5% | +11.4 [7.5, 15.5] |
  | ADNI M_minus | gap pairs | 18.7% | 9.7% | +9.0 [7.3, 10.6] |
  | M_minus_LM | first->last | 18.7% | 11.2% | +7.6 [4.0, 11.0] |
  | M_minus_LM | gap pairs | 16.2% | 9.7% | +6.5 [5.1, 7.9] |
  | M_minus2 | first->last | 15.4% | 9.6% | +5.8 [2.7, 9.0] |
  | M_minus2 | gap pairs | 15.2% | 9.2% | +6.0 [4.8, 7.1] |
  | PPMI | first->last | 30.7% | 20.3% | +10.4 [-0.0, 20.5] (borderline) |
  | PPMI | gap pairs | 30.3% | 25.0% | +5.3 [2.5, 7.9] |
  | PPMI, labelled visits only | first->last | | | +5.5 [-4.0, 15.1] (n.s.) |
  | PPMI, labelled visits only | gap pairs | | | -2.3 [-5.6, 0.9] (n.s.) |
  | ADNI M_minus, labelled visits only | first->last | | | +9.7 [6.2, 13.3] |
  | ADNI M_minus, labelled visits only | gap pairs | | | +3.1 [1.2, 5.1] |

- **Reading:** on every ADNI variant the drift is clearly above retraining noise, with the CI excluding 0. On PPMI it is weak and does not hold with labelled visits only. PPMI's own retraining noise is high (20-25%), consistent with its weak model (BA 0.676).
- **Trust score I:**
  - Medians: PPMI 0.030; M_minus 0.015, M_minus_LM 0.013, M_minus2 0.009.
  - It rises with the number of visits T.
  - It is positively correlated with step-by-step-vs-direct disagreement in every run: Spearman 0.38-0.74, every CI above 0.
- **TRACE changes the top cause of a direct pair:** PPMI about 17% of pairs; ADNI about 6-7%.
- **Locality price** (an unchanged feature gets more than 10% of a pair's TRACE explanation): 0.03-1.2% of (pair, unchanged feature) cases. Its 99th-percentile size is 0.10 log-odds (PPMI) and 0.24 (M_minus).

## 14 Oct 2026: Stages 5 and 7 (running) and headline figures

**Stage 5 baselines** (`src/trace_pd/evaluation/trace_baselines.py`):
- Methods: trace, direct, chain, fixed_first, fixed_last, population (500 training visits, fixed seed), path_dep, ar1.
- AR(1) is the team's implementation (48 perm x 24 draws). It runs for the primary models only, and is undefined at a patient's first visit.
- Population and path-dependent SHAP were computed for all four models.

**Stage 7** (`trace_stage7.py`):
- Deletion uses exact closed forms from the cached F. Retraining uses the 10 seed models.
- AR(1) is excluded from retraining stability (cost).
- Deletion results so far, M_minus_LM and M_minus2 (439 patients each with >= 4 visits): Proposition 4 bound never violated; the worst case is 1.7% of the bound.

**Headline figures** (`trace_figures.py`; `results/figures/fig_headline_*`):
- **Rule A** (largest score change among clear chain-vs-direct disagreements) picks patients whose change is mostly **missingness**:
  - PPMI Patient A has 25 features recorded at the first visit and **3** at the last.
  - ADNI Patient A has 13 recorded at the first visit, then 4-5 at most later visits.
- **Rule B** (the same, but first and last visits each have >= 80% of features recorded in PPMI, >= 60% in ADNI; no ADNI pair reaches 75%) gives:
  - PPMI Patient B: 25/26 recorded; step-by-step names LEDD_TOTAL_MG, direct and TRACE name YRS_SINCE_DIAGNOSIS.
  - ADNI Patient B: 14/18 recorded; step-by-step names ECOG_PARTNER_TOTAL, direct and TRACE name LOGMEM_DELAYED.
- **Missingness check:** drift persists among well-recorded pairs.
  - PPMI, both visits >= 80% recorded: 30.5% top-cause disagreement (59 patients).
  - ADNI, both >= 60% recorded: 19.9% (176 patients).
  - Drift is higher when many features switch recorded <-> missing: PPMI 38.9% vs 25.0%; ADNI 28.1% vs 19.2%.

## 14 Oct 2026: Stages 5 and 7 DONE; guide summary written

- Results: `results/stage5/STAGE5.md`, `results/stage7/STAGE7.md`, `results/stage7/never_changing_credit.json`.
- **Summary for the guide:** `results/SUMMARY_FOR_GUIDE.md` (aggregates only).
- **Key results:**
  - **Deletion stability:** TRACE beats chaining (intermediate deletions) and the fixed references (when their reference visit is deleted), with all paired CIs excluding 0.
  - **Retraining stability:** mixed. TRACE is better than chaining on PPMI (+0.041) and slightly worse on ADNI (-0.007 to -0.011).
  - **Population and path-dependent SHAP differences** are fully deletion-proof and slightly more retraining-stable. But they give 2-4% of their credit to never-changing features; TRACE gives 0.
  - **AR(1)-SHAP differences** miss the actual change by 1.5-2 log-odds on average (not a change explanation).
- **Next:** Stage 6 (needs eps), Stage 8 (sanity check), Stage 9 (needs feature groups).

## 14 Oct 2026: Stages 6, 8, 9, 10 DONE (all stages complete)

- **Stage 6 faithfulness** (`trace_stage689.py`):
  - TRACE is only clearly below direct Baseline Shapley at k = 1 (the upper reference by construction).
  - It beats chaining and AR(1).
  - On PPMI at k = 3 it is borderline vs population and path-dependent (CI touches 0).
  - eps 0.05 was used while still open; 0.01, 0.1 and 0.25 give the same conclusions.
- **Stage 8 sanity check:**
  - Shuffled-label BA is 0.526 (PPMI) and 0.516 (ADNI).
  - Per-visit tally agreement is about 0.
  - Per-feature importance Spearman stays 0.30 / 0.46.
- **Stage 9** (draft groups):
  - PPMI flips: 233 model flips. Medication takes 14% of flip credit (not dominant); motor daily-living 36%.
  - Only 18 of 99 true-label flips are model flips.
  - Measurement accounting: 44.5% (PPMI) and **88.4% (ADNI)** of TRACE credit comes from recorded/missing status differences.
  - Within/between identity: **PASS** (5e-8).
- **Bug found and fixed:**
  - shap silently subsampled the population background from 500 to 100 rows. Found when the Stage 9.4 identity failed (0.065 / 0.55).
  - Fixed in `trace_engine.explainer` (Independent masker, `max_samples` = all rows).
  - All population-based results were recomputed. TRACE itself was unaffected.
- **Stage 10:**
  - `results/REPORT.md` is assembled by `trace_report.py`; checklist and deviations are in `results/REPORT_checklist_deviations.md`.
  - `results/SUMMARY_FOR_GUIDE.md` is updated.

## (earlier) Stage 4 setup notes

- Module `src/trace_pd/evaluation/trace_stage4.py`; outputs in `results/stage4/`.
- **PPMI reproduces the pilot exactly on all 8 numbers:** top differs 30.7%, clear 11.4%, total disagreement median 51.8%, sign flip 0.0%, TRACE change 17.3% over 10,997 pairs, locality 0.1%, I median 0.030.
- **Seed models:** 10 per model (seeds 0-9), saved in `models/trace/seeds/`. The seed does change the model (max margin difference 1.06 on PPMI).

**Need from you**
1. Confirm the feature groups (`configs/feature_groups.yaml`) before Stage 9.
2. Faithfulness eps (open; needed before Stage 6).
