# TRACE: Stage 0 discovery (14 Oct 2026)

Governing brief: `docs/TRACE_BRIEF.md`. Aggregates only in this file; no patient identifiers.

## What exists

| Item | Where | Notes |
|---|---|---|
| PPMI preprocessed data | `data/processed/ppmi_tdpigd_long.csv` | 6,922 visits, 439 patients; id `PATNO`, time `VISIT_MONTH`, label `LABEL` (TD / PIGD / Indeterminate / NaN) |
| PPMI feature list | `data/processed/ppmi_tdpigd_dictionary.csv`, bucket `CHEAP_FEATURE` | 26 features, all numeric |
| PPMI locked split | `data/interim/locked_test_patients.csv`; `trace_pd.evaluation.splits.split_dev_test` | 20% of patients, seed 2026 |
| ADNI preprocessed data | `data/processed/adni_dx_long.csv` | 24,302 visits, 4,470 patients; id `RID`; label `LABEL` (CN / MCI / Dementia); study phase column `COLPROT` exists |
| ADNI feature list | `data/processed/adni_dx_dictionary.csv`, bucket `CHEAP_FEATURE` | 23 features; `SEX` and `HANDED` are text, encoded as integers by sorted `pd.factorize` (as in `evaluate_ar1_shap_adni.load`) |
| ADNI locked split | `data/interim/locked_test_patients_adni.csv`; `trace_pd.evaluation.splits_adni.split_dev_test` | 894 test patients |
| PPMI model | inside `models/ar1_shap_artifacts.pkl` (`classifier`) | XGBoost, `backend.make_classifier("xgb", 2)` (depth 3, 300 trees, lr 0.05, subsample 0.8, colsample 0.8, seed 42), balanced weights, trained on development TD/PIGD visits. **Verified: retraining reproduces it exactly** (max margin difference 0.0). |
| ADNI full model (`M_full`) | inside `models/ar1_shap_artifacts_adni.pkl` (`classifier`) | Same recipe, development CN/Dementia visits, all 23 features |
| ADNI `M_minus` | **not saved** | Will be retrained (same recipe, same locked development split, no tuning on test) and saved, per Stage 1 |
| Pilot code | `notebooks/trace_pilot.py` (copied from the session scratchpad) | Has its own exact tree-Shapley code; defines "total disagreement" = `sum_j abs(direct_j - chain_j) / sum_j abs(direct_j)` and "clear winner" = direct's 2nd-largest abs credit < 80% of the largest. Uses **all** visits of locked-test patients with >= 3 visits (including unlabelled / Indeterminate / MCI visits). |
| AR(1)-SHAP (conditional) | `src/trace_pd/explain/ar1.py`, `src/trace_pd/explain/conditional_shapley.py` | Sampling-based; usable as the Stage 5 conditional baseline |
| Bootstrap helper | `trace_pd.evaluation.splits.patient_bootstrap_ci` | Patient-level percentile CI (default 200 draws; TRACE config asks for 1,000) |

## What is missing

1. **Asymmetric Shapley: no implementation exists in the repo.** Per the brief I have not invented one. This baseline is blocked until you decide.
2. **FA-SHAP forecasts: none in the repo.** Optional baseline, so it will be skipped.
3. **Feature groupings: none exist for TRACE.** A draft is below for you to confirm.
4. **`shap` was not installed.** Installed `shap 0.49.1`.

## Problem found: shap 0.49.1 cannot read our XGBoost 3.1.2 models

`shap.TreeExplainer` crashes when it parses the model's `base_score`. XGBoost 3.x stores it as a one-element vector string (`"[4.9999997E-1]"`); shap expects a plain number.

- **Planned fix:** a small, documented compatibility shim inside the TRACE engine. It converts a one-element `"[x]"` base score to `"x"` while shap reads the model, and raises an error for anything else (for example a multi-element vector). Nothing else about the model or the explanation changes.
- **Rejected alternative:** downgrading XGBoost. That would change the environment behind every number already reported in the project.
- **How it is checked:** gate checks G1 (add-up) and G4 (brute force against `booster.predict`) independently verify the result.

## Model facts (Stage 1 pre-check)

- `best_iteration`: not set (no early stopping); all 300 trees are used by `predict`.
- `base_score` = 0.5 (logit 0), objective `binary:logistic`; explanations use `output_margin=True`.

## Rule 4 check: other ADNI features that help define the diagnosis

From ADNI's published enrolment criteria (from memory, **to be verified against the ADNI procedures manual**):

- **`LOGMEM_DELAYED` (Logical Memory II, delayed recall) is a diagnostic criterion.** Education-adjusted cut-offs separate CN from MCI/AD at enrolment. It is in the 23 features and in `M_minus`. **Flagged; not removed (needs your decision).**
- `LOGMEM_IMMEDIATE`: same test battery but not itself a cut-off criterion. Flagged as closely related.
- `GDS_TOTAL` (depression scale): an eligibility exclusion (GDS >= 6), not a CN-vs-Dementia criterion. Low risk.
- CDR is already banned. MMSE and FAQ are dropped in `M_minus`.

## Draft feature groups (for Stage 9; please confirm or edit)

PPMI (26):
- **demographic/static:** SEX, HANDED
- **time/clock:** AGE_AT_VISIT, YRS_SINCE_DIAGNOSIS, YRS_SINCE_SYMPTOM_ONSET
- **medication:** LEDD_TOTAL_MG, PDTRTMNT, N_CONMEDS
- **motor daily living (MDS-UPDRS Part II items):** NP2DRES, NP2EAT, NP2HOBB, NP2HWRT, NP2HYGN, NP2RISE, NP2SALV, NP2SPCH, NP2SWAL, NP2TURN
- **non-motor questionnaires:** NP1PTOT, NP1RTOT, ESS_TOTAL, GDS_TOTAL, RBDSQ_TOTAL, SCOPA_AUT_TOTAL, STAI_TOTAL
- **cognition:** MCATOT

ADNI (23):
- **demographic/static:** SEX, HANDED, EDUCATION_YEARS
- **time/clock:** AGE_AT_VISIT
- **vitals/body:** BP_SYSTOLIC, BP_DIASTOLIC, PULSE, HEIGHT_CM, WEIGHT_KG
- **cognitive tests:** MMSE_TOTAL, MOCA_TOTAL, LOGMEM_IMMEDIATE, LOGMEM_DELAYED, BOSTON_NAMING, CATEGORY_FLUENCY_ANIMALS, DIGIT_SPAN, TRAILS_A_SEC, TRAILS_B_SEC
- **function/daily living:** FAQ_TOTAL, ECOG_PATIENT_TOTAL, ECOG_PARTNER_TOTAL
- **mood/behaviour:** GDS_TOTAL, NPI_TOTAL

ADNI has no medication features.

## Proposed decisions (waiting for confirmation)

- **Decision threshold:** margin 0.0. Both models use balanced weights and `predict()` at probability 0.5.
- **Faithfulness eps:** 0.05 log-odds.
- **Analysis visits:** all visits of a locked-test patient (as in the pilot), not only labelled ones. The model scores every visit, and the PPMI subtype-flip analysis needs consecutive scored visits.
- **Code location:** `src/trace_pd/explain/trace_engine.py`, not `src/trace/engine.py`. A top-level package named `trace` would shadow Python's built-in `trace` module, and the repo's package is `trace_pd`.
- **Config file:** `configs/trace.yaml`, not `config.yaml`. The repo already has a documentation-only `configs/config.yaml`.
