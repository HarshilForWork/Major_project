# TRACE: summary of results so far (14 Oct 2026)

**TRACE** gives consistent explanations of how a patient's model score changes across clinic visits.

Implemented to the brief `docs/TRACE_BRIEF.md`. **All stages (0-10) are done.** The full technical report is `results/REPORT.md`. Two choices were still open when the last stages ran: eps (0.05 used, with sensitivity) and feature groups (draft used). Both are flagged below.

All numbers are aggregates over locked-test patients; no patient identifiers appear anywhere. Patient-level arrays stay in a local, git-ignored cache. CIs are 95% patient-level bootstrap intervals (1,000 draws) unless stated otherwise.

**In one paragraph:**
- **The maths is exact on real data.** Every exactness check passes on every model.
- **The drift is real in ADNI.** Step-by-step explanations of a patient's change name a different main cause from direct comparisons about twice as often as retraining noise alone would, in every ADNI model variant.
- **PPMI is weak.** Its model is weak and its retraining noise is high, so the effect there is borderline.
- **Stability when a visit is deleted:** TRACE moves only slightly, and 3-11x less than step-by-step chaining or a fixed reference visit in the cases where those methods are exposed.
- **Stability across retraining:** TRACE is **not** the most stable method. On ADNI it is slightly below most baselines.
- **The strongest competitor is per-visit population SHAP differences.** They never move when a visit is deleted, and are slightly more retraining-stable. But they give about 2.9-4.0% of their credit to features that never changed for the patient; TRACE gives exactly zero.
- **Faithfulness:** TRACE points at the features that really move the model as well as the best methods. It is beaten only by direct comparison on its own pair, at k = 1, which is expected by construction. It beats step-by-step chaining.
- **Sanity check:** with shuffled labels, explanations fall apart at the visit level.

---

## 1. Data and models

| | PPMI (Parkinson's) | ADNI (Alzheimer's) |
|---|---|---|
| Cohort | 439 patients, 6,922 visits | 4,470 patients, 24,302 visits |
| Task | TD vs PIGD motor subtype | cognitively normal vs dementia |
| Locked test split | 20% of patients (88) | 894 patients |
| **Analysed** (locked test, >= 3 visits, all visits) | 88 patients, 1,382 visits (median 17, range 4-21) | 475 patients, 4,484 visits (median 8, range 3-27) |
| Labelled-visits-only sensitivity | 88 patients, 1,130 visits (median 14) | 426 patients, 2,509 visits (median 5) |
| Used for deletion stability (>= 4 visits) | 88 | 439 |

**Models.** All are XGBoost: depth 3, 300 trees, learning rate 0.05, subsample and colsample 0.8, seed 42, balanced weights. All are trained on the development split only, with no tuning on test. Balanced accuracy (BA) is on locked-test labelled visits; the threshold is margin 0 (probability 0.5).

| Model | Role | Features | Locked-test BA |
|---|---|---|---|
| `ppmi` | PPMI model | 26 | 0.676 (1,018 visits) |
| `adni_minus` (**M_minus**) | **ADNI primary**: no MMSE, FAQ | 21 | 0.930 (1,685 visits) |
| `adni_minus_lm` (M_minus_LM) | variant: also no LOGMEM_DELAYED | 20 | 0.916 |
| `adni_minus2` (M_minus2) | variant: also no LOGMEM_DELAYED, LOGMEM_IMMEDIATE | 19 | 0.887 |
| `adni_full` (M_full) | sensitivity, all features | 23 | 0.980 |

- `ppmi` and `adni_full` are identical to the models used in the earlier AR(1)-SHAP work (maximum score difference 0.0).
- Model-file SHA-256 hashes were recorded before and after the work. All pre-existing model files are byte-identical.
- PPMI's 0.676 is below the project's headline ~0.69, which came from a different, CV-selected model. This is the same model the pilot used.

**Static features.**
- **ADNI encoding:** SEX and HANDED are text, encoded {Female: 0, Male: 1} and {Left: 0, Right: 1}; missing stays missing. PPMI stores them as numbers.
- **No patient** in either cohort has SEX, HANDED or (ADNI) years of education changing across visits.
- In ADNI these are missing for whole patients only: 193, 208 and 212 patients respectively. They are never missing at just some visits.

## 2. Exactness gate (Stage 3): passed on all models

Pytest: 25 of 25 tests pass (`tests/test_trace_gate.py`). Limits: G1, G2, G4 <= 1e-5; G3 <= 1e-6.

| Model | G1 add-up, vs exact 64-bit sum (pair / trajectory) | G1 vs XGBoost 32-bit score | G2 consistency (triples / least squares / background) | G3 zeros (never-changing / unused) | G4 brute force (max, pairs) | Pilot-code match (max, 2,000 pairs) |
|---|---|---|---|---|---|---|
| ppmi | 7.9e-8 / 6.6e-8 | 3.3e-6 / 2.2e-6 | 2e-16 / 1e-15 / 0 | 0 (221 cases) / none unused | 1.5e-6 (2,000, k <= 13) | 2.8e-8 |
| adni_minus | 6.4e-8 / 4.4e-8 | 7.4e-6 / 4.9e-6 | 9e-16 / 4e-15 / 0 | 0 (2,920) / none unused | 3.9e-6 (2,000, k <= 12) | 4.1e-8 |
| adni_minus_lm | 5.6e-8 / 3.6e-8 | 9.0e-6 / 6.3e-6 | 9e-16 / 4e-15 / 0 | 0 (2,881) / none unused | 4.3e-6 (2,000) | 3.1e-8 |
| adni_minus2 | 6.2e-8 / 4.3e-8 | 9.0e-6 / 5.6e-6 | 9e-16 / 3e-15 / 0 | 0 (2,845) / none unused | 4.1e-6 (2,000) | 3.4e-8 |
| adni_full | 9.1e-8 / 5.3e-8 | **1.0e-5** | 2e-15 / 6e-15 / 0 | 0 (2,946) / none unused | 5.2e-6 (2,000) | 6.5e-8 |

G4 errors in each missing-value category (NaN only in the reference visit, only in the explained visit, in both, none) are all at the same 1e-6 level. Antisymmetry is exact (0).

**Rule changes, with reasons:**
1. **G1 is measured against the exact 64-bit tree sum.**
   - XGBoost's own `predict` adds up 300 trees in 32-bit precision. It is off by up to 8.8e-6 per visit, so a *difference* of two scores can exceed 1e-5 even when the explanation is exact. On `adni_full`, G1 against the 32-bit score was 1.03e-5, a fail.
   - Against the exact 64-bit sum of the same model's leaves (same routing), G1 is 4e-8 to 9e-8 on every model. The 1e-5 limit was **not** changed, and both numbers are reported.
   - Diagnosis: `results/gate/G1_diagnosis.md`.
2. **G4 for PPMI uses up to 13 changed features (not 12).** Only 1,488 PPMI visit pairs differ in 12 or fewer of the 26 features (the median pair differs in 17), so 2,000 pairs at k <= 12 do not exist. All 1,488 pairs at k <= 12 also matched (max 1.3e-6).
3. **Added check: pilot-code cross-check.** TRACE's pairwise values are compared with the pilot's independent exact tree-Shapley code on 2,000 random pairs of any size (PPMI pairs up to 24 changed features). Max difference is 6.5e-8 or less on every model.

G3's "unused feature" check is vacuous: every model splits on all of its features.

## 3. Pilot reproduction (all 8 numbers reproduced exactly)

| Metric | PPMI now | PPMI pilot | ADNI M_minus now | ADNI M_minus pilot |
|---|---|---|---|---|
| First->last: top cause differs, step-by-step vs direct | 30.7% | 30.7% | 22.9% | 22.9% |
| ... with a clear top cause in the direct explanation | 11.4% | 11.4% | 8.8% | 8.8% |
| First->last total disagreement (median) | 51.8% | 51.8% | 38.6% | 38.6% |
| Sign of the top cause flips | 0.0% | 0.0% | 0.0% | 0.0% |
| TRACE changes the top cause (pairs with a real change) | 17.3% | 17.3% | 7.7% | 7.7% |
| ... number of such pairs | 10,997 | 10,997 | 24,099 | 24,099 |
| Unchanged feature gets > 10% of a pair's TRACE credit | 0.1% | 0.1% | 1.2% | 1.2% |
| Trust score I, median | 0.030 | 0.030 | 0.015 | 0.015 |

## 4. Size of the problem (Stage 4)

### 4.1 Step-by-step vs direct, against retraining noise

**Seed noise** is how often two copies of the model, retrained with different seeds (10 models, 45 pairs), name a different top cause for the same direct comparison. The seed really does change the model: the maximum score difference between seeds is 1.06 (PPMI) to 1.62 (ADNI).

| Run | Patients | First->last: top differs | Seed noise | **Difference** | All pairs with a visit in between: top differs | Seed noise | **Difference** | First->last total disagreement (median) |
|---|---|---|---|---|---|---|---|---|
| PPMI | 88 | 30.7% [20.5, 40.9] | 20.3% [14.9, 25.9] | **+10.4 [-0.0, 20.5]** | 30.3% [28.0, 32.6] | 25.0% [23.6, 26.5] | **+5.3 [2.5, 7.9]** | 51.8% |
| ADNI M_minus | 475 | 22.9% [19.2, 26.9] | 11.5% [9.7, 13.2] | **+11.4 [7.5, 15.5]** | 18.7% [17.0, 20.2] | 9.7% [8.9, 10.6] | **+9.0 [7.3, 10.6]** | 38.6% |
| ADNI M_minus_LM | 475 | 18.7% [15.4, 22.3] | 11.2% [9.3, 12.9] | **+7.6 [4.0, 11.0]** | 16.2% [14.8, 17.6] | 9.7% [8.9, 10.5] | **+6.5 [5.1, 7.9]** | 34.6% |
| ADNI M_minus2 | 475 | 15.4% [12.2, 18.5] | 9.6% [8.0, 11.2] | **+5.8 [2.7, 9.0]** | 15.2% [14.0, 16.5] | 9.2% [8.5, 10.0] | **+6.0 [4.8, 7.1]** | 26.7% |
| PPMI, labelled visits only | 88 | 28.4% [20.5, 38.6] | 23.0% [17.8, 28.5] | **+5.5 [-4.0, 15.1]** | 23.6% [21.2, 26.1] | 25.8% [23.7, 28.3] | **-2.3 [-5.6, 0.9]** | 45.4% |
| ADNI M_minus, labelled visits only | 426 | 20.7% [16.9, 24.9] | 10.9% [9.2, 13.0] | **+9.7 [6.2, 13.3]** | 14.3% [12.3, 16.5] | 11.1% [9.9, 12.5] | **+3.1 [1.2, 5.1]** | 32.8% |

**Reading:**
- **ADNI:** the drift is clearly above retraining noise in every variant and in the labelled-only run.
- **PPMI:** borderline for first->last; small but positive over all pairs; absent with labelled visits only. PPMI's retraining noise is itself high (20-26%), consistent with its weak model.

### 4.2 How often TRACE changes the main cause of a direct comparison

| Run | First->last | Pairs with a visit in between | All pairs |
|---|---|---|---|
| PPMI | 8.0% [3.4, 13.6] | 17.0% [15.6, 18.4] | 16.9% [15.6, 18.0] |
| ADNI M_minus | 6.1% [4.2, 8.4] | 7.1% [6.3, 8.0] | 7.1% [6.3, 7.8] |
| ADNI M_minus_LM | 5.3% [3.4, 7.4] | 6.1% [5.4, 6.8] | 6.6% [5.9, 7.4] |
| ADNI M_minus2 | 4.8% [2.9, 6.7] | 6.1% [5.5, 6.8] | 6.4% [5.7, 7.0] |
| PPMI, labelled only | 11.4% [5.7, 18.2] | 12.5% [11.1, 13.8] | 12.4% [11.1, 13.7] |
| ADNI M_minus, labelled only | 6.1% [4.0, 8.5] | 5.0% [4.1, 6.0] | 6.1% [5.4, 6.9] |

### 4.3 Cost of the impossibility result

No method can be exact, contradiction-free *and* always give zero to a feature unchanged between two visits. The table measures how often TRACE pays that price. It counts (pair, unchanged feature) cases, all pairs.

| Run | Unchanged feature gets > 10% of the pair's credit | > 5% | abs(credit) > 0.05 log-odds | 99th percentile of abs(credit) |
|---|---|---|---|---|
| PPMI | 0.14% | 0.66% | 5.72% | 0.100 |
| ADNI M_minus | 1.16% | 3.12% | 9.53% | 0.240 |
| ADNI M_minus_LM | 0.99% | 2.85% | 8.38% | 0.207 |
| ADNI M_minus2 | 1.23% | 3.05% | 8.11% | 0.168 |
| PPMI, labelled only | 0.03% | 0.49% | 3.74% | 0.085 |
| ADNI M_minus, labelled only | 0.21% | 1.10% | 6.34% | 0.179 |

### 4.4 Trust score I

| Run | Median [IQR] | 90th percentile | Median by number of visits T (n: median) | Spearman(I, first->last disagreement) | Spearman(I, all-pairs disagreement) |
|---|---|---|---|---|---|
| PPMI | 0.030 [0.022, 0.035] | 0.042 | 3-4 (1): 0.013; 5-9 (8): 0.018; 10-14 (23): 0.022; 15+ (56): 0.033 | 0.38 [0.16, 0.57] | 0.39 [0.16, 0.61] |
| ADNI M_minus | 0.015 [0.010, 0.020] | 0.027 | 3-4 (76): 0.009; 5-9 (203): 0.014; 10-14 (116): 0.015; 15+ (80): 0.019 | 0.50 [0.42, 0.58] | 0.43 [0.34, 0.51] |
| ADNI M_minus_LM | 0.013 [0.008, 0.017] | 0.023 | 0.007; 0.013; 0.013; 0.017 | 0.50 [0.42, 0.58] | 0.41 [0.32, 0.49] |
| ADNI M_minus2 | 0.009 [0.004, 0.015] | 0.020 | 0.003; 0.007; 0.011; 0.017 | 0.55 [0.47, 0.62] | 0.51 [0.43, 0.58] |
| PPMI, labelled only | 0.025 [0.019, 0.032] | 0.040 | 0.012; 0.018; 0.029; 0.027 | 0.49 [0.28, 0.64] | 0.74 [0.62, 0.84] |
| ADNI M_minus, labelled only | 0.013 [0.007, 0.019] | 0.024 | 0.008; 0.015; 0.018; 0.023 | 0.58 [0.51, 0.64] | 0.61 [0.54, 0.67] |

**Reading:**
- I is small: 1-3% of the pairwise explanation energy is contradiction.
- It rises with the number of visits.
- It is positively correlated with step-by-step-vs-direct disagreement in every run (all CIs above 0), so it carries information about which patients' explanations to distrust.

### 4.5 Added check: is the drift just a missing-data effect? No

| | First->last top cause differs |
|---|---|
| PPMI, both visits >= 80% of features recorded (59 patients) | 30.5% (vs 30.7% overall) |
| ADNI M_minus, both visits >= 60% recorded (176 patients; no pair reaches 75%) | 19.9% (vs 22.9% overall) |
| PPMI, few vs many features switching recorded <-> missing (split at the median) | 25.0% vs 38.9% |
| ADNI M_minus, few vs many switching | 19.2% vs 28.1% |

Drift persists in well-recorded pairs, and is larger when the set of recorded tests changes.

## 5. Baselines (Stage 5)

Efficiency gap = abs(sum of credits - actual score change), over all pairs s < t, with the exact 64-bit score.

| Method | PPMI max gap | ADNI M_minus max gap | Note |
|---|---|---|---|
| TRACE | 7.9e-8 | 6.4e-8 | exact |
| direct (Baseline Shapley) | 1.2e-7 | 8.6e-8 | exact, but contradicts itself across routes |
| step-by-step chain (DeltaSHAP-style) | 1.0e-7 | 1.0e-7 | exact |
| fixed first-visit reference | 1.1e-7 | 1.0e-7 | exact |
| fixed last-visit reference | 1.1e-7 | 9.2e-8 | exact |
| population SHAP difference (500 training visits) | 6.3e-8 | 7.1e-8 | exact |
| path-dependent TreeSHAP difference | 1.4e-6 | 4.3e-6 | exact up to shap's 32-bit arithmetic |
| conditional AR(1)-SHAP difference (team's method) | 6.5 (mean 1.45) | 10.9 (mean 2.02) | **not** a change explanation: it explains each visit against the patient's forecast, so differences do not add up to the change. Undefined at the first visit (9,755 / 21,398 pairs) |

AR(1)'s own Monte Carlo noise (same visit, two seeds; 95th percentile of abs difference per feature) is 0.013 (PPMI) and 0.039 (ADNI).

## 6. Stability (Stage 7)

### 6.1 Delete one real visit

Average over pairs of the largest per-feature movement, in log-odds. Methods that never move in a given case are shown as 0.

| Model | Deleted visit | TRACE | step-by-step | fixed first | fixed last | AR(1) | direct / population / path-dependent |
|---|---|---|---|---|---|---|---|
| PPMI | first | 0.013 [0.012, 0.014] | 0 | **0.148** [0.139, 0.158] | 0 | 0 | 0 |
| PPMI | intermediate | 0.012 [0.011, 0.013] | **0.050** [0.047, 0.054] | 0 | 0 | 0.031 [0.026, 0.036] | 0 |
| PPMI | last | 0.014 [0.013, 0.016] | 0 | 0 | **0.170** [0.150, 0.188] | 0 | 0 |
| ADNI M_minus | first | 0.045 [0.042, 0.048] | 0 | **0.362** [0.347, 0.379] | 0 | 0 | 0 |
| ADNI M_minus | intermediate | 0.044 [0.041, 0.047] | **0.163** [0.154, 0.172] | 0 | 0 | 0.136 [0.117, 0.154] | 0 |
| ADNI M_minus | last | 0.041 [0.038, 0.045] | 0 | 0 | **0.350** [0.323, 0.376] | 0 | 0 |
| ADNI M_minus_LM | first / intermediate / last | 0.041 / 0.041 / 0.039 | 0 / 0.152 / 0 | 0.326 / 0 / 0 | 0 / 0 / 0.327 | not run | 0 |
| ADNI M_minus2 | first / intermediate / last | 0.030 / 0.030 / 0.030 | 0 / 0.111 / 0 | 0.281 / 0 / 0 | 0 / 0 / 0.250 | not run | 0 |

**Paired differences (TRACE minus method), where the method moves.** Negative means TRACE moves less; every CI excludes 0.

| Model | vs step-by-step (intermediate) | vs fixed first (first deleted) | vs fixed last (last deleted) | vs AR(1) (intermediate) |
|---|---|---|---|---|
| PPMI | -0.038 [-0.042, -0.036] | -0.135 [-0.144, -0.126] | -0.156 [-0.174, -0.136] | -0.019 [-0.024, -0.015] |
| ADNI M_minus | -0.119 [-0.126, -0.113] | -0.317 [-0.333, -0.303] | -0.309 [-0.333, -0.283] | -0.095 [-0.111, -0.078] |

**Bound check (Proposition 4: change <= 4R/(T-1)):** never violated. The worst case uses 0.9% (PPMI) to 1.8% (ADNI) of the bound, so the bound is valid but loose.

**Reading:**
- **Weakness:** TRACE moves a little in every deletion case, because it uses all visits.
- **Strength:** it never moves much. Each competitor that is consistent across routes is badly exposed in one case:
  - step-by-step when a middle visit is deleted, 4x TRACE in ADNI;
  - a fixed reference when its reference visit is deleted, 8-11x TRACE.
- Direct comparisons and per-visit population or path-dependent differences never move under deletion. Direct is not consistent across routes. Population and path-dependent differences are consistent, but see 6.3.

### 6.2 Retraining stability (10 seed models)

Spearman correlation across features of the same pair's explanation between two retrained models.

| Model | TRACE | direct | step-by-step | fixed first | fixed last | population diff | path-dependent diff |
|---|---|---|---|---|---|---|---|
| PPMI | 0.845 [0.837, 0.853] | 0.834 [0.827, 0.840] | 0.804 [0.796, 0.811] | 0.819 [0.811, 0.826] | 0.815 [0.808, 0.822] | 0.882 [0.875, 0.888] | 0.882 [0.875, 0.888] |
| ADNI M_minus | 0.823 [0.816, 0.829] | 0.859 [0.853, 0.865] | 0.830 [0.824, 0.835] | 0.814 [0.808, 0.820] | 0.848 [0.842, 0.854] | 0.842 [0.834, 0.848] | 0.841 [0.834, 0.848] |
| ADNI M_minus_LM | 0.820 [0.813, 0.826] | 0.863 [0.858, 0.868] | 0.830 [0.824, 0.835] | 0.812 [0.805, 0.818] | 0.850 [0.844, 0.855] | 0.840 [0.832, 0.847] | 0.837 [0.829, 0.844] |
| ADNI M_minus2 | 0.793 [0.786, 0.799] | 0.837 [0.831, 0.843] | 0.804 [0.799, 0.809] | 0.783 [0.776, 0.789] | 0.827 [0.820, 0.832] | 0.829 [0.822, 0.835] | 0.825 [0.818, 0.831] |

**Paired differences, TRACE minus method** (positive means TRACE is more stable across retraining):

| Model | vs direct | vs step-by-step | vs fixed first | vs fixed last | vs population | vs path-dependent |
|---|---|---|---|---|---|---|
| PPMI | +0.011 [0.008, 0.015] | **+0.041** [0.037, 0.045] | +0.026 [0.023, 0.030] | +0.029 [0.024, 0.035] | -0.037 [-0.041, -0.033] | -0.037 [-0.042, -0.033] |
| ADNI M_minus | -0.036 [-0.042, -0.031] | **-0.007** [-0.012, -0.002] | +0.009 [0.007, 0.011] | -0.025 [-0.031, -0.020] | -0.019 [-0.022, -0.015] | -0.018 [-0.022, -0.015] |
| ADNI M_minus_LM | -0.043 [-0.049, -0.038] | **-0.010** [-0.016, -0.005] | +0.008 [0.006, 0.010] | -0.030 [-0.036, -0.024] | -0.020 [-0.024, -0.016] | -0.017 [-0.021, -0.013] |
| ADNI M_minus2 | -0.044 [-0.051, -0.038] | **-0.011** [-0.018, -0.006] | +0.010 [0.007, 0.013] | -0.033 [-0.040, -0.027] | -0.036 [-0.040, -0.031] | -0.032 [-0.036, -0.027] |

**Reading:**
- All methods are fairly stable across retraining (Spearman about 0.78-0.88), and the differences between methods are small (at most 0.04).
- **PPMI:** TRACE is more retraining-stable than direct, step-by-step and both fixed references, but less than the population and path-dependent differences.
- **ADNI:** TRACE is slightly *less* retraining-stable than every method except the fixed first-visit reference. The gap to step-by-step is tiny (0.007-0.011), but its CI excludes 0.
- **So TRACE is not the most retraining-stable method.** Its stability advantage is specific to deleting or adding visits (6.1).

### 6.3 Added check: credit given to features that never changed for the patient

Share of all credit (over all pairs) given to features whose value is identical at every visit of that patient (for example sex, or a test never recorded).

| Model | TRACE | step-by-step | direct | fixed first | population SHAP diff | path-dependent diff |
|---|---|---|---|---|---|---|
| PPMI | 0 | 0 | 0 | 0 | 2.88% | 2.94% (max 0.41 log-odds) |
| ADNI M_minus | 0 | 0 | 0 | 0 | 2.86% | 2.32% (max 0.35) |
| ADNI M_minus_LM | 0 | 0 | 0 | 0 | 3.33% | 2.57% (max 0.45) |
| ADNI M_minus2 | 0 | 0 | 0 | 0 | 3.99% | 3.15% (max 0.54) |

CIs and maxima for every method: `results/stage7/never_changing_credit.json`.

This is the price of the population methods' perfect deletion stability: they say something "changed" that did not. TRACE guarantees zero for these features (Theorem 1c).

## 6b. Faithfulness (Stage 6)

Revert the top-k features of the later visit to the earlier visit's real values, and measure what share of the score change is recovered. Pairs with a score change below eps = 0.05 are excluded: 311 of 11,049 in PPMI and 1,539 of 25,407 in ADNI.

| Method | PPMI k=1 | PPMI k=3 | PPMI k=5 | ADNI k=1 | ADNI k=3 | ADNI k=5 |
|---|---|---|---|---|---|---|
| **TRACE** | 0.503 [0.462, 0.544] | 0.765 [0.727, 0.805] | 0.892 [0.864, 0.918] | 0.706 [0.603, 0.815] | 0.886 [0.828, 0.942] | 0.959 [0.934, 0.987] |
| direct | 0.528 | 0.771 | 0.886 | 0.745 | 0.898 | 0.964 |
| step-by-step | 0.464 | 0.734 | 0.857 | 0.621 | 0.865 | 0.941 |
| fixed first | 0.485 | 0.749 | 0.858 | 0.687 | 0.892 | 0.957 |
| fixed last | 0.443 | 0.738 | 0.856 | 0.694 | 0.891 | 0.955 |
| population | 0.492 | 0.788 | 0.893 | 0.664 | 0.857 | 0.947 |
| path-dependent | 0.497 | 0.788 | 0.902 | 0.687 | 0.840 | 0.934 |
| AR(1)-SHAP | 0.256 | 0.472 | 0.567 | 0.444 | 0.661 | 0.840 |

**Paired TRACE minus method** (> 0 means TRACE recovers more; values for k = 1 / 3 / 5):
- **vs direct:** PPMI -0.025* / -0.006 / +0.005; ADNI -0.040* / -0.012 / -0.005.
  - Direct wins at k = 1, as expected: it is built for exactly that pair.
- **vs step-by-step:** PPMI +0.039* / +0.031* / +0.035*; ADNI +0.084* / +0.021 / +0.018.
- **vs population:** PPMI +0.012 / -0.022 (CI -0.046 to -0.000, borderline) / -0.001; ADNI +0.042 / +0.029 / +0.012.
- **vs path-dependent:** PPMI +0.007 / -0.023 (CI -0.048 to 0.000) / -0.010; ADNI +0.019 / +0.045* / +0.025*.
- **vs AR(1)-SHAP:** TRACE +0.13 to +0.32, all significant.

(\* = 95% CI excludes 0.) **Conclusions are the same at eps = 0.01, 0.1 and 0.25.** Figure: `results/figures/fig_faithfulness.png`.

## 6c. Sanity check (Stage 8): shuffled labels

Models were retrained with each patient's label shuffled across patients (real visits, shuffled labels).

| | PPMI | ADNI M_minus |
|---|---|---|
| Shuffled-label model, locked-test BA | 0.526 (chance) | 0.516 (chance) |
| Per-visit agreement of TRACE tallies with the real model (Spearman, mean) | -0.02 | -0.11 |
| Per-feature importance ranking, mean abs(u) (Spearman) | 0.30 | 0.46 |
| Total credit, real / shuffled | 2.58 / 2.01 | 4.13 / 0.69 |

**Reading:**
- At the level of individual visits, the explanations fall apart, as they should.
- A weak-to-moderate feature-level ranking survives, because features that change often can pick up credit even from a model with no real signal.

## 6d. Clinical and descriptive analyses (Stage 9; descriptive only, feature groups are a DRAFT)

**PPMI model-predicted subtype flips** (the score crosses the threshold between consecutive visits): 233 flips (150 towards PIGD, 83 towards TD).

Mean share of each flip's TRACE credit by group:

| Group | Share |
|---|---|
| motor daily-living (MDS-UPDRS II) | 35.9% |
| non-motor questionnaires | 27.6% |
| **medication** | **14.3%** |
| time | 12.1% |
| cognition | 10.1% |
| demographics | 0% |

- **Medication does not dominate** (Simuni et al. 2016 found therapy did not drive subtype change). It still takes about one-seventh of the credit, and medication can track severity.
- Top features: NP2RISE 10.6%, MCATOT 10.1%, LEDD_TOTAL_MG 9.7%.
- **Model flips and true-label flips rarely coincide:** of 99 true TD <-> PIGD label flips among 799 consecutive labelled pairs, only 18 are also model flips. This is consistent with the model's modest accuracy.

**Measurement accounting:**
- **PPMI:** 44.5% of total TRACE credit comes from comparisons where a feature's recorded/missing status differs.
- **ADNI:** 88.4% (by phase: ADNI1 87%, ADNIGO 89%, ADNI2 92%, ADNI3 84%, ADNI4 82%).
- **So in ADNI, most of what moves the model's score between visits is which tests were done, not their values.** This matters for interpretation and is a finding in itself.

**Within- vs between-patient split of population SHAP:**
- The identity holds: max error 3.6e-8 (PPMI) and 5.1e-8 (ADNI).
- The within-patient (TRACE) share of total credit is 54% (PPMI) and 49% (ADNI).
- Demographic features carry only between-patient credit (TRACE 0), as theory says.

Headline figures: section 11.

## 7. Decision checklist (brief Section 6): current status

| # | Condition | Status |
|---|---|---|
| 1 | G1-G4 pass on PPMI, M_minus (and M_full) | **Yes**, all five models, with the two documented rule changes |
| 2 | Size of the problem clearly above seed noise | **ADNI: yes** (all variants). **PPMI: borderline** (first->last CI touches 0; labelled-only not significant) |
| 3 | TRACE more stable than chaining, paired CI excluding 0 | **Mixed.** Deleting a visit: **yes** for intermediate deletions in all models (TRACE moves 3-4x less); when the first or last visit is deleted, chaining does not move and TRACE moves slightly. Retraining: **yes on PPMI** (+0.041); **no on ADNI**, where TRACE is very slightly less stable than chaining (-0.007 to -0.011, CI excludes 0) |
| 4 | TRACE not clearly worse than the best baseline on faithfulness | **Yes, with one borderline case.** TRACE is only clearly below direct comparison at k = 1, which is expected. On PPMI at k = 3, population and path-dependent SHAP are slightly ahead, with the CI just touching 0 |
| 5 | Honest answer to the fixed first-visit question | **Yes.** The fixed first-visit reference is unaffected by intermediate or last deletions, but moves 0.15 (PPMI) to 0.36 (ADNI) when the first visit is deleted, vs TRACE's 0.013-0.045 |
| 6 | Sanity check (shuffled labels) | **Yes at visit level** (BA 0.53 / 0.52; per-visit agreement about 0). **Partly at feature level** (ranking Spearman 0.30 / 0.46) |

## 8. Deviations from the brief

1. **Code locations.** Code is in `src/trace_pd/explain/trace_engine.py` and `src/trace_pd/evaluation/trace_*.py`, not `src/trace/engine.py`: a package named `trace` would shadow Python's built-in module. The config is `configs/trace.yaml`. Both accepted.
2. **shap compatibility.**
   - shap 0.49.1 (the newest release) cannot read XGBoost 3.1.2 models. It is fixed on shap's GitHub but not yet released.
   - shap therefore explains a verified *copy* of each model whose saved form states the base score as a plain number. The copy is checked to be identical to the original apart from that text.
   - All shap explainers use it. Model files are unchanged (hashes).
3. **G1 measured against the exact 64-bit tree sum** (user decision; both numbers reported).
4. **G4 for PPMI at up to 13 changed features** (user decision).
5. **Added pilot-code cross-check** (user request).
6. **Fast exact algorithm for retrained models.**
   - The 10 seed models in Stages 4 and 7 use the pilot's exact leaf-path algorithm instead of per-reference shap calls, which would take hours.
   - It is verified equal to the shap route on **every** pair of every patient: max 2.8e-8 (PPMI), 4.3e-8 (ADNI).
   - All main TRACE results use shap interventional TreeSHAP, as the brief requires.
7. **ADNI models.**
   - M_minus is primary (matches the pilot and the proposal). M_minus_LM and M_minus2 are variants; M_full is a sensitivity run (user decision).
   - LOGMEM_DELAYED was flagged because, as far as we know, it is an ADNI enrolment cut-off test. **This should be checked against the ADNI procedures manual.**
8. **Labelled-visits-only sensitivity runs added** (user request). Primary analysis uses all visits of each patient (accepted).
9. **Baselines not run.** Asymmetric Shapley is skipped (no implementation in the repo; user decision). FA-SHAP is skipped (no forecasts exist). DynSHAP and C-SHAP are out of scope per the brief.
10. **AR(1)-SHAP scope.**
    - Primary models only.
    - Excluded from retraining stability (about 1 s per visit per model).
    - Undefined at a patient's first visit, so those pairs are excluded for AR(1) only.
    - After deleting the first visit, pairs involving visit 2 are also undefined for AR(1).
11. **Stage 4 definitions.** "Total disagreement" and "clear winner" reuse the pilot's definitions exactly. Seed noise uses 10 new seeds (0-9); the main models use seed 42.
12. **Headline figures use two pre-set selection rules.**
    - Rule A (as planned) picked patients whose change was mostly missing data: PPMI Patient A has 25 features recorded at the first visit and 3 at the last.
    - Rule B (added) applies the same rule within well-recorded visits.
    - Both are shown and labelled.
13. **Added analyses not in the brief:** the missing-data check (4.5) and credit to never-changing features (6.3).
14. **shap background subsampling bug, found and fixed.**
    - shap silently cut the population background from 500 to 100 visits.
    - The Stage 9 identity check caught it.
    - Fixed by passing all 500 rows explicitly. Every population-based number was recomputed; the numbers above are corrected. TRACE itself was never affected.
15. **eps and feature groups were still open** when Stages 6 and 9 ran. eps = 0.05 was used, with 0.01 / 0.1 / 0.25 sensitivity (same conclusions). The draft feature groups were used, with per-feature numbers also reported.
16. **Stage 8 shuffling:** each development patient's label is shuffled across patients and applied to all of that patient's training visits.

## 9. Open decisions

1. **Confirm faithfulness eps** (0.05 used; results are the same at 0.01-0.25).
2. **Confirm feature groups** (`configs/feature_groups.yaml`; draft used in Stage 9).
3. **How to frame the population-SHAP comparison.** It is the strongest competitor on stability (deletion and retraining). TRACE's advantage over it is zero credit to unchanged features (about 3-4% vs 0) and a within-patient meaning. This needs a decision on the paper's main claim.
4. **How to present PPMI.** The effect is weak and does not hold with labelled visits only, probably because of the weak model.
5. **Interpret ADNI measurement accounting.** 88% of score movement comes from comparisons where which tests were recorded changed. Should the paper lead with this?
6. **Verify** in ADNI documentation that LOGMEM_DELAYED is a diagnostic cut-off.

## 10. Metric definitions (one line each)

- **G1 add-up:** largest abs(sum of credits - actual score change) over every pair; also for the full trajectory (score minus the patient's own average).
- **G2 consistency:** largest abs(psi(s->t) + psi(t->q) - psi(s->q)), plus agreement of three independent ways of computing TRACE.
- **G3 zeros:** largest abs(TRACE credit) for features that never change for the patient, or are never used by the model.
- **G4 brute force:** largest difference between TRACE's pairwise values and exact Shapley values from scoring every 2^k mixed visit with XGBoost.
- **Pilot-code match:** largest difference from the pilot's independent exact algorithm.
- **Direct:** Baseline Shapley explanation comparing visit s directly with visit t.
- **Step-by-step (chain):** sum of the direct explanations of consecutive visits from s to t.
- **Top cause:** the feature with the largest abs(credit).
- **Top cause differs:** two explanations of the same change name different top causes.
- **Clear top cause:** the direct explanation's second-largest abs(credit) is under 80% of the largest.
- **Total disagreement:** sum over features of abs(chain - direct), divided by the sum of abs(direct).
- **Seed noise:** how often two models retrained with different seeds name different top causes for the same direct comparison.
- **TRACE change rate:** how often TRACE's top cause differs from the direct explanation's.
- **Impossibility cost:** how often a feature unchanged between two visits still gets a large TRACE credit.
- **Trust score I:** share of the pairwise explanations' squared size that is contradiction removed by TRACE (0 = fully consistent).
- **Efficiency gap:** abs(sum of a method's credits - actual score change).
- **Deletion stability:** largest per-feature movement of a pair's explanation when one other real visit is deleted.
- **Proposition 4 bound:** TRACE can move by at most 4R/(T-1), where R bounds the model's output range.
- **Retraining stability:** Spearman correlation across features of the same pair's explanation between two retrained models.
- **Never-changing credit:** share of a method's credit given to features identical at every visit of the patient.

## 11. Figures made (all aggregate or anonymised; no IDs, dates, ages or raw values)

| File | Content |
|---|---|
| `results/figures/fig_headline_ppmi_routes.png` | PPMI Patient A (rule A): first->last credit by feature, step-by-step vs direct vs TRACE |
| `results/figures/fig_headline_ppmi_curves.png` | PPMI Patient A: TRACE feature curves summing exactly to the score curve; recorded-feature counts per visit |
| `results/figures/fig_headline_ppmi_wellrecorded_routes.png` | PPMI Patient B (rule B, well-recorded visits): step-by-step names LEDD_TOTAL_MG; direct and TRACE name YRS_SINCE_DIAGNOSIS |
| `results/figures/fig_headline_ppmi_wellrecorded_curves.png` | PPMI Patient B feature curves |
| `results/figures/fig_headline_adni_minus_routes.png` | ADNI Patient A (rule A) |
| `results/figures/fig_headline_adni_minus_curves.png` | ADNI Patient A feature curves |
| `results/figures/fig_headline_adni_minus_wellrecorded_routes.png` | ADNI Patient B (rule B): step-by-step names ECOG_PARTNER_TOTAL; direct and TRACE name LOGMEM_DELAYED |
| `results/figures/fig_headline_adni_minus_wellrecorded_curves.png` | ADNI Patient B feature curves |
| `results/figures/fig_faithfulness.png` | Faithfulness curves (k = 1-5), all methods, both cohorts, with 95% CIs |
| `results/stage4/fig_I_hist_<run>.png` (6 files) | Trust score I histograms: PPMI, M_minus, M_minus_LM, M_minus2, and the two labelled-only runs |

Full report: `results/REPORT.md`. Detailed tables: `results/gate/GATE.md`, `results/stage4/STAGE4.md`, `results/stage5/STAGE5.md`, `results/stage6/STAGE6.md`, `results/stage7/STAGE7.md`, `results/stage8/STAGE8.md`, `results/stage9/STAGE9.md`. Log: `results/PROGRESS.md`.
