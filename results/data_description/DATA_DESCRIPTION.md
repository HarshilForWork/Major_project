# Data description: PPMI and ADNI (TRACE)

Computed from the locked processed data, the data dictionaries and the saved models by `src/trace_pd/evaluation/trace_data_description.py`. Aggregates only; no patient identifiers.

## PPMI (Parkinson's)

Cohort file: 439 patients, 6,922 visits. Visit-level label counts: TD 3,016, PIGD 2,004, no label 1,284, INDETERMINATE 618.

### 1. Model input / output

- **Target (Y):** `LABEL`, the motor subtype at a visit (Stebbins et al. 2013). TREMOR_SCORE = mean of 11 tremor items, PIGD_SCORE = mean of 5 gait/balance items (all items required). Ratio >= 1.15 -> TD, <= 0.90 -> PIGD, otherwise Indeterminate; if PIGD_SCORE = 0, TD when tremor > 0, else Indeterminate.
- **Model task:** binary TD vs PIGD; Indeterminate and unlabelled visits are not used for training. **PIGD is coded 1**, TD 0. Output: raw log-odds of PIGD; predicted PIGD when > 0 (probability 0.5).
- **Inputs (X):** 26 features (saved model expects 26); listed in section 3.
- **Training data:** development split only (locked test patients excluded): 4,002 labelled visits from 351 patients (TD 2,405 / PIGD 1,597); balanced sample weights; XGBoost, 300 trees, depth 3, seed 42.
- **Locked test set:** 88 patients (88 with visits in the file); 1,018 labelled TD/PIGD test visits; balanced accuracy 0.676. TRACE analyses use the 88 test patients with >= 3 visits.

### 2. Slide table

| Column | Role | Description | % missing |
|---|---|---|---|
| `LABEL` | Target | TD / PIGD / Indeterminate — the target (Y) | 18.5% |
| `PATNO` | Key (patient) | Patient identifier | 0.0% |
| `VISIT_MONTH` | Key/Time | Months since baseline (derived) | 0.0% |
| `NP2RISE` | Predictor: motor daily living | Getting out of a bed, car or deep chair | 8.0% |
| `NP1PTOT` | Predictor: non-motor questionnaire | MDS-UPDRS Part I total (self-completed) | 8.0% |
| `LEDD_TOTAL_MG` | Predictor: medication | Levodopa-equivalent daily dose (date-interval join) | 8.2% |
| `MCATOT` | Predictor: cognition | MoCA total | 47.7% |
| `YRS_SINCE_DIAGNOSIS` | Predictor: time | Years since diagnosis (derived) | 8.2% |
| `SEX` / `HANDED` | Predictor: demographic | Sex; Handedness | 0.0% / 0.0% |
| `NP3RTARU` | Banned: one of the 11 tremor items that define Y | Rest tremor amplitude, right upper limb | 16.6% |
| `NP2WALK` | Banned: one of the 5 PIGD items that define Y | Walking and balance (self-reported) | 8.1% |
| `NP3TOT` | Banned: Part III total, contains the label items (likewise NP2PTOT, NHY) | Part III total — contains the 13 items above | 18.5% |

### 3. Model input features

Groups are the DRAFT in `configs/feature_groups.yaml` (not yet confirmed). % missing = share of all visits in the cohort file with no value; training-visit missingness in the last column.

| # | Feature | Group | Description | % missing (all visits) | % missing (training visits) |
|---|---|---|---|---|---|
| 1 | `HANDED` | demographic / static | Handedness | 0.0% | 0.0% |
| 2 | `SEX` | demographic / static | Sex | 0.0% | 0.0% |
| 3 | `AGE_AT_VISIT` | time | Age at visit | 0.4% | 0.1% |
| 4 | `YRS_SINCE_DIAGNOSIS` | time | Years since diagnosis (derived) | 8.2% | 0.0% |
| 5 | `YRS_SINCE_SYMPTOM_ONSET` | time | Years since symptom onset (derived) | 9.2% | 1.3% |
| 6 | `LEDD_TOTAL_MG` | medication | Levodopa-equivalent daily dose (date-interval join) | 8.2% | 0.0% |
| 7 | `N_CONMEDS` | medication | Count of active concomitant medications | 8.2% | 0.0% |
| 8 | `PDTRTMNT` | medication | On PD treatment at this visit | 8.2% | 0.0% |
| 9 | `NP2DRES` | motor daily living (MDS-UPDRS II) | Dressing | 8.0% | 0.0% |
| 10 | `NP2EAT` | motor daily living (MDS-UPDRS II) | Eating tasks | 8.1% | 0.0% |
| 11 | `NP2HOBB` | motor daily living (MDS-UPDRS II) | Hobbies and other activities | 8.0% | 0.0% |
| 12 | `NP2HWRT` | motor daily living (MDS-UPDRS II) | Handwriting | 8.0% | 0.0% |
| 13 | `NP2HYGN` | motor daily living (MDS-UPDRS II) | Hygiene | 8.0% | 0.0% |
| 14 | `NP2RISE` | motor daily living (MDS-UPDRS II) | Getting out of a bed, car or deep chair | 8.0% | 0.0% |
| 15 | `NP2SALV` | motor daily living (MDS-UPDRS II) | Saliva and drooling | 8.0% | 0.0% |
| 16 | `NP2SPCH` | motor daily living (MDS-UPDRS II) | Speech | 8.0% | 0.0% |
| 17 | `NP2SWAL` | motor daily living (MDS-UPDRS II) | Chewing and swallowing | 8.0% | 0.0% |
| 18 | `NP2TURN` | motor daily living (MDS-UPDRS II) | Turning in bed | 8.0% | 0.0% |
| 19 | `ESS_TOTAL` | non-motor questionnaires | Epworth Sleepiness Scale item sum | 33.1% | 30.3% |
| 20 | `GDS_TOTAL` | non-motor questionnaires | Geriatric Depression Scale item sum | 33.3% | 30.5% |
| 21 | `NP1PTOT` | non-motor questionnaires | MDS-UPDRS Part I total (self-completed) | 8.0% | 0.0% |
| 22 | `NP1RTOT` | non-motor questionnaires | MDS-UPDRS Part I total (rater-completed) | 7.9% | 0.0% |
| 23 | `RBDSQ_TOTAL` | non-motor questionnaires | REM Sleep Behaviour Disorder screening item sum | 33.1% | 30.2% |
| 24 | `SCOPA_AUT_TOTAL` | non-motor questionnaires | SCOPA-AUT item sum | 33.2% | 30.3% |
| 25 | `STAI_TOTAL` | non-motor questionnaires | State-Trait Anxiety Inventory item sum | 33.4% | 30.6% |
| 26 | `MCATOT` | cognition | MoCA total | 47.7% | 44.9% |

### 4. Banned and excluded columns

**Dictionary bucket BANNED_LABEL_DEFINING (19):** `NHY`, `NP2FREZ`, `NP2PTOT`, `NP2TRMR`, `NP2WALK`, `NP3FRZGT`, `NP3GAIT`, `NP3KTRML`, `NP3KTRMR`, `NP3PSTBL`, `NP3PTRML`, `NP3PTRMR`, `NP3RTALJ`, `NP3RTALL`, `NP3RTALU`, `NP3RTARL`, `NP3RTARU`, `NP3RTCON`, `NP3TOT`.

**Other non-input buckets:** ADMIN (6): APPRDX, COHORT, GENETIC_COHORT, PDDXDT, PDSTATE_USED, SXDT; KEY (4): EVENT_ID, INFODT, PATNO, VISIT_MONTH; RESOURCE_DEPENDENT_FEATURE (12): APOE, CAUDATE_REF_CWM, DATSCAN_VISINTRP, GBA, LRRK2, PARK7, PINK1, PRKN, PUTAMEN_REF_CWM, SNCA, STRIATUM_REF_CWM, VPS35; TARGET (8): LABEL, LABEL_FLIPPED_NEXT, MONTHS_TO_NEXT_VISIT, NEXT_LABEL, NEXT_VISIT_MONTH, PIGD_SCORE, TD_PIGD_RATIO, TREMOR_SCORE.

**Check:** banned columns among model inputs = none; every input is in CHEAP_FEATURE = yes; saved model input count = 26 = code list 26.

## ADNI (Alzheimer's), primary model M_minus

Cohort file: 4,470 patients, 24,302 visits. Visit-level label counts: no label 9,966, MCI 5,903, CN 5,652, Dementia 2,781.

### 1. Model input / output

- **Target (Y):** `LABEL`, the clinical diagnosis at a visit, from DXSUM.DIAGNOSIS (CN / MCI / Dementia).
- **Model task:** binary CN vs Dementia; MCI and unlabelled visits are not used for training. **Dementia is coded 1**, CN 0. Output: raw log-odds of dementia; predicted Dementia when > 0.
- **Inputs (X):** 21 features (saved model expects 21); listed in section 3.
- **Training data:** development split only (locked test patients excluded): 6,748 labelled visits from 2,121 patients (CN 4,547 / Dementia 2,201); balanced sample weights; XGBoost, 300 trees, depth 3, seed 42.
- **Locked test set:** 894 patients (894 with visits in the file); 1,685 labelled CN/Dementia test visits; balanced accuracy 0.930. TRACE analyses use the 475 test patients with >= 3 visits.

### 2. Slide table

| Column | Role | Description | % missing |
|---|---|---|---|
| `LABEL` | Target | Clinical diagnosis CN / MCI / Dementia (DXSUM.DIAGNOSIS) | 41.0% |
| `RID` | Key (patient) | Patient identifier (key) | 0.0% |
| `VISIT_MONTH` | Key/Time | Months since baseline (visit time) | 0.0% |
| `LOGMEM_DELAYED` | Predictor: cognitive test | Logical Memory delayed recall (NEUROBAT.LDELTOTAL) | 61.4% |
| `ECOG_PARTNER_TOTAL` | Predictor: function | Everyday Cognition, study-partner total (ECOGSP) | 88.6% |
| `NPI_TOTAL` | Predictor: mood/behaviour | Neuropsychiatric Inventory-Q total (NPIQ.NPISCORE) | 68.5% |
| `HEIGHT_CM` | Predictor: vitals/body | Height, cm (VITALS; unit-converted; carried forward/back within patient) | 41.7% |
| `AGE_AT_VISIT` | Predictor: time | Age at visit (derived from PTDEMOG birth date and exam date) | 3.8% |
| `SEX` / `HANDED` | Predictor: demographic | Sex (PTDEMOG.PTGENDER); Handedness (PTDEMOG.PTHAND) | 3.5% / 3.7% |
| `CDGLOBAL` | Banned: CDR exam, which the diagnosis is informed by (CDR 0 -> CN 89.7% of the time) | CDR global score | 52.4% |
| `MMSE_TOTAL` | Excluded in M_minus: an ADNI diagnostic criterion (MMSE ranges gate CN / dementia) | Mini-Mental State Exam total (MMSE.MMSCORE) | 53.2% |
| `FAQ_TOTAL` | Excluded in M_minus: functional impairment, part of the dementia diagnosis | Functional Activities Questionnaire total (FAQ.FAQTOTAL) | 44.7% |

### 3. Model input features

Groups are the DRAFT in `configs/feature_groups.yaml` (not yet confirmed). % missing = share of all visits in the cohort file with no value; training-visit missingness in the last column.

| # | Feature | Group | Description | % missing (all visits) | % missing (training visits) |
|---|---|---|---|---|---|
| 1 | `EDUCATION_YEARS` | demographic / static | Years of education (PTDEMOG.PTEDUCAT) | 3.8% | 3.5% |
| 2 | `HANDED` | demographic / static | Handedness (PTDEMOG.PTHAND) | 3.7% | 3.6% |
| 3 | `SEX` | demographic / static | Sex (PTDEMOG.PTGENDER) | 3.5% | 3.4% |
| 4 | `AGE_AT_VISIT` | time | Age at visit (derived from PTDEMOG birth date and exam date) | 3.8% | 3.4% |
| 5 | `BP_DIASTOLIC` | vitals / body | Diastolic BP (VITALS.VSBPDIA) | 42.4% | 7.7% |
| 6 | `BP_SYSTOLIC` | vitals / body | Systolic BP (VITALS.VSBPSYS) | 42.4% | 7.7% |
| 7 | `HEIGHT_CM` | vitals / body | Height, cm (VITALS; unit-converted; carried forward/back within patient) | 41.7% | 6.7% |
| 8 | `PULSE` | vitals / body | Pulse (VITALS.VSPULSE) | 42.5% | 7.8% |
| 9 | `WEIGHT_KG` | vitals / body | Weight, kg (VITALS; unit-converted, -1 sentinel -> missing) | 42.6% | 8.2% |
| 10 | `GDS_TOTAL` | mood / behaviour | Geriatric Depression Scale total (GDSCALE.GDTOTAL) | 56.7% | 33.3% |
| 11 | `NPI_TOTAL` | mood / behaviour | Neuropsychiatric Inventory-Q total (NPIQ.NPISCORE) | 68.5% | 60.9% |
| 12 | `BOSTON_NAMING` | cognitive tests | Boston Naming Test (NEUROBAT.BNTTOTAL) | 63.5% | 43.8% |
| 13 | `CATEGORY_FLUENCY_ANIMALS` | cognitive tests | Category fluency, animals (NEUROBAT.CATANIMSC) | 46.2% | 13.1% |
| 14 | `DIGIT_SPAN` | cognitive tests | Digit span score (NEUROBAT.DIGITSCOR) | 84.4% | 73.6% |
| 15 | `LOGMEM_DELAYED` | cognitive tests | Logical Memory delayed recall (NEUROBAT.LDELTOTAL) | 61.4% | 39.1% |
| 16 | `LOGMEM_IMMEDIATE` | cognitive tests | Logical Memory immediate recall (NEUROBAT.LIMMTOTAL) | 61.2% | 38.6% |
| 17 | `MOCA_TOTAL` | cognitive tests | MoCA total (MOCA.MOCA) | 83.2% | 70.1% |
| 18 | `TRAILS_A_SEC` | cognitive tests | Trail Making A, seconds (NEUROBAT.TRAASCOR) | 46.7% | 14.0% |
| 19 | `TRAILS_B_SEC` | cognitive tests | Trail Making B, seconds (NEUROBAT.TRABSCOR) | 48.2% | 17.3% |
| 20 | `ECOG_PARTNER_TOTAL` | function / daily living | Everyday Cognition, study-partner total (ECOGSP) | 88.6% | 81.0% |
| 21 | `ECOG_PATIENT_TOTAL` | function / daily living | Everyday Cognition, self-report total (ECOGPT) | 88.6% | 80.9% |

### 4. Banned and excluded columns

**Dictionary bucket BANNED_LABEL_DEFINING (8):** `CDCARE`, `CDCOMMUN`, `CDGLOBAL`, `CDHOME`, `CDJUDGE`, `CDMEMORY`, `CDORIENT`, `CDRSB`.

**Excluded from this model by the model definition (dictionary bucket CHEAP_FEATURE, dropped in `trace_run.MODELS['adni_minus']`):** `FAQ_TOTAL`, `MMSE_TOTAL`.

**Other non-input buckets:** ADMIN (2): COLPROT, ORIGPROT; KEY (4): EXAMDATE, RID, VISCODE2, VISIT_MONTH; RESOURCE_DEPENDENT_FEATURE (1): APOE_GENOTYPE; TARGET (5): LABEL, LABEL_FLIPPED_NEXT, MONTHS_TO_NEXT_VISIT, NEXT_LABEL, NEXT_VISIT_MONTH.

**Check:** banned columns among model inputs = none; every input is in CHEAP_FEATURE = yes; saved model input count = 21 = code list 21.

## Things that do not match the dictionaries or the project docs

- Automated checks: no differences between dictionary % present and the data, no undocumented columns, every model input is in CHEAP_FEATURE, no banned column is an input.
- **ADNI MMSE_TOTAL and FAQ_TOTAL are not in the dictionary's banned bucket.** They are CHEAP_FEATURE in `adni_dx_dictionary.csv`; they are excluded only by the M_minus model definition in code. The dictionary's only BANNED_LABEL_DEFINING columns are the 8 CDR columns.
- **PPMI GENETIC_COHORT** is described in CLAUDE.md as banned (a recruitment flag that alone scored 0.629), but the dictionary puts it in the ADMIN bucket, not BANNED_LABEL_DEFINING. It is not a model input either way.
- **ADNI HEIGHT_CM is carried forward/back within each patient** during preprocessing (preprocess_adni.py, step 7b). This is a small exception to the 'missing stays missing' rule and should be disclosed.
- **ADNI has no column-description file** (PPMI has one); ADNI descriptions above were taken from preprocess_adni.py.
- **Feature groups are a draft** (`configs/feature_groups.yaml`, not yet confirmed).
- **Older slides say 5,742 labelled PPMI rows;** the current data has 5,638 (30 Sep 2026 preprocessing fix).
