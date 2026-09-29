# Model Card — TRACE-PD Subtype Classifier

Format after Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).

---

## Model details

| | |
|---|---|
| **Name** | TRACE-PD motor-subtype classifier (Layer 1) |
| **Version** | 0.1.0 — research prototype |
| **Type** | gradient-boosted trees (XGBoost / sklearn `HistGradientBoostingClassifier`); logistic regression baseline |
| **Task** | predict TD / PIGD / Indeterminate, or binary TD vs PIGD, per patient-visit |
| **Hyperparameters** | XGBoost: `max_depth=3`, `n_estimators=300`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5` · HGB: `max_depth=3`, `max_iter=200`, `learning_rate=0.06` · balanced sample weights |
| **Code** | `src/trace_pd/models/` |
| **Developers** | Aishwarya Deshmukh, Rutu Mehta, Harshil Bhanushali · guide Dr. Kriti Srivastava · DJSCE |
| **Status** | Layer 1 only. The confidence, tracking and transition layers are designed but not built. |

## Intended use

| | |
|---|---|
| **Primary use** | **research** into whether the published TD/PIGD subtype can be estimated from low-cost clinical data, without the specialist motor exam |
| **Intended users** | researchers; eventually, as a **triage aid** for clinicians in settings where the full MDS-UPDRS isn't routinely completed |
| **Intended decision** | *"Is the subtype clear enough to plan around, or should the full motor exam be done now?"* |

### Out of scope

- ❌ **Clinical diagnosis or treatment decisions.** This isn't a medical device and hasn't been
  prospectively validated.
- ❌ **Replacing the motor exam.** If the full exam is done, the Stebbins formula gives the
  answer directly and the model adds nothing.
- ❌ **Diagnosing Parkinson's.** The model assumes a PD diagnosis.
- ❌ **Use outside the PPMI population** — later-stage patients, other countries, non-research
  clinics — without external validation.

## Training data

| | |
|---|---|
| Source | PPMI via LONI IDA, extract 16 Aug 2026 |
| Cohort | 439 PD patients (242 sporadic, 197 genetic-cohort) |
| Unit | patient-visit; 5,638 labelled rows |
| Label | Stebbins et al. 2013 ratio over 16 MDS-UPDRS items |
| Inputs | 26 `CHEAP_FEATURE` columns (`docs/data_dictionary.md`); `GENETIC_COHORT` removed |
| Excluded | 19 label-defining columns (banned), 12 imaging / genetic columns (held out) |

## Evaluation

5-fold `StratifiedGroupKFold` grouped by patient. Balanced accuracy. 3-class chance 0.333,
binary chance 0.500.

| Setting | Logistic | Boosted |
|---|---|---|
| Baseline visit, 3-class | 0.450 | 0.458 |
| All visits, 3-class | 0.461 | 0.452 |
| All visits, TD vs PIGD | **0.694** | **0.689** |
| **Sporadic, baseline visit, TD vs PIGD** | 0.553 | **0.530** |

Leakage positive control (formula items fed back in): 0.905 accuracy — confirms the pipeline
works and the task is genuinely hard.

## Factors and subgroup behaviour

| Factor | Finding |
|---|---|
| **Recruitment arm** | `GENETIC_COHORT` **alone** scored 0.629 on the binary task — **now removed from X**; removing it cost ~0.013 |
| **Label fragility** | 39.5% of labels are one scoring point from a different label; conformal coverage is 84.8% on fragile visits vs 94.2% on robust ones |
| **Exam medication state** | 25.6% of same-day OFF/ON exams give a different label; 1,730 labelled visits use an ON exam |
| **Visit** | first-visit performance is much lower than pooled; pooling repeated visits inflates the numbers |
| **Class** | Indeterminate (11.0%) is effectively unlearnable — 76% unstable visit to visit |
| **Sex, age** | ⏳ fairness breakdown not yet run |

## Ethical considerations

- **Risk of false reassurance.** A confident "TD" call on a patient who is actually PIGD could
  delay closer monitoring. That's why the design pairs every call with a calibrated confidence
  set and a flip risk, and why a completed exam always overrides the model.
- **Population mismatch.** PPMI is early-stage, specialist-centre, mostly high-income. The
  intended beneficiaries — patients in low-resource settings, including India — aren't
  represented in training.
- **Proxy leakage.** The axial items `NP2RISE` and `NP2TURN` measure the same construct as the
  PIGD score. The model may be partly re-deriving PIGD from near-duplicate inputs.
- **Data governance.** Trained on DUA-restricted PPMI data. The repository must stay private.

## Caveats and recommendations

1. ~~Remove `GENETIC_COHORT`~~ — done 28 Sep.
2. Report the sporadic first-visit result as the headline, not the pooled binary result.
3. Run a sensitivity analysis excluding the 845 zero-PIGD edge-case rows.
4. Run the fairness breakdown by sex and age.
5. External validation on a non-PPMI cohort before any clinical claim.
