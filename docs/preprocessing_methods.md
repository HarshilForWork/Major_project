# Preprocessing Methods — TD/PIGD Motor Subtype Dataset

**Script:** `01_preprocess_tdpigd.py`
**Data source:** PPMI (Parkinson's Progression Markers Initiative), LONI IDA extract dated 16 Aug 2026
**Outputs:** `ppmi_tdpigd_long.csv`, `ppmi_tdpigd_dictionary.csv`, `ppmi_tdpigd_runlog.txt`
**Last run:** 10 Sep 2026

This document records every preprocessing decision, its rationale, and its
consequence, so the methods section of the paper can be written directly from it.

---

## 1. Final dataset at a glance

| Property | Value |
|---|---|
| Patients | **439** (PPMI PD cohort) |
| Rows (patient-visits, scheduled visits only) | **6,922** |
| Columns | 75 |
| Rows with a computable TD/PIGD label | **5,742 (83.0%)** |
| Duplicate `PATNO`+`EVENT_ID` keys | 0 (asserted in code) |
| Label distribution (visit level) | TD 52.6%, PIGD 36.6%, Indeterminate 10.8% |
| Visit-to-visit label flip rate | **29.1%** |
| Patients who ever changed label | 355 / 439 (**80.9%**) |

### Comparison with the superseded pipeline

| | Old (`01_preprocess.py`, slow/mod/fast) | New (TD/PIGD) |
|---|---|---|
| Patients | 229 | **439** |
| Labelled training units | 229 (one label per patient) | **5,742** (one label per visit) |
| Patients usable at the 4th round | 143 | **275** |
| Label origin | K-Means clustering (invented) | Published formula (Stebbins 2013) |

---

## 2. The label: TD / PIGD / Indeterminate

Computed per patient-visit using the Stebbins et al. (2013, *Movement Disorders*)
MDS-UPDRS operationalisation of the Jankovic (1990) tremor/PIGD ratio.

**Tremor score** = mean of 11 items:

| MDS-UPDRS item | Description | Column |
|---|---|---|
| 2.10 | Tremor (patient-reported) | `NP2TRMR` |
| 3.15a / 3.15b | Postural tremor of hands, R / L | `NP3PTRMR`, `NP3PTRML` |
| 3.16a / 3.16b | Kinetic tremor of hands, R / L | `NP3KTRMR`, `NP3KTRML` |
| 3.17a–e | Rest tremor amplitude: RUE, LUE, RLE, LLE, lip/jaw | `NP3RTARU`, `NP3RTALU`, `NP3RTARL`, `NP3RTALL`, `NP3RTALJ` |
| 3.18 | Constancy of rest tremor | `NP3RTCON` |

**PIGD score** = mean of 5 items:

| MDS-UPDRS item | Description | Column |
|---|---|---|
| 2.12 | Walking and balance (patient-reported) | `NP2WALK` |
| 2.13 | Freezing (patient-reported) | `NP2FREZ` |
| 3.10 | Gait | `NP3GAIT` |
| 3.11 | Freezing of gait | `NP3FRZGT` |
| 3.12 | Postural stability | `NP3PSTBL` |

**Classification rule:** `ratio = tremor_score / pigd_score`

- `ratio ≥ 1.15` → **TD**
- `ratio ≤ 0.90` → **PIGD**
- `0.90 < ratio < 1.15` → **INDETERMINATE**

**Edge cases (explicitly defined, since the ratio is otherwise undefined):**

- `pigd_score == 0` and `tremor_score > 0` → **TD** (tremor present, no PIGD signs)
- `pigd_score == 0` and `tremor_score == 0` → **INDETERMINATE** (0/0; asymptomatic on both constructs)

This edge case affects **845 rows (14.7% of labelled rows)** — 709 classified TD,
136 Indeterminate. It is material enough that it must be stated in the paper, and
a sensitivity analysis excluding these rows is advisable.

**Completeness rule:** a label is computed only if **all 16 constituent items** are
present. Rows missing any item receive `NaN` rather than a label derived from a
partial mean. This is why 17% of rows are unlabelled.

**Verification performed:** the label was independently recomputed from
`TREMOR_SCORE` / `PIGD_SCORE` and compared against the stored `LABEL` — 0
mismatches across all 5,742 labelled rows. Ratio ranges per class confirm the
cutoffs were applied correctly (PIGD max = 0.892; Indeterminate spans 0.909–1.136;
TD min = 1.162).

---

## 3. Cohort definition

**Included:** `COHORT == 1` (the PPMI Parkinson's disease cohort) → **439 patients**.

This comprises two PPMI diagnostic sub-codes, both of which are PD:

- `APPRDX == 1` — sporadic PD (n = 242)
- `APPRDX == 5` — genetic-cohort PD (n = 197)

A binary `GENETIC_COHORT` flag is retained so this subgroup can be adjusted for or
excluded downstream. **This matters:** the genetic cohort is enriched for *LRRK2*,
*GBA* and *SNCA* mutation carriers by recruitment design, not by chance, so it is
not a random sample of PD and any genetics-related finding must be interpreted
with that in mind.

**Note on the previous pipeline:** it filtered on `APPRDX == 1` alone, which
silently discarded the 197 genetic-cohort PD patients. That was an unintended
restriction rather than a deliberate design choice.

---

## 4. Visit handling

Visit codes were mapped to months since baseline:

`BL`=0, `V01`=3, `V02`=6, `V03`=9, `V04`=12, `V05`=15, `V06`=18, `V07`=21,
`V08`=24, `V09`=30, `V10`=36, `V11`=42, `V12`=48, `V13`=54, `V14`=60, `V15`=66,
`V16`=72, `V17`=78, `V18`=84, `V19`=90, `V20`=96.

**3,703 rows dropped** for non-scheduled visit codes with no fixed position on the
study timeline: `SC` (screening), `LOG`, `ED` (early discontinuation), `ST`
(symptomatic therapy), `U01`/`U02` (unscheduled), `R06`–`R22` (remote/telephone
check-ins), `V21`/`V22`.

**No minimum follow-up filter was applied.** The previous pipeline required ≥3
years of follow-up because its label was a progression *rate* needing a long
trajectory. The TD/PIGD label is computed independently at each visit, so that
filter is obsolete. Removing it is the single largest contributor to the increase
in usable sample.

---

## 5. ON/OFF medication state (MDS-UPDRS Part III)

Part III is sometimes recorded twice at a visit (`PDSTATE` = `OFF` and `ON`).
Dopaminergic medication suppresses tremor, so ON-state scoring can shift the
tremor/PIGD ratio and flip a patient's computed category.

**Decision:** keep one row per patient-visit, preferring the **OFF-state**
assessment (untreated severity). `PDSTATE_USED` records which state was actually
used for each retained row, so the sensitivity of results to this choice can be
tested later.

---

## 6. Missing data — deliberately NOT imputed

**Missing values are left as `NaN`.** No imputation is performed in preprocessing.

Two reasons:

1. **The previous pipeline's median imputation destroyed a real signal.** It
   median-imputed every numeric column, which flattened DaTscan striatal binding
   ratio to a constant 0.68 across all subtypes, making the imaging validation
   check meaningless.
2. **Imputing before splitting leaks test-fold information into training.** Any
   imputation must happen inside cross-validation folds, not upstream.

XGBoost handles `NaN` natively (learned default split directions), so no
imputation is required for the planned Layer 1 model.

### Missingness is structural, not random

Per the PPMI protocol (confirmed against the published "Milestone-Based Strategy"
paper), the study runs two tiers of visit: **annual visits** (12, 24, 36, 48, 60
months) with the full battery, and lighter **interim visits** (3, 6, 9, 18, 30, 42,
54 months) that administer only a subset of instruments. The observed missingness
matches this exactly:

| Instrument | % present | Explanation |
|---|---|---|
| `SEX`, `HANDED`, `GENETIC_COHORT` | 100% | Static |
| `AGE_AT_VISIT` | 99.6% | Collected every visit |
| Part II items, Part I totals, `PDTRTMNT`, `LEDD_TOTAL_MG` | ~92% | Collected every visit |
| `SCOPA_AUT_TOTAL`, `GDS_TOTAL`, `STAI_TOTAL`, `ESS_TOTAL`, `RBDSQ_TOTAL` | ~67% | Annual / 6-month-plus-annual schedule |
| `MCATOT` (MoCA) | 52.3% | Annual visits only |

**This is missing-by-design, not data corruption.** It should be described as such,
and models must not treat these as randomly missing.

---

## 7. Medication features — bug found and fixed

**Bug (also present in the previous pipeline):** the LEDD and concomitant-medication
logs are *running records*, not visit-linked. Their `EVENT_ID` only ever takes the
values `LOG`, `ED`, or `SC`. Joining them on `PATNO` + `EVENT_ID` against scheduled
visits (`BL`, `V01`, …) therefore matches **nothing** and silently produces an
all-missing column. In the old `ppmi_joined_baseline_table.csv`, `LEDD_TOTAL_MG`
was 100% `NaN` as a result.

**Fix:** each medication record carries `STARTDT` and `STOPDT` (format `MM/YYYY`;
blank `STOPDT` means still ongoing). For every patient-visit, the LEDD of all
medications whose interval covers that visit date is summed. Patients with no
active PD medication on a visit date are assigned `LEDD = 0` (a true zero, not
missing), provided the visit date itself is known.

**Validation that the fix works** — median LEDD by visit month:

| Month | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| Median LEDD (mg) | 0 | 0 | 179 | 100 | 300 | 330 | 400 | 450 | 500 |

Baseline is 0 mg, which independently confirms correctness: PPMI enrols *de novo,
drug-naive* PD patients by design. The monotonic rise thereafter matches expected
treatment escalation. `LEDD_TOTAL_MG` is now 91.8% present (limited only by
`INFODT` visit-date availability), median 500 mg, with 1,006 rows on no PD
medication.

---

## 8. Leakage control — the critical design element

Because the label is a deterministic function of 16 MDS-UPDRS items, any model
that sees those items would merely re-derive arithmetic and report a meaningless
near-perfect score. Feature eligibility is therefore **machine-enforced** via the
`bucket` column of `ppmi_tdpigd_dictionary.csv`, and downstream code must select
features by bucket, never by hand.

| Bucket | n cols | Usable as model input? |
|---|---|---|
| `BANNED_LABEL_DEFINING` | 16 | **Never** |
| `FORMULA_PART2_PATIENT_REPORTED` | 3 | Variant A only |
| `CHEAP_FEATURE` | 27 | Yes — core input set |
| `RESOURCE_DEPENDENT_FEATURE` | 12 | Separate variant only |
| `TARGET` | 8 | No (outcomes) |
| `KEY` / `ADMIN` | 9 | No |

### `BANNED_LABEL_DEFINING` (16 columns)

The 13 Part III formula items, plus three columns that encode the same
information indirectly:

- **`NP3TOT`** — the Part III total sums the tremor and gait items into itself
- **`NP2PTOT`** — the Part II total sums the three Part II formula items into itself
- **`NHY`** (Hoehn & Yahr) — stage ≥3 is *defined* by postural instability, i.e. it
  restates the PIGD construct in another form

The script asserts that no banned column is ever tagged as a feature; the run log
records `Leakage guard: PASSED`.

### The Variant A / Variant B decision (unresolved — needs your call)

The three Part II items (`NP2TRMR`, `NP2WALK`, `NP2FREZ`) are patient-reported *and*
contribute to the label formula — 1 of 11 tremor items, and 2 of 5 PIGD items.

- **Variant A (recommended primary):** include them. Research question — *"given the
  3 cheap patient-reported items, can we infer what the full 16-item formula,
  including the 12 specialist-exam items, would conclude?"* The partial overlap
  **must be disclosed explicitly** in the paper.
- **Variant B (stricter comparison):** exclude them. Pure non-motor prediction.
  Cleaner, harder, expected to perform substantially worse.

### `RESOURCE_DEPENDENT_FEATURE` (12 columns)

DaTscan measures and the gene columns. Held out of the primary model because
survey evidence shows ~94% of Indian clinicians rarely order DaTscan (94.4%) or
genetic testing (94.9%) — the settings this work is meant to serve. Report
separately whether adding them helps.

---

## 9. Transition / label-flip target

For each patient-visit, `NEXT_LABEL` is the label at that patient's following
scheduled visit; `LABEL_FLIPPED_NEXT` is 1 if it differs, 0 if it holds, `NaN` if
either label is unknown. `MONTHS_TO_NEXT_VISIT` records the gap, which varies
(3–12 months) and should be adjusted for.

**Results (4,918 consecutive visit pairs with both labels known):**

| Current label | Flip rate at next visit | n pairs |
|---|---|---|
| TD | 20.9% | 2,744 |
| PIGD | 27.1% | 1,629 |
| Indeterminate | **76.3%** | 545 |
| **Overall** | **29.1%** | 4,918 |

Two observations worth reporting:

1. The overall 29.1% per-visit flip rate is consistent with the published claim
   (cited in the renewed-approach document) that roughly a third to a half of
   patients get reclassified within one to two years — an independent
   corroboration that the label was implemented correctly.
2. Indeterminate is highly unstable (76.3%), which is expected — it is a narrow
   boundary band (0.90 < ratio < 1.15) rather than a distinct clinical entity, so
   small measurement fluctuations push patients out of it. This has a real modelling
   consequence: a three-class flip model will behave very differently from a
   two-class one, and it may be worth reporting TD↔PIGD flips separately from
   flips involving Indeterminate.

---

## 10. Round structure (reshape happens downstream, not here)

The output is deliberately **long format** (one row per patient-visit). The
per-round wide tables are built from it by a downstream reshape, appending a
suffixed copy of the time-varying features per included visit plus between-visit
deltas.

| Round | Visits included | Patients with a computable label at every included visit |
|---|---|---|
| 1 | BL | **438** |
| 2 | BL + 6mo | **349** |
| 3 | BL + 6mo + 12mo | **323** |
| 4 | BL + 6mo + 12mo + 24mo | **275** |
| Alt. annual-only | BL + 12mo | 403 |
| Alt. annual-only | BL + 12mo + 24mo | 343 |

**Recommendation to consider:** the 6-month interim visit is the weak link
(349 vs 438 patients) and the PPMI protocol explains why — it is a lighter interim
visit. An annual-only 3-round design (BL → 12mo → 24mo) retains 343 patients at the
final round versus 275, with more complete instrument coverage at each round.

**Mandatory downstream constraint:** with ~16 rows per patient, a row-wise random
train/test split would place the same patient on both sides and leak. All splits
must be grouped by `PATNO` (`GroupKFold` or equivalent).

---

## 11. Known limitations

1. **Item-sum proxies are not validated clinical scores.** `SCOPA_AUT_TOTAL`,
   `GDS_TOTAL`, `STAI_TOTAL`, `ESS_TOTAL` and `RBDSQ_TOTAL` are plain arithmetic
   sums of raw item codes, because the PPMI download did not include pre-computed
   totals. They are *not* the official reverse-coded scoring for these instruments
   and must not be reported as validated instrument totals — only as monotonic
   severity proxies.
2. **The zero-PIGD edge case is non-trivial** (14.7% of labelled rows). The
   TD assignment for `pigd_score == 0, tremor_score > 0` is our explicit
   convention; a sensitivity analysis excluding these rows should accompany any
   headline result.
3. **Class imbalance.** At baseline: TD 60.7%, PIGD 27.9%, Indeterminate 11.4%.
   Class weighting or per-class reporting is required, as the renewed-approach
   document anticipated.
4. **The genetic cohort is mutation-enriched by recruitment design**, so it is not
   representative of sporadic PD genetics.
5. **17% of rows are unlabelled**, concentrated at later visits (months 54–90) —
   97.5% of unlabelled rows are missing the Part III gait item, reflecting visits
   where the motor exam was not administered plus study attrition.
6. **PPMI is not representative of the target deployment setting.** Participants
   are early-stage, largely from specialised centres in higher-income countries.
   Any claim of usefulness in low-resource settings requires external validation.
7. **`INFODT` is missing for 8.2% of rows**, which caps disease-duration and
   medication feature completeness at ~92%.

---

## 12. Reproducibility

- Deterministic: no random seeds are involved in preprocessing.
- Provenance: `ppmi_tdpigd_runlog.txt` records every filtering step with row and
  patient counts, the exact list of dropped visit codes, label distribution, flip
  rates, bucket counts, and the leakage-guard result.
- Assertions in code: zero duplicate `PATNO`+`EVENT_ID` keys; no banned column in
  any feature bucket. The script fails loudly rather than producing a bad file.
