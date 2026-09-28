# Data Dictionary — `ppmi_tdpigd_long.csv`

> **Generated** by `src/trace_pd/data/export_dictionary.py` (`make docs`). Do not edit by hand.

**75 columns** · 6,922 rows · one row per patient-visit · 439 patients.

Every column's bucket is assigned **by rule** in `preprocess.py`, and training code selects
features **by bucket only**. `tests/test_dataset_contract.py` fails if a label-defining
column ever lands in a feature bucket.

| Bucket | Columns | Used for training |
|---|---|---|
| `CHEAP_FEATURE` | 26 | **yes** |
| `RESOURCE_DEPENDENT_FEATURE` | 12 | no |
| `BANNED_LABEL_DEFINING` | 19 | no |
| `TARGET` | 8 | no |
| `KEY` | 4 | no |
| `ADMIN` | 6 | no |

---

## ✅ Model input (X) — `CHEAP_FEATURE` (26)

*Patient-reported or low-cost clinical measure, plausibly available in a non-specialist setting. Core input set.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `AGE_AT_VISIT` | Age at visit | Age at Visit | numeric | 99.6 | 540 |
| `ESS_TOTAL` | Epworth Sleepiness Scale item sum | Epworth (item sum) | numeric | 66.9 | 25 |
| `GDS_TOTAL` | Geriatric Depression Scale item sum | GDS-15 (item sum) | numeric | 66.7 | 16 |
| `HANDED` | Handedness | Demographics | numeric | 100.0 | 3 |
| `LEDD_TOTAL_MG` | Levodopa-equivalent daily dose (date-interval join) | LEDD Med Log (date interval) | numeric | 91.8 | 753 |
| `MCATOT` | MoCA total | MoCA | numeric | 52.3 | 28 |
| `NP1PTOT` | MDS-UPDRS Part I total (self-completed) | MDS-UPDRS Part I (self-completed) | numeric | 92.0 | 26 |
| `NP1RTOT` | MDS-UPDRS Part I total (rater-completed) | MDS-UPDRS Part I | numeric | 92.1 | 18 |
| `NP2DRES` | Dressing | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2EAT` | Eating tasks | MDS-UPDRS Part II | numeric | 91.9 | 5 |
| `NP2HOBB` | Hobbies and other activities | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2HWRT` | Handwriting | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2HYGN` | Hygiene | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2RISE` | Getting out of a bed, car or deep chair | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2SALV` | Saliva and drooling | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2SPCH` | Speech | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2SWAL` | Chewing and swallowing | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2TURN` | Turning in bed | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `N_CONMEDS` | Count of active concomitant medications | Concomitant Med Log (date interval) | numeric | 91.8 | 45 |
| `PDTRTMNT` | On PD treatment at this visit | MDS-UPDRS Part III | numeric | 91.8 | 2 |
| `RBDSQ_TOTAL` | REM Sleep Behaviour Disorder screening item sum | RBD Screening (item sum) | numeric | 66.9 | 13 |
| `SCOPA_AUT_TOTAL` | SCOPA-AUT item sum | SCOPA-AUT (item sum) | numeric | 66.8 | 88 |
| `SEX` | Sex | Demographics | numeric | 100.0 | 2 |
| `STAI_TOTAL` | State-Trait Anxiety Inventory item sum | STAI (item sum) | numeric | 66.6 | 72 |
| `YRS_SINCE_DIAGNOSIS` | Years since diagnosis (derived) | Derived (PD Diagnosis History + visit date) | numeric | 91.8 | 740 |
| `YRS_SINCE_SYMPTOM_ONSET` | Years since symptom onset (derived) | Derived (PD Diagnosis History + visit date) | numeric | 90.8 | 831 |

---

## ⏸ Held out — imaging / genetics — `RESOURCE_DEPENDENT_FEATURE` (12)

*Imaging or genotyping. Rarely available in low-resource settings (~94% of surveyed Indian clinicians rarely order these). Use only in the 'with-imaging/genetics' variant.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `APOE` | APOE genotype | IU Genetic Consensus | text | 99.2 | 6 |
| `CAUDATE_REF_CWM` | DaTscan caudate binding ratio | Xing Core Lab Quant SBR | numeric | 12.6 | 141 |
| `DATSCAN_VISINTRP` | DaTscan visual read | Xing Core Lab Visual Read | text | 95.2 | 2 |
| `GBA` | GBA variant status | IU Genetic Consensus | text | 99.1 | 6 |
| `LRRK2` | LRRK2 variant status | IU Genetic Consensus | text | 99.1 | 5 |
| `PARK7` | PARK7 variant status | IU Genetic Consensus | numeric | 99.1 | 1 |
| `PINK1` | PINK1 variant status | IU Genetic Consensus | numeric | 99.1 | 1 |
| `PRKN` | PRKN variant status | IU Genetic Consensus | text | 99.1 | 6 |
| `PUTAMEN_REF_CWM` | DaTscan putamen binding ratio | Xing Core Lab Quant SBR | numeric | 12.6 | 123 |
| `SNCA` | SNCA variant status | IU Genetic Consensus | text | 99.2 | 2 |
| `STRIATUM_REF_CWM` | DaTscan striatum binding ratio | Xing Core Lab Quant SBR | numeric | 12.6 | 121 |
| `VPS35` | VPS35 variant status | IU Genetic Consensus | numeric | 99.1 | 1 |

---

## ⛔ Banned — defines the label — `BANNED_LABEL_DEFINING` (19)

*Participates in computing Y (a Part II or Part III formula item, or a total/stage encoding them). MUST NEVER be used as a model input, regardless of how cheap it is to collect.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `NHY` | Hoehn & Yahr stage — stage 3 is defined by postural instability | MDS-UPDRS Part III | numeric | 83.4 | 7 |
| `NP2FREZ` | Freezing (self-reported) | MDS-UPDRS Part II | numeric | 91.9 | 5 |
| `NP2PTOT` | Part II total — contains the 3 items above | MDS-UPDRS Part II | numeric | 91.8 | 47 |
| `NP2TRMR` | Tremor (self-reported) | MDS-UPDRS Part II | numeric | 92.0 | 5 |
| `NP2WALK` | Walking and balance (self-reported) | MDS-UPDRS Part II | numeric | 91.9 | 5 |
| `NP3FRZGT` | Freezing of gait | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3GAIT` | Gait | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3KTRML` | Kinetic tremor, left hand | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3KTRMR` | Kinetic tremor, right hand | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3PSTBL` | Postural stability | MDS-UPDRS Part III | numeric | 83.2 | 6 |
| `NP3PTRML` | Postural tremor, left hand | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3PTRMR` | Postural tremor, right hand | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3RTALJ` | Rest tremor amplitude, lip/jaw | MDS-UPDRS Part III | numeric | 83.4 | 5 |
| `NP3RTALL` | Rest tremor amplitude, left lower limb | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3RTALU` | Rest tremor amplitude, left upper limb | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3RTARL` | Rest tremor amplitude, right lower limb | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3RTARU` | Rest tremor amplitude, right upper limb | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3RTCON` | Constancy of rest tremor | MDS-UPDRS Part III | numeric | 83.4 | 6 |
| `NP3TOT` | Part III total — contains the 13 items above | MDS-UPDRS Part III | numeric | 81.5 | 91 |

---

## 🎯 Targets (Y and derived) — `TARGET` (8)

*Outcome or label-derived quantity. Never a predictor.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `LABEL` | TD / PIGD / Indeterminate — the target (Y) | Computed (Stebbins formula) | text | 83.0 | 3 |
| `LABEL_FLIPPED_NEXT` | Whether the label changes at the next visit (transition target) | Computed (next-visit shift) | numeric | 71.0 | 2 |
| `MONTHS_TO_NEXT_VISIT` | Interval to next visit | Computed (next-visit shift) | numeric | 93.7 | 9 |
| `NEXT_LABEL` | Label at the following visit (transition target) | Computed (next-visit shift) | text | 76.6 | 3 |
| `NEXT_VISIT_MONTH` | Month of the following visit | Computed (next-visit shift) | numeric | 93.7 | 20 |
| `PIGD_SCORE` | Mean of the 5 gait/balance items (Y component) | Computed (Stebbins formula) | numeric | 83.0 | 45 |
| `TD_PIGD_RATIO` | Tremor / PIGD ratio (Y component) | Computed (Stebbins formula) | numeric | 81.0 | 265 |
| `TREMOR_SCORE` | Mean of the 11 tremor items (Y component) | Computed (Stebbins formula) | numeric | 83.0 | 41 |

---

## 🔑 Keys — `KEY` (4)

*Identifier / join key. Not a predictor.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `EVENT_ID` | PPMI visit code | Join key (all tables) | text | 100.0 | 21 |
| `INFODT` | Visit date | MDS-UPDRS Part III | text | 91.8 | 191 |
| `PATNO` | Patient identifier | Join key (all tables) | integer | 100.0 | 439 |
| `VISIT_MONTH` | Months since baseline (derived) | Derived (EVENT_ID) | integer | 100.0 | 21 |

---

## 🗂 Admin / provenance — `ADMIN` (6)

*Provenance / bookkeeping. Not a predictor.*

| Column | Description | Source table | Type | % present | Unique |
|---|---|---|---|---|---|
| `APPRDX` | PPMI diagnostic sub-code | Subject Cohort History | numeric | 100.0 | 2 |
| `COHORT` | PPMI cohort code | Subject Cohort History | numeric | 100.0 | 1 |
| `GENETIC_COHORT` | Recruited into the genetic cohort (see limitation below) | Derived (Subject Cohort History) | numeric | 100.0 | 2 |
| `PDDXDT` | Diagnosis date (source) | PD Diagnosis History | text | 100.0 | 113 |
| `PDSTATE_USED` | Medication state of the retained Part III record | MDS-UPDRS Part III | text | 72.5 | 2 |
| `SXDT` | Symptom onset date (source) | PD Diagnosis History | text | 98.9 | 131 |

