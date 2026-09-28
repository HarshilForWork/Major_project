# TRACE-PD

**Temporal Reasoning and Confidence Explanation for Parkinson's Disease**

Motor-subtype classification (TD / PIGD / Indeterminate) on the PPMI cohort, with
calibrated confidence and visit-to-visit stability tracking.

---

## What this does

Parkinson's patients fall into two clinically distinct motor subtypes — tremor-dominant
(**TD**) and postural instability / gait difficulty (**PIGD**). PIGD patients decline
faster, fall more, and respond less well to levodopa and DBS. Knowing the subtype early
changes how a patient is monitored.

The subtype is defined by the published **Jankovic (1990) / Stebbins et al. (2013)** ratio
over 16 MDS-UPDRS items. That formula is deterministic — if the full motor exam is
completed, no model is needed. This project targets the case where it is **not**
completed, predicting the subtype from the routine clinical data that *is* collected,
and saying how much that prediction can be trusted.

The full motor battery is banned from the feature set by contract, so the model cannot
re-derive the label arithmetically.

---

## Results (honest summary)

Balanced accuracy, `StratifiedGroupKFold` grouped by patient. 3-class chance = 0.333,
binary chance = 0.500.

| Experiment | Majority | Logistic | XGBoost |
|---|---|---|---|
| Round 1 — baseline visit only | 0.333 | 0.442 | 0.427 |
| All scheduled visits pooled | 0.333 | 0.487 | 0.460 |
| **TD vs PIGD** (Indeterminate dropped) | 0.500 | **0.707** | **0.701** |

**Read this carefully.** The headline 0.70 is the *binary* task with Indeterminate
removed. The three-class problem sits near 0.46 — barely above chance — and baseline-visit
prediction for sporadic patients alone scores **0.518**, which is chance. Indeterminate is
a narrow boundary band (0.90 < ratio < 1.15), not a stable clinical entity, and the model
cannot separate it.

Ablation on the binary task (`reports/metrics/ablation_results.txt`):

| Configuration | Features | Balanced acc |
|---|---|---|
| Full | 27 | 0.701 |
| Drop `GENETIC_COHORT` | 26 | 0.685 |
| Drop axial items (`NP2RISE`, `NP2TURN`) | 25 | 0.691 |
| **Only `GENETIC_COHORT`** (sanity floor) | **1** | **0.629** |

That last row is the important one. A single PPMI *recruitment-arm flag* — not a clinical
measurement — reaches 0.629 on its own. Much of the apparent performance is cohort
structure, not physiology. `GENETIC_COHORT` should be dropped from X before any claim is
made, and the axial items measure the same construct as the PIGD score, so they are
plausible proxy leakage and are disclosed rather than defended.

---

## Repository layout

```
trace-pd/
├── configs/
│   └── config.yaml                 cohort, label cutoffs, CV and leakage policy
├── data/                           git-ignored — PPMI DUA, see data/README.md
│   ├── raw/ppmi_csv/               23 source CSVs from LONI IDA
│   ├── raw/zips_as_downloaded/     original archives, unmodified
│   └── processed/                  ppmi_tdpigd_long.csv (6,922 × 75) + dictionary
├── docs/
│   ├── preprocessing_methods.md    every preprocessing decision, with rationale
│   ├── architecture.md             HLD walkthrough of the four layers
│   └── clinical_need_evidence_case.pdf
├── models/                         trained artifacts (git-ignored)
├── notebooks/                      exploratory work
├── reports/
│   ├── figures/                    final figures only
│   ├── metrics/                    run logs and result tables
│   ├── paper/                      TRACE-PD_paper_draft_v7.docx
│   └── slides/                     TRACE-PD_first_review.pptx
├── src/trace_pd/
│   ├── config.py                   single source of truth for all paths
│   ├── data/preprocess.py          raw PPMI → labelled long table
│   ├── models/train_baseline.py    majority / logistic / HistGradientBoosting
│   ├── models/train_xgboost.py     XGBoost baseline
│   ├── evaluation/ablation.py      feature-group ablation study
│   └── viz/                        every figure, as reproducible code
├── tests/
│   └── test_dataset_contract.py    8 invariants incl. the leakage guarantee
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# place the PPMI extract in data/raw/ppmi_csv/ first — see data/README.md
make preprocess     # → data/processed/ppmi_tdpigd_long.csv
make baseline       # → reports/metrics/baseline_model_results.txt
make xgboost        # → reports/metrics/baseline_model_results_xgb.txt
make ablation       # → reports/metrics/ablation_results.txt
make figures        # → reports/figures/
make test           # dataset contract tests
```

`make all` runs the whole pipeline in order.

---

## Dataset

| | |
|---|---|
| Source | PPMI via LONI IDA, extract 16 Aug 2026, 19 source tables |
| Cohort | `COHORT == 1` → **439 patients** (242 sporadic, 197 genetic-arm) |
| Shape | **6,922 × 75**, one row per patient-visit, ~17 visits per patient |
| Labelled rows | 5,742 — a label is assigned only when all 16 items are present |
| Class balance | TD 52.6% · PIGD 36.6% · Indeterminate 10.8% |
| Transition pairs | 4,918 consecutive visit pairs, flip rate 29.1% |
| Label instability | 355 of 439 patients (80.9%) change label at least once |

`ppmi_tdpigd_dictionary.csv` assigns every column a role programmatically —
`CHEAP_FEATURE`, `RESOURCE_DEPENDENT_FEATURE`, `BANNED_LABEL_DEFINING`, `TARGET`,
`KEY`, `ADMIN`. Nothing is hand-picked.

---

## Leakage control

The label is a deterministic function of 16 MDS-UPDRS items. A model allowed to see them
reaches 0.906 while measuring nothing. The strict policy is therefore:

> **No column that participates in computing Y may be used for training.**

This is enforced three ways: the feature registry assigns roles by rule, the preprocessing
stage asserts at runtime that all 16 banned items are excluded, and
`tests/test_dataset_contract.py::test_no_label_defining_item_is_a_feature` fails the build
if the contract is ever violated.

Two validation rules follow from the data shape and are non-negotiable:

- **Grouped CV by `PATNO`.** Each patient contributes ~17 rows; a row-wise split puts the
  same patient in train and test.
- **Per-class reporting.** With 10.8% Indeterminate, aggregate accuracy hides the failure.

---

## Status

| Layer | State |
|---|---|
| 1 — Preprocessing + label engine | **built**, validated, 0 label mismatches |
| 2 — Subtype classifier | **built**, evaluated, results above |
| 3 — Conformal confidence | designed, not built |
| 4 — Longitudinal tracker + transition risk | target column built; model not trained |
| 5 — Explanation orchestrator | designed, not built |

`docs/architecture.md` and `reports/figures/fig_HLD_system_architecture.png` describe the
full intended system; only layers 1 and 2 exist today.

---

## Known limitations

- Three-class performance is near chance; only the binary TD vs PIGD task is usable.
- `GENETIC_COHORT` alone scores 0.629 — recruitment-design signal, not clinical signal.
- Axial items (`NP2RISE`, `NP2TURN`) measure the same construct as the PIGD score and are
  plausible proxy leakage.
- Conformal prediction assumes exchangeability, which ~17 correlated visits per patient
  violate; calibration must be one row per patient, or use a cluster-conformal variant.
- With 439 patients, class-conditional conformal calibration for Indeterminate (~9
  calibration patients) is not feasible at α = 0.10.
- No external validation cohort. Every number here is internal to PPMI.

---

## Data use

PPMI data are governed by a Data Use Agreement and are **not redistributable**. `data/` is
git-ignored and this repository contains no patient-level data. Obtain access at
<https://ida.loni.usc.edu>.

---

## Authors

Aishwarya Deshmukh · Rutu Mehta · Harshil Bhanushali
Guide: Dr. Kriti Srivastava
Department of Computer Science and Engineering (Data Science),
Dwarkadas J. Sanghvi College of Engineering
