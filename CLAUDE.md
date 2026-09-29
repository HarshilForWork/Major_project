# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TRACE-PD predicts a Parkinson's patient's motor subtype (TD / PIGD / Indeterminate) from
**routine, low-cost clinical intake only**, on the PPMI cohort. Four layers sit on top of the
classifier: conformal confidence sets, a longitudinal patient tracker, a 12-month
transition-risk model, and FCX (Formula-Coordinate Explanations).

`README.md` is the authoritative project document (data, label, decision log, results,
limitations). `HANDOFF_README.md` is an earlier handover whose numbers are **stale** — a
30 Sep 2026 preprocessing fix changed labelled visits 5,742 → 5,638 and pairs 4,918 → 4,818.
When quoting numbers, use `README.md` §9 and `reports/metrics/*`, never `HANDOFF_README.md`.

## Running things

`trace_pd` is a `src/` layout package and is **not installed** in the default environment, so
`python -m trace_pd.…`, `make test`, `make conformal`, `make transition` and `make demo` fail
with `ModuleNotFoundError`. Either install it once:

```bash
pip install -e .
```

or prefix module invocations with `PYTHONPATH=src` (bash) / `$env:PYTHONPATH="src"`
(PowerShell). The `make preprocess/baseline/xgboost/ablation/sporadic/fcx/c3/figures/docs`
targets run scripts by path and work without installing.

```bash
make preprocess   # raw PPMI -> data/processed/ppmi_tdpigd_long.csv (deterministic, no seeds)
make baseline     # Exp 1-4: majority / logistic / HistGradientBoosting
make xgboost      # Exp 1-3 with XGBoost
make ablation     # Exp 5: feature-group ablation
make sporadic     # Exp 6: sporadic PD, first visit only -- the hardest honest test
make fcx          # FCX evaluation (C1/C2/C3) + fig_fcx_summary
make c3           # C3 validation experiments (E1 cheap / E2 exam-informed / E3 sustained)
make conformal    # trains + pickles models/conformal_model_production.pkl
make transition   # trains models/transition_model_production.pkl (run AFTER conformal)
make figures docs # reports/figures/*, docs/data_dictionary.md
make test         # pytest
make all          # everything, in order
```

Single test: `PYTHONPATH=src python -m pytest tests/test_fcx.py::test_name -v`.

`models/` is git-ignored, so on a fresh clone `test_inference.py` and `test_transition.py`
fail until `make conformal transition` has been run. 29/31 tests pass without them.

Scripts that train a black box accept `--backend xgb|hgb` (`src/trace_pd/models/backend.py`).
The backend picks the output suffix: `reports/metrics/fcx_results_xgb.txt` vs
`fcx_results.txt`, `c3_validation_xgb.txt` vs `c3_validation.txt`. Both variants are committed
on purpose — don't "clean up" the unsuffixed ones.

## Non-negotiable invariants

These are enforced in code and tested; violating one invalidates every downstream number.

1. **Strict leakage policy.** No column that participates in computing Y may be a feature.
   That is the 16 Stebbins formula items **plus** `NP3TOT`, `NP2PTOT`, `NHY` (indirect
   encoders), plus `GENETIC_COHORT` (a recruitment-arm flag that alone scored 0.629).
   Training code selects X **by bucket** from `data/processed/ppmi_tdpigd_dictionary.csv`
   (`CHEAP_FEATURE` = the 26 features) — never by a hand-written column list.
   `tests/test_dataset_contract.py` fails if a label-defining column lands in a feature bucket.
2. **Patient-grouped CV.** ~17 visits per patient, so any row-wise split leaks. Use
   `StratifiedGroupKFold(n_splits=5, random_state=42)` grouped by `PATNO`.
3. **Preprocessing is deterministic** — no seeds, byte-identical output from the same extract.
4. **No imputation outside a fold.** `NaN` is left as-is (XGBoost/HistGBM handle it natively);
   logistic regression imputes/scales inside the fold pipeline.
5. **Balanced sample weights for classification metrics, unbalanced when probabilities are
   reported to a user** — `fit_classifier(..., balanced=False)` (balanced weights push
   probabilities toward 0.5).
6. **PPMI missing codes are not scores**: Part III `101` ("unable to rate") and SCOPA-AUT `9`
   ("not applicable") must become `NaN`. Test-guarded.
7. **Balanced accuracy + macro-F1** are the headline metrics; plain accuracy is meaningless
   here (majority class already scores 0.607).

`configs/config.yaml` documents the run that produced the committed artifacts — **editing it
does not drive the code**; constants live in the scripts.

## Architecture

All paths resolve through `src/trace_pd/config.py` (`ROOT` from `__file__`), so scripts run
correctly from any working directory. Never hardcode a path.

Pipeline: `data/preprocess.py` joins 19 PPMI CSVs (visit-level on `PATNO+EVENT_ID`, static on
`PATNO`, medication logs on **date interval** — a key join matched nothing) into
`ppmi_tdpigd_long.csv`, one row per patient-visit, plus the dictionary that governs every
column's bucket. Long format is deliberate: the label is per-visit and the flip target needs
consecutive visits; wide per-round tables are built downstream.

Three black boxes, explained by three FCX components:

| Model | Trained in | FCX component |
|---|---|---|
| g1 subtype classifier (TD vs PIGD) | `models/train_xgboost.py`, `models/backend.py` | C1 `explain/implied_exam.py` |
| g2 split-conformal LAC/APS sets on g1 | `models/conformal.py` | C2 `explain/straddle.py` |
| g3 transition risk (label flips next visit) | `models/train_transition.py` | C3 `explain/pdn.py` |

FCX explains all three **in the Stebbins formula's own coordinates** — the tremor/gait
log-ratio `l = log(T+eps) - log(P+eps)` and its two cutoffs (`explain/formula.py`, `EPS = 0.5/16`,
chosen so the zone rule reproduces the stored labels). Shapley is an **own implementation**
(`explain/shapley.py`), no SHAP dependency; SHAP-style attribution appears only as a baseline.
`explain/trajectory.py` shifts over the **full scheduled-visit sequence** before dropping
unlabelled visits, so "previous"/"next" never silently span a gap.

Serving path: `inference.py` loads the two pickles and composes classifier → conformal set →
`tracking/tracker.py` (history store, deltas, stability index) →
`explanations/orchestrator.py` (structured payload + fact-validated narrative).
Training and inference never call each other; they meet at the data layer.

## Known-broken, don't mistake for working

Tracked in `docs/NEXT_STEPS_RUTU.md` and `README.md` §13:

- **Conformal APS** calibrates with the randomised score but predicts with the non-randomised
  rule → sets too large (99% coverage vs 90% target). LAC is correct and is the default.
- **Transition model** upstream features (`PROB_*`, `CONFORMAL_SET_SIZE`) come from the
  production subtype model and are in-sample for 80% of patients; AUC 0.65 is optimistic.
  Probabilities are uncalibrated.
- **Orchestrator validator is a no-op**; delta sorting is on raw scales.
- **Tracker** first-visit deltas are `0.0` at serving vs `NaN` in training.
- `inference.py --patient-id` is not wired; only `--demo` works.
- Clinically: sporadic first-visit prediction is near chance (~0.53) and three-class is ~0.46.
  Only the binary TD vs PIGD task (~0.69) is usable today. Don't write claims stronger than that.

## Data and repo hygiene

`data/` holds a **DUA-restricted PPMI extract committed deliberately** — this repository must
stay **private** and must not be redistributed. Do not copy raw rows, patient IDs, or extract
contents into docs, issues, artifacts, or anything leaving the repo.

Line endings: set `git config core.autocrlf true` — a past commit showed 26 files "modified"
with CRLF-only diffs.
