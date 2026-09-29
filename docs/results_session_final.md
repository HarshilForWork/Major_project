# Session results — 30 Sep 2026

Black box: XGBoost unless a table says otherwise. Every model choice was made by 5-fold
`StratifiedGroupKFold` grouped by `PATNO` (seed 42) on the **development 80%** of patients;
the **locked 20%** (88 patients, seed 2026, stratified on baseline label,
`data/interim/locked_test_patients.csv`) was scored once per step. Confidence intervals are
200× **patient-level** bootstrap percentiles — patients are resampled, not rows, because ~17
visits of one patient are not independent. All 321 runs, including the losers, are in
`reports/metrics/experiment_log.csv` (append-only, so a script re-run appends a second
identical block rather than overwriting — repeated rows are repeated runs, not duplicates).

**One-line summary.** Two leaks were removed and most of the transition model's apparent
signal went with them (AUROC 0.66 → 0.56). Tuning the subtype classifier bought nothing.
Giving the transition model the patient's history and the exam produced a genuinely good
model (AUROC 0.80, locked test 0.803) — and that better model made the C3 explanation
*worse*, not better.

---

## 1. Bugs fixed, and what the numbers were before → after

| # | Bug | Where | Before → After | Status |
|---|---|---|---|---|
| 1 | **APS calibrated with the randomised score but predicted with the non-randomised accumulation rule**, so sets were strictly larger than calibration implied | `models/conformal.py` | binary coverage **99.04% → 89.36%** (target 90%), set size 1.92 → 1.60; 3-class **98.57% → 90.23%**, set size 2.87 → 2.49 | ✅ fixed, regression-tested |
| 2 | **`PROB_*` / `CONFORMAL_SET_SIZE` were in-sample** — taken from a production model fitted on 80% of patients and then used to score all of them | `models/train_transition.py` | next-visit AUROC **0.6469 → 0.5350**; 12-month AUROC **0.6556 → 0.5630** | ✅ fixed (out-of-fold, shared patient folds) |
| 3 | **Brier computed from a balanced-weight model** and captioned "well-calibrated"; balanced weights push probabilities toward 0.5 | `models/train_transition.py` | next-visit Brier 0.2271 (weighted) → **0.2084 unweighted vs 0.2047 base rate — no better than the base rate** | ✅ fixed |
| 4 | **OUTCOME partition used a signed comparison** `phi_N+phi_P > phi_D` while the shares beside it use magnitudes. `phi_D` is pinned near zero (sd 0.012), `phi_P` reaches −2.22, so the rule really tested "is phi_P negative" | `evaluation/evaluate_fcx.py` | claimed **863 of 1203 flips "drift-dominated"** in a model where drift is 3% of risk; on the magnitude rule the split is **1200 / 3**, so the contrast (37% vs 36%) is **not computable** and the line now says so | ✅ fixed |
| 5 | **Synthetic planted-mechanism test is tautological** — plants `g = sigmoid(−0.9 + 1.5·z(s_c))` from the surrogate's *own* component, so `logit(g)` is exactly affine in `s_c` and NNLS recovers `w = e_c·1.5/std(s_c)` exactly (verified to 8 d.p.) | `evaluation/validate_c3.py` | "**3/3 recovered**" was never evidence — it passes for any data, model or patient | ✅ flagged at source, withdrawn in `docs/fcx.md`, falsifiable replacement added |
| 6 | Stale hardcoded counts printed as if computed | `conformal.py`, `viz/architecture_hld.py` | "5,122 / 5,742 rows" → computed; figure "4,918 pairs" → **4,818** | ✅ fixed |

### Not fixed (deliberately) — non-critical, none changes a reported metric

| Issue | Where | Why left |
|---|---|---|
| `FactValidator` never rejects anything | `explanations/orchestrator.py` | Affects narrative text, not a metric. Pre-existing, tracked in `NEXT_STEPS_RUTU.md` §4. |
| First-visit deltas are `0.0` at serving vs `NaN` in training | `tracking/tracker.py` | Train/serve skew in the demo path only. |
| `--patient-id` not wired | `inference.py` | Feature gap, not a bug. |
| Docstring cites `GENETIC_COHORT` importance "#2 (0.095)" | `evaluation/ablation.py` | Prose only — the column is bucketed `ADMIN` and never enters the importance table. Real values 0.1576 / 0.0991. |
| `c != "GENETIC_COHORT"` filter is a no-op | `evaluation/sporadic_baseline.py` | The column was never in `CHEAP_FEATURE`, so the printed "removed" is misleading but the feature count (25) and every number are correct. |
| `ell`/label use raw ratio vs `(T+ε)/(P+ε)` | `data/preprocess.py` vs `explain/formula.py` | Verified inert: `classify()` reproduces `formula.zone()` on **100.00% of 5,638 labelled rows**, 0 mismatches. |

*Audit coverage: `preprocess.py`, `train_baseline.py`, `train_xgboost.py`, `backend.py`,
`ablation.py`, `sporadic_baseline.py`, `evaluate_fcx.py`, `validate_c3.py` and all six
`explain/` modules were read line-by-line against the raw PPMI CSVs. No leakage was found in
any of them: every split is grouped by `PATNO`, upstream probabilities are out-of-fold, and
no imputer or scaler is fitted outside a fold.*

---

## 2. Subtype classifier — every config tried

Binary TD vs PIGD, per visit. Development set 4,002 visits / 351 patients.

| Config | CV bal. acc | sd | CV AUROC | Inside winner ±1 sd? |
|---|---|---|---|---|
| **logistic (L2)** | **0.6932** | 0.0310 | 0.7523 | — winner |
| xgb d4 lr0.03 | 0.6876 | 0.0345 | 0.7466 | ✅ tie |
| xgb default (d3 lr0.05) | 0.6858 | 0.0308 | 0.7440 | ✅ tie |
| xgb d2 lr0.03 | 0.6844 | 0.0404 | 0.7494 | ✅ tie |
| xgb + missingness indicators | 0.6842 | 0.0335 | 0.7449 | ✅ tie |
| xgb d3 lr0.03 | 0.6829 | 0.0329 | 0.7491 | ✅ tie |
| xgb tuned threshold (inner CV) | 0.6815 | 0.0353 | 0.7440 | ✅ tie |
| xgb d2 lr0.08 | 0.6782 | 0.0307 | 0.7409 | ✅ tie |
| formula-aware (regress log T, log P → Stebbins cutoffs) | 0.6765 | 0.0321 | 0.7469 | ✅ tie |
| xgb d3 lr0.08 | 0.6761 | 0.0264 | 0.7348 | ✅ tie |
| xgb d4 lr0.08 | 0.6711 | 0.0363 | 0.7299 | ✅ tie |
| lightgbm default | 0.6693 | 0.0306 | 0.7331 | ✅ tie |

**Winner evaluated once on the locked test** (1,018 visits / 88 patients):

| Metric | Value | 95% CI (200× patient bootstrap) | Significant? |
|---|---|---|---|
| Balanced accuracy | **0.6934** | [0.6472, 0.7384] | **vs chance (0.50): YES** |
| AUROC | **0.7528** | [0.6983, 0.8035] | vs chance: YES |
| Majority baseline | 0.5000 | — | — |

**What did not hold.** Every one of the 11 losing configs sits inside the winner's ±1 sd
band, and the full spread across all 12 (0.0239) is *smaller* than the winner's own per-fold
sd (0.0310). "Logistic beats XGBoost" is **not a result** — the 12 configs are statistically
indistinguishable. Hyper-parameter tuning, missingness indicators, threshold tuning and the
formula-aware two-head model all bought **nothing**. The honest reading is that this task
saturates around 0.69 on cheap features regardless of model, which matches the project's
standing 0.696 / 0.690 figures. The locked-test CI spans 0.09 because it rests on 88 patients.

---

## 3. C1 per round

One row per patient; 26 cheap features per included visit (`_m<month>`) plus deltas between
consecutive included visits; target TD vs PIGD **at month 24**. Black box = logistic (Step 2
winner). C1 heads trained on the **true** log T / log P at month 24.

| Round | Visits | n | CV bal. acc | sd | CV AUROC | LAC singleton | Locked bal. acc | 95% CI |
|---|---|---|---|---|---|---|---|---|
| **R1** | BL | 328 | **0.6830** | 0.068 | 0.7299 | 35.0% | 0.6637 | [0.547, 0.767] |
| R2 | BL+6 | 267 | 0.6080 | 0.055 | 0.6651 | 35.5% | 0.6939 | [0.578, 0.805] |
| R3 | BL+6+12 | 248 | 0.6308 | 0.112 | 0.6098 | 41.4% | 0.6341 | [0.498, 0.759] |
| R4 | BL+6+12+24 | 248 | 0.6145 | 0.059 | 0.6325 | 28.8% | 0.6443 | [0.510, 0.765] |
| A3 | BL+12+24 | 307 | 0.6277 | 0.060 | 0.7057 | 29.8% | 0.5742 | [0.449, 0.686] |

| Round | C1 fidelity | align T | align P | gait share | latest visit | earlier visits | Top feature |
|---|---|---|---|---|---|---|---|
| R1 | 0.765 | 0.252 | 0.352 | 48.5% | 100.0% | 0.0% | `LEDD_TOTAL_MG_m0` |
| R2 | 0.771 | 0.267 | 0.386 | 48.7% | 54.8% | 45.2% | `LEDD_TOTAL_MG_m6` |
| R3 | 0.696 | 0.197 | 0.387 | 47.9% | 47.1% | 52.9% | `LEDD_TOTAL_MG_m6` |
| R4 | 0.738 | 0.231 | 0.463 | 47.6% | 35.3% | 64.7% | `LEDD_TOTAL_MG_m6` |
| A3 | 0.715 | 0.275 | 0.480 | 48.9% | 43.6% | 56.4% | `LEDD_TOTAL_MG_m12` |

**What did not hold.**

- **The confidence curve does not exist.** More visits do not help: the best round is **R1,
  the one with the fewest visits**, and R4 is the *least* decisive round (28.8% singletons vs
  35.0% at baseline). Each added visit multiplies features 26 → 182 while the cohort shrinks
  328 → 248. The premise that accumulating visits sharpens the prediction is **not supported**.
- **The rounds do not share a cohort** (only 248 patients are common to all five), so
  round-to-round differences are partly survivorship, not information.
- **R4 and A3 are not forecasts.** They include the month-24 intake, i.e. features from the
  same visit as the label. Reading R1→R4 as one curve mixes two different tasks.
- **Locked-test CIs overlap completely** (50–68 patients each); no round is separated from any
  other.
- **Tremor is the weak channel everywhere** — alignment 0.20–0.28 vs gait 0.35–0.48 — and
  `LEDD_TOTAL_MG` is the top attributed feature in **all five rounds**. The model keeps reading
  medication dose as tremor information, reproducing the known C1 finding at every round.
- Earlier visits do receive attribution (latest-visit share falls 100% → 35%), so the extra
  visits are *used*; they just don't improve accuracy.

---

## 4. C3 — old (E1–E4) vs new (E5–E6)

| Exp | Setting | AUROC | Brier | base-rate Brier | C3 R² | prox / drift / noise | C3 noise-vs-drift | Shapley |
|---|---|---|---|---|---|---|---|---|
| E1 | cheap features, any flip (deployment) | 0.531 | — | — | 0.163 | — | 0.520 | 0.492 |
| E2 | exam-informed, any flip | 0.771 | — | — | 0.605 | 83 / 3 / 13 | 0.491 | 0.513 |
| E3 | exam-informed, sustained flip | 0.796 | — | — | 0.521 | 84 / 6 / 10 | 0.620 | 0.666 |
| E4 | planted mechanisms | see below | | | | | | |
| **E2ref** | exam-informed, no history (same box as E5) | 0.775 | 0.1632 | 0.2047 | **0.609** | 80 / 3 / 17 | 0.491 | 0.512 |
| **E5** | **exam-informed + history, any flip** | **0.802** | **0.1550** | 0.2047 | 0.509 | 80 / 20 / **0** | **0.378** | 0.486 |
| **E6** | **exam-informed + history, flip ≤ 12 months** | **0.795** | **0.1802** | 0.2433 | 0.419 | 80 / 20 / **0** | **0.358** | 0.393 |

E5 locked test: **AUROC 0.803, 95% CI [0.771, 0.833]**, Brier 0.1447.
E6 locked test: AUROC 0.764, 95% CI [0.714, 0.809], Brier 0.1897.
Both chose `max_depth=4, learning_rate=0.03, 300 trees`, trained unweighted.

Planted mechanisms (E5 feature set): trained proximity-only **PASS** (82% proximity share);
trained drift-only **FAIL** (drift share 33%); trained noise-only **INCONCLUSIVE** (that box
has no signal, AUROC 0.50). Falsifiable synthetic version **3/3**, with head correlations
0.81 / 0.32 / 0.45.

**What did not hold.**

- **The headline negative: a better black box produced a worse explanation.** Adding history
  raised AUROC 0.775 → 0.802 but collapsed the noise weight to **b_N = +0.00** (it was +0.62
  without history). With no noise channel left, C3's noise-vs-drift AUROC falls to **0.378 and
  0.358 — below chance** — and stays below grouped Shapley. C3 fidelity R² also *drops*
  0.609 → 0.509. Improving the model being explained degraded the decomposition.
- **C3 still never beats grouped Shapley** at the one thing it was designed for. That was true
  at E3 (0.62 vs 0.67) and is worse now. C3 buys interpretable *structure*, not accuracy.
- **Brier gains are real but the calibration story is narrow**: +24% and +26% skill over the
  base rate for E5/E6, against the Step-1 cheap-feature model which is **no better than the
  base rate** at all.
- **E5/E6 are not the deployment setting.** They use the true exam at visit *t*, so they answer
  "will today's subtype hold?", not "what is the subtype when no exam was done". The
  deployment-setting model (E1) remains at 0.531 — essentially chance.
- **Planted-mechanism evidence is thinner than it looked**: 1/3 trained (the other two boxes
  have no signal to test), and the previously-cited synthetic 3/3 was tautological (§1, bug 5).
- History features are `NaN` at a patient's first visit, so early visits carry less information
  than the row count suggests.

---

## 5. What a reader should take away

1. **Two leaks were removed.** The transition model's headline AUROC was mostly leak; on cheap
   features it is at chance (0.535) and its probabilities are no better than the base rate.
2. **The subtype classifier is saturated at ~0.69** on cheap features. Twelve configs, no
   significant difference. Further tuning is not where the gains are.
3. **More visits did not help** subtype prediction at month 24 — baseline alone was the best
   round, against the project's expectation of a rising confidence curve.
4. **Transition risk becomes genuinely predictable (0.80) once the exam is available**, which
   is a real result but for a different clinical question than the one TRACE-PD is aimed at.
5. **FCX C3's correctness claim is weaker than previously reported**: it has never beaten
   grouped Shapley, one of its two planted-mechanism validations was tautological, and it
   degrades on a stronger black box.

Artifacts: `reports/metrics/experiment_log.csv` (311 runs) ·
`subtype_model_comparison_xgb.txt` · `c1_rounds_xgb.txt` · `c3_improved_xgb.txt` ·
`conformal_results.txt` · `transition_results.txt` · `fcx_results_xgb.txt`.
