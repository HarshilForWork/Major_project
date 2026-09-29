# Next Steps — Rutu

From Harshil's review of commit `ce6bb05` (28 Sep) and the data fix in `d6f259c` (30 Sep).
Rutu's commit moved the project a long way: conformal, transition, tracker, orchestrator and
inference are all built. Below is what has to happen before any of those numbers go into the
paper, in order.

---

## 0. Pull, and stop the line-ending noise

```bash
git pull
git config core.autocrlf true      # the 26 "modified" files were CRLF-only, no content change
```

## 1. Re-run everything that needs `xgboost` — the data changed underneath it

A preprocessing bug (Harshil's, in `preprocess.py`) let PPMI's numeric missing codes through:

- Part III **101** ("unable to rate") went into the label formula. 104 visits were mislabelled,
  98 of them as PIGD.
- SCOPA-AUT **9** ("not applicable") inflated `SCOPA_AUT_TOTAL`.

It's fixed: labelled visits went 5,742 → 5,638 and pairs 4,918 → 4,818. **Every result produced
before 30 Sep is stale.** Harshil re-ran the non-XGBoost stages; these need your machine:

```bash
make xgboost ablation conformal transition     # conformal before transition — transition loads its model
make test
```

Then commit the updated `reports/metrics/*` files.

---

## 2. Fix the APS conformal bug  —  `src/trace_pd/models/conformal.py`

**Problem.** `compute_nonconformity_scores` uses the **randomised** APS score
(`p_cum + u · p_rank`), but `predict_sets` uses the **non-randomised** rule (add classes until
`cum_p ≥ q_hat`, keeping the one that crosses). The two don't match, so sets are too large.
That's why APS shows 99% coverage against a 90% target.

**Proof.** On perfectly calibrated synthetic probabilities, the current class covers **97.9%**,
LAC covers 90.7%, and a consistent APS covers **90.8%**.

**Fix.** Compute one score per class with the *same* formula, and include a class iff its score
≤ `q_hat`:

```python
def _aps_scores_all(self, probs, u):
    """Randomised APS score for EVERY class, in the original class order."""
    order = np.argsort(-probs, axis=1)
    sp = np.take_along_axis(probs, order, axis=1)
    s_sorted = np.cumsum(sp, axis=1) - sp + u[:, None] * sp
    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, s_sorted, axis=1)
    return scores

# calibration:  s_i = scores[i, y_i]    with u ~ U(0,1)
# prediction:   include class k iff scores[i, k] <= q_hat   (fresh u); if empty -> argmax
```

Keep LAC as the default in `inference.py` — it's already correct.

## 3. Transition model: out-of-fold upstream features  —  `models/train_transition.py`

**Problem.** `PROB_*`, `PROB_MARGIN_TOP2` and `CONFORMAL_SET_SIZE` come from the **production**
subtype model. That model was trained on 80% of patients and then scores *all* of them, so for
those patients the features are in-sample: the model already saw their true labels. These are
the top transition features, so AUC 0.65 is likely optimistic.

**Fix.** Generate them out-of-fold, using the **same patient folds** as the transition CV:

```python
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X_lab, y_lab, g_lab))
oof = np.full((len(df), 3), np.nan)
for tr, te in folds:
    m = XGBClassifier(**params).fit(X_lab.iloc[tr], y_lab[tr], sample_weight=w[tr])
    oof[lab_idx[te]] = m.predict_proba(X_lab.iloc[te])
# conformal set size: calibrate q_hat inside each training fold, apply to that fold's test rows
# then run the transition CV on the SAME `folds`
```

## 4. Transition model: calibration and the Brier claim

**Problem.** For next-visit flips, Brier **0.227** is *worse* than always predicting the 29%
base rate (**0.206**). Balanced sample weights push probabilities toward 0.5. The report still
says "(well-calibrated probabilities)".

**Fix.**

- Either drop the balanced weights, or wrap the model in `CalibratedClassifierCV(method="isotonic")`
  inside each fold.
- Report the base-rate Brier next to the model's Brier.
- Delete "well-calibrated" unless the model actually beats the base rate.
- Watch `HANDED` showing up as a top-5 feature for the 12-month model. It's almost certainly
  noise; worth a check.

## 5. Train / serve skew in deltas  —  `tracking/tracker.py`, `inference.py`

**Problem.** In training, first-visit and missing deltas are `NaN` (from `shift`). At serving,
`compute_feature_deltas` returns `0.0`. XGBoost treats those differently.

**Fix.** Return `np.nan` when there's no previous visit or no previous value. Also stop
`inference.py` from mapping missing deltas to `0.0` (`deltas.get(c, 0.0)`).

## 6. Explanation orchestrator  —  `explanations/orchestrator.py`

| Problem | Fix |
|---|---|
| **Validator is a no-op.** On failure, `explain()` regenerates the *same* text and marks it valid | on failure use a minimal fallback that states only subtype + set, and return `is_valid: False, fallback_applied: True` |
| **Deltas ranked on raw scale.** LEDD in mg always outranks a 1-point `NP2RISE` change | rank by standardised delta (Δ / cohort SD), or by SHAP |
| **Every increase is "worsened"** — the demo prints `YRS_SINCE_SYMPTOM_ONSET (worsened)` and `N_CONMEDS (worsened)` | per-feature direction map; neutral wording ("increased") for LEDD, med counts and time |
| **"42%, which is above the base rate of 42%"** | compare unrounded values; say "in line with" when within ±1 point |
| Hard-coded base rates 0.42 / 0.29 | read them from the transition artifact |

## 7. Tests should pass on a fresh clone

`test_transition_model_artifacts_exist` and `test_inference` fail on a clone, because
`models/*.pkl` is git-ignored. Either:

- `pytest.skip(...)` when the artifact is missing (like the dataset tests do), or
- add a session fixture that trains a small model when the artifact is absent.

"17/17 passing" in `HANDOFF_README.md` is only true on your machine after training.

## 8. `HANDOFF_README.md`

After the re-run:

- update the numbers
- change "zero hallucination" to what the validator actually guarantees
- note that APS was buggy (fixed) and LAC is the method used
- remove the "17/17" claim until item 7 is done

---

## Smaller items

- `--patient-id` in `inference.py` isn't wired; it always runs the demo.
- The demo patient (3001) is in the training data. Hold one patient out for the demo.
- `preprocess_runlog.txt` embeds the local Windows path of whoever ran it. Harmless, but it
  makes every run a diff.

---

## Team decisions needed (not only Rutu)

1. **Medication state of the exam.** 25.6% of same-day OFF / ON exams give a different label,
   and 1,730 labelled visits use an ON exam (README §9.7). Options: label OFF-state exams
   only, or stratify / add exam state as a covariate.
2. **Explainability contribution.** The lead is "flip anatomy"
   (`docs/novelty_feasibility_results.md`). It needs a novelty search before we commit.
3. **Project name.** A June 2026 paper is already called TRACE (arXiv 2606.30313).
