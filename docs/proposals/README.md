# Proposals — Where the Project Started, and How It Diverged

Two source documents define the project's intent. This page traces each promise in them to
what was actually built, changed, or dropped — so a reviewer can see the reasoning behind
every divergence rather than discovering it.

| File | Date | What it is |
|---|---|---|
| [`01_consolidated_proposal.docx`](01_consolidated_proposal.docx) | before 8 Sep 2026 | Original research proposal: *"Modelling the Evolution of Clinical Certainty in Parkinson's Disease: A Longitudinal Decision Intelligence Framework with Calibrated Confidence and LLM-Augmented Clinical Reasoning."* Slow / moderate / fast progressor labels via K-Means. |
| [`02_renewed_approach.pdf`](02_renewed_approach.pdf) | 8 Sep 2026 | The pivot: *"Longitudinal Confidence Modelling for Motor Subtype Classification in Early Parkinson's Disease."* Drop the invented labels, use the published TD / PIGD / Indeterminate formula; add the transition-risk angle. |

---

## 1. Original proposal

### Core idea — still the core idea

> *"The real question isn't 'what will happen to this patient?' — it's 'how quickly can we
> become confident enough to act?'"*

The proposal's guiding principle — **model how clinical belief evolves across visits, don't
just build a better classifier** — survives intact. What changed is the label, the data rules,
and what we can honestly claim.

### Promise-by-promise

| Proposal said | What happened | Why |
|---|---|---|
| **Label:** K-Means (k = 3) on motor / cognitive / autonomic decline **rates** → slow / moderate / fast | ❌ **replaced** by TD / PIGD / Indeterminate (Stebbins formula) | clusters didn't hold up — HDBSCAN called 39% of patients noise; an invented label needs its own clinical validation (README §3) |
| **Validate labels:** fast progressors should be older, lower DaTscan, more mutations | ❌ not applicable | the formula label is published and needs no new validation |
| **Cohort:** `APPRDX == 1`, drop < 3 yr follow-up → ~400–500 patients (actual: 229) | ✅ **changed** to `COHORT == 1`, no follow-up filter → **439** | `APPRDX == 1` silently dropped 197 genetic-cohort PD patients; the per-visit label needs no trajectory |
| **Cleaning:** replace codes with `NaN`, **median-impute** | ✅ **changed**: no imputation | median fill flattened DaTscan to a constant 0.68; pre-split imputation leaks (README §4.6) |
| **Join:** everything on `PATNO + EVENT_ID` | ✅ **changed** for medications: date-interval join | key join matched nothing — LEDD was 100% missing (README §4.5) |
| **Features:** ~50 columns including motor score, DaTscan, genetics | ✅ **changed**: 27 cheap features; motor exam **banned**; imaging / genetics **held out** | the motor exam *defines* the new label; imaging and genetics are scarcer than the exam in target settings (README §6) |
| **Layer 1:** XGBoost + RF + logistic baselines, 5-fold stratified CV | ✅ built — majority / logistic / boosted trees / XGBoost; **grouped** by patient | ~17 rows per patient make ungrouped CV leak |
| **Hyperparameter tuning with Optuna** | ⏳ not done | fixed, conservative settings for now (`max_depth = 3`) |
| **Layer 2:** MAPIE, APS, 90% coverage | ⏳ designed, not built | see `docs/conformal_design.md` — incl. why n_cal ≈ 88 limits it |
| **Layer 3: "What test should the doctor order?"** — information-value ranking of missing tests (e.g. "DaTscan resolves 45%") | ❌ **dropped**, replaced by **transition risk** | its main candidate tests — DaTscan, genetics — are the inputs we deliberately hold out. Under the triage framing, the recommendation for an uncertain patient is simply *"do the full motor exam"* |
| **Layer 4:** LLM report, validated for faithfulness vs SHAP, medical accuracy, data consistency | ⏳ designed as the **Explanation Orchestrator**, with a stricter validator and a template fallback | see `docs/explanation_orchestrator.md` |
| **Main experiment:** 4 rounds (BL, +6 mo, +12 mo, +24 mo), separate model per round, confidence curve | ⏳ **round structure computed** (438 / 349 / 323 / 275 patients); models not yet run per round | annual-only design (BL → 12 → 24) recommended — keeps 343 vs 275 (README §4.4) |
| **Primary metrics:** singleton rate, stability, calibration (ECE, Brier), time-to-confidence; accuracy only as a sanity check | ⏳ **not computed** — they all need the conformal layer | today only the "sanity check" metrics exist, so they're reported as such |
| **Figures 1–6:** confidence curve, patient heatmap, stability, calibration, time-to-confidence, information value | ⏳ none exist yet (Fig 6 no longer applies) | a fabricated confidence-curve figure was **removed** from the paper draft |
| **Fairness analysis** by sex, age, genetics | ⏳ not done | — |
| **Limitation:** "subtypes come from clustering — subjective choices" | ✅ **resolved** | the label is now a published formula |
| **Limitation:** marginal (not per-patient) coverage | ✅ still true, now quantified — class-conditional coverage for Indeterminate is infeasible at this sample size | `docs/conformal_design.md` |
| **Target journals:** npj Digit Med, JMIR Med Inform, PLOS Digit Health, AI in Medicine, ML4H | — | unchanged |

---

## 2. Renewed approach (8 Sep 2026)

| Renewed approach said | What happened |
|---|---|
| Stop inventing the category; use TD / PIGD / Indeterminate via Jankovic 1990 / Stebbins 2013 | ✅ done — label engine built, **0 mismatches** on independent recomputation |
| "Same dataset, **same 229 patients**" | ✅ better than planned: the cohort fix and dropping the follow-up filter gave **439 patients** and **5,742** labelled visits |
| Same core question: reach a confident classification **earlier** | ⚠️ **narrowed.** PPMI administers the full exam at every visit, so "earlier" only means something outside PPMI. The defensible claim is *"estimate what the exam would say where it isn't done, and say how much to trust it"* (`docs/clinical_need.md`) |
| New angle: **transition risk** — per-patient probability of holding vs flipping | ✅ target built (`LABEL_FLIPPED_NEXT`, 4,918 pairs, 29.1% base rate); ⏳ model not trained |
| **Next step 1:** confirm item-level Part II / III scores exist, not only totals | ✅ confirmed — all 16 items present at item level |
| **Next step 2:** compute the label per patient, per visit | ✅ done |
| **Next step 3:** re-run the pipeline (classification + conformal + stability) | ⚠️ **classification only** so far |
| **Next step 4:** watch for class imbalance | ✅ balanced sample weights, balanced accuracy, per-class reporting |
| "The swap removes the clinical-verification bottleneck" | ✅ true — and it also exposed a **new** requirement the old label didn't have: the **strict leakage policy**, because the new label is a formula over columns in the same table |

---

## What neither document anticipated

These came out of doing the work, not from either plan:

1. **Leakage is the central design problem.** A formula label computed from columns in the same
   table means a naïve model scores 0.906 while learning nothing. That's what produced the
   19-column ban, the feature registry, and the contract test.
2. **Recruitment structure leaks too.** `GENETIC_COHORT` alone scores 0.629 on the binary task
   — signal from PPMI's recruitment design, not from physiology.
3. **First-visit prediction for sporadic patients is near chance** (~0.53). The use case both
   documents open with is the one that isn't solved.
4. **Indeterminate is a boundary band, not a class.** It's 76% unstable visit to visit, which
   shapes both the classifier (binary vs 3-class) and the transition model.
