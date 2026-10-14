# Decision checklist (brief Section 6)

| # | Condition | Result |
|---|---|---|
| 1 | G1-G4 pass on PPMI, M_minus (and M_full) | **Yes**, all five models (with the two documented rule changes). The extra Stage 9.4 identity also passes (max 5.1e-8). |
| 2 | Size of the problem clearly above seed noise | **ADNI: yes** in every variant (first->last +5.8 to +11.4 points; all pairs +6.0 to +9.0; all CIs exclude 0). **PPMI: borderline** (first->last +10.4 [-0.0, 20.5]; all pairs +5.3 [2.5, 7.9]; labelled-only not significant). |
| 3 | TRACE more stable than chaining (paired CI excluding 0) | **Deletion: yes** for intermediate deletions in every model (TRACE moves 3-4x less); when the first or last visit is deleted, chaining does not move and TRACE moves slightly. **Retraining: mixed.** Better than chaining on PPMI (+0.041 [0.037, 0.045]); very slightly worse on ADNI (-0.007 [-0.012, -0.002] M_minus; -0.010 / -0.011 for the variants). |
| 4 | TRACE not clearly worse than the best baseline on faithfulness | **Yes, with one borderline case.** TRACE's paired CI is entirely below zero only against direct Baseline Shapley at k = 1 (-0.025 PPMI, -0.040 ADNI). Direct is the upper reference by construction. Against population / path-dependent on PPMI at k = 3 the CI just touches zero (-0.022 [-0.046, -0.000]). TRACE beats chaining and AR(1)-SHAP. |
| 5 | Honest answer to the fixed first-visit question | **Yes.** It is unaffected by intermediate or last deletions, but moves 0.15 (PPMI) to 0.36 (ADNI) log-odds when the first visit is deleted, vs TRACE's 0.013-0.045. Faithfulness: TRACE ties it on ADNI and beats it on PPMI at k = 5 (+0.034 [0.018, 0.050]). |
| 6 | Sanity check: explanations collapse under shuffled labels | **Yes at visit level:** shuffled models have BA 0.53 / 0.52, and per-visit agreement of tallies is about 0 (-0.02 PPMI, -0.11 ADNI). **Partly at feature level:** per-feature mean abs(u) keeps Spearman 0.30 (PPMI) / 0.46 (ADNI), as features that change often can receive credit even without signal. ADNI's total credit shrinks 6x. |

# Deviations from the brief

1. **Code location.** `src/trace_pd/explain/trace_engine.py` and `src/trace_pd/evaluation/trace_*.py`, not `src/trace/engine.py`: a package named `trace` would shadow the Python standard-library module. Config: `configs/trace.yaml`.
2. **shap compatibility.**
   - shap 0.49.1 (newest release) cannot read XGBoost 3.1.2 models. It is fixed on shap's GitHub but not yet released.
   - shap therefore explains a verified *copy* of the model whose serialised base score is a plain number. The copy is checked identical apart from that text. Model-file hashes are unchanged.
3. **shap background subsampling, found and fixed.**
   - shap silently subsamples a plain-array background to 100 rows.
   - The population baseline first ran with 100 instead of 500 background visits. This was detected when the Stage 9.4 identity failed (error 0.065 PPMI, 0.55 ADNI).
   - Fixed by passing the background as a masker with all 500 rows; the identity now passes at 5e-8.
   - Every population-based number (Stages 5, 6, 7 retraining, 9; never-changing credit) was recomputed. TRACE itself was never affected: it uses one reference row, or the patient's own <= 27 visits.
4. **G1 is measured against the exact 64-bit tree sum** (user decision). XGBoost's 32-bit predict exceeds 1e-5 on adni_full. Both numbers are reported.
5. **G4 for PPMI uses k <= 13** (user decision; only 1,488 pairs exist at k <= 12).
6. **Added check: pilot-code cross-check column** (user request).
7. **Fast exact algorithm for retrained and shuffled models.** The pilot's leaf-path algorithm is used where per-reference shap calls would take hours (seed and shuffled models). It is verified equal to the shap route on every pair of every patient (<= 4.3e-8). All main TRACE results use shap interventional TreeSHAP.
8. **ADNI model roles.** M_minus is primary; M_minus_LM and M_minus2 are variants; M_full is a sensitivity run (user decision). Stages 4 and 7 were also run on the variants.
9. **Labelled-visits-only sensitivity runs added** (Stage 4). The primary analysis uses all visits.
10. **Baselines not run.** Asymmetric Shapley is skipped (no implementation in the repo; user decision). FA-SHAP is skipped (no forecasts). DynSHAP and C-SHAP are out of scope.
11. **AR(1)-SHAP scope.**
    - Primary models only.
    - Excluded from retraining stability (cost).
    - Undefined at a patient's first visit, so those pairs are excluded for AR(1) only, with paired comparisons on matching pairs.
12. **Stage 4 definitions** reuse the pilot's exactly.
13. **Faithfulness eps** was still open: 0.05 was used with sensitivity at 0.01 / 0.1 / 0.25. Conclusions are the same at every eps.
14. **Feature groups** are the DRAFT (`configs/feature_groups.yaml`), not yet confirmed. Per-feature numbers are also reported.
15. **Stage 8 shuffling design.** Each development patient's baseline label is permuted across patients and applied to all of that patient's training visits.
16. **Stage 9.2 flips** are reported for ADNI too, for completeness. ADNI's binary task has almost no consecutive CN <-> dementia label flips (1), because MCI visits are excluded from labels.
17. **Headline figures** use two pre-set selection rules (A: as planned; B: well-recorded visits), after rule A picked missingness-driven patients.
18. **Added analyses:** the missing-data check on drift, and credit to never-changing features.

# Open questions for the user / guide

1. Confirm eps (0.05 used; results are robust to it).
2. Confirm the feature groups (draft used).
3. Main claim versus population SHAP differences, the strongest competitor on stability.
4. How to present PPMI (weak model, borderline effect).
5. Verify in ADNI documentation that LOGMEM_DELAYED is a diagnostic cut-off.
