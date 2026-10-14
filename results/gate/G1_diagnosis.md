# G1 failure on adni_full: diagnosis (14 Oct 2026)

**Official result (unchanged):** `adni_full` G1 = 1.03e-5 > 1e-5, so it **FAILS**. The tolerance was not loosened.

**Cause: XGBoost's own float32 arithmetic, not TRACE.**
- `booster.predict(output_margin=True)` sums 300 tree outputs in float32.
- Against the exact float64 sum of the same leaf values (the pilot's tree walker, `notebooks/trace_pilot.py`), XGBoost's margins are off by up to 8.8e-6 per visit.
- A difference of two margins can therefore be off by up to about 1.8e-5. That exceeds the G1 tolerance even when the explanation is exact.

| Model | G1 vs XGBoost float32 margin (official) | G1 vs exact float64 tree sum | XGBoost predict error vs exact float64 | max abs(margin) |
|---|---|---|---|---|
| ppmi | 3.28e-06 | 7.88e-08 | 2.44e-06 | 5.3 |
| adni_minus | 7.43e-06 | 6.35e-08 | 6.15e-06 | 9.4 |
| adni_minus_lm | 8.98e-06 | 5.59e-08 | 7.45e-06 | 8.9 |
| adni_minus2 | 9.03e-06 | 6.23e-08 | 5.60e-06 | 8.4 |
| adni_full | **1.03e-05** | 9.13e-08 | 8.79e-06 | 9.8 |

**Corroboration:** the pilot-code cross-check, which compares F with exact float64 tree Shapley, agrees to at most 6.5e-8 on adni_full.

The same float32 rounding is why G4 (which also scores hybrids with `booster.predict`) shows errors of 1e-6 to 5e-6 rather than ~1e-8.

**Options (user decision):**
1. Keep XGBoost's float32 margin as `f` and record G1 on adni_full as a documented fail.
2. For the exactness gates (G1, G4), use the exact float64 tree sum as `f`. This is the same model with more precise arithmetic. All G1 errors are then around 1e-7.
