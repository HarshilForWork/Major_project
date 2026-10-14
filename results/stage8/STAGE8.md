# Stage 8: sanity check (patient-level shuffled labels)

Retrained on the same development visits with each patient's baseline label permuted across patients (real visits, shuffled labels; not synthetic data). TRACE recomputed for the same locked-test patients.

| Model | Shuffled-label model: locked-test BA | Spearman of per-feature mean abs(u), real vs shuffled | Per-visit Spearman of tallies (mean) | Total mean abs(u): real / shuffled |
|---|---|---|---|---|
| ppmi | 0.526 | 0.30 | -0.02 | 2.58 / 2.01 |
| adni_minus | 0.516 | 0.46 | -0.11 | 4.13 / 0.69 |

Reading: at the level of individual visits the explanations fall apart (per-visit agreement about 0). The per-feature importance ranking keeps a weak-to-moderate correlation, because features that change more often or more strongly can receive credit even from a model with no real signal.
