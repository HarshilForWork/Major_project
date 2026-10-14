# TRACE Stage 3 gate

Tolerances: G1, G2, G4 <= 1e-5; G3 <= 1e-6 (configs/trace.yaml).
G1 is measured against the exact float64 tree sum (user decision, 14 Oct 2026); the number in
brackets is against XGBoost's float32 predict (results/gate/G1_diagnosis.md). G4 uses k <= 13
for PPMI (user decision: only 1,488 PPMI pairs have k <= 12) and k <= 12 otherwise.

| Model | G1 add-up vs exact float64 (pair / trajectory) [vs XGBoost float32] | G2 consistency (triples / lstsq / background) | G3 zero (constant / unused) | G4 brute force (max / antisymmetry, pairs) | Pilot code (max, pairs) | Result |
|---|---|---|---|---|---|---|
| ppmi | 7.9e-08 / 6.6e-08 [3.3e-06 / 2.2e-06] | 2.2e-16 / 1.3e-15 / 0.0e+00 | 0.0e+00 (221 cases) / n/a, every feature is used | 1.5e-06 / 0.0e+00, 2000 (k <= 13) | 2.8e-08, 2000 | G1 PASS G2 PASS G3 PASS G4 PASS pilot PASS |
| adni_minus | 6.4e-08 / 4.4e-08 [7.4e-06 / 4.9e-06] | 8.9e-16 / 3.6e-15 / 0.0e+00 | 0.0e+00 (2920 cases) / n/a, every feature is used | 3.9e-06 / 0.0e+00, 2000 (k <= 12) | 4.1e-08, 2000 | G1 PASS G2 PASS G3 PASS G4 PASS pilot PASS |
| adni_minus_lm | 5.6e-08 / 3.6e-08 [9.0e-06 / 6.3e-06] | 8.9e-16 / 4.0e-15 / 0.0e+00 | 0.0e+00 (2881 cases) / n/a, every feature is used | 4.3e-06 / 0.0e+00, 2000 (k <= 12) | 3.1e-08, 2000 | G1 PASS G2 PASS G3 PASS G4 PASS pilot PASS |
| adni_minus2 | 6.2e-08 / 4.3e-08 [9.0e-06 / 5.6e-06] | 8.9e-16 / 3.3e-15 / 0.0e+00 | 0.0e+00 (2845 cases) / n/a, every feature is used | 4.1e-06 / 0.0e+00, 2000 (k <= 12) | 3.4e-08, 2000 | G1 PASS G2 PASS G3 PASS G4 PASS pilot PASS |
| adni_full | 9.1e-08 / 5.3e-08 [1.0e-05 / 7.8e-06] | 1.8e-15 / 5.8e-15 / 0.0e+00 | 0.0e+00 (2946 cases) / n/a, every feature is used | 5.2e-06 / 0.0e+00, 2000 (k <= 12) | 6.5e-08, 2000 | G1 PASS G2 PASS G3 PASS G4 PASS pilot PASS |
