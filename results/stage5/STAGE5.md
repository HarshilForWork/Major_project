# TRACE Stage 5: baselines and their efficiency gap

Efficiency gap = |sum_j A_j(s->t) - (f(x^t) - f(x^s))|, with f the exact float64 margin, over all pairs s < t of the analysed locked-test patients. Aggregates only.

| Model | Method | Pairs | Max gap | Mean gap (patient means) |
|---|---|---|---|---|
| ppmi | trace | 11049 | 7.88e-08 | 1.20e-08 |
| ppmi | direct | 11049 | 1.22e-07 | 1.26e-08 |
| ppmi | chain | 11049 | 1.03e-07 | 1.38e-08 |
| ppmi | fixed_first | 11049 | 1.07e-07 | 1.72e-08 |
| ppmi | fixed_last | 11049 | 1.07e-07 | 1.57e-08 |
| ppmi | population | 11049 | 6.34e-08 | 1.22e-08 |
| ppmi | path_dep | 11049 | 1.38e-06 | 1.90e-07 |
| ppmi | ar1 | 9755 | 6.54e+00 | 1.45e+00 |
| ppmi | ar1 Monte Carlo noise (same visit, two seeds) | 6864 values | p95 0.013 | mean 0.0032 |
| adni_minus | trace | 25407 | 6.35e-08 | 1.09e-08 |
| adni_minus | direct | 25407 | 8.59e-08 | 1.19e-08 |
| adni_minus | chain | 25407 | 1.01e-07 | 1.47e-08 |
| adni_minus | fixed_first | 25407 | 1.04e-07 | 1.35e-08 |
| adni_minus | fixed_last | 25407 | 9.24e-08 | 1.32e-08 |
| adni_minus | population | 25407 | 7.05e-08 | 1.10e-08 |
| adni_minus | path_dep | 25407 | 4.31e-06 | 4.46e-07 |
| adni_minus | ar1 | 21398 | 1.09e+01 | 2.02e+00 |
| adni_minus | ar1 Monte Carlo noise (same visit, two seeds) | 29169 values | p95 0.039 | mean 0.0070 |
