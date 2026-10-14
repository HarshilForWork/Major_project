# TRACE Stage 4: size of the problem

Patient-level means with patient-bootstrap 95% CIs (1,000 draws). Definitions are in the module docstring of `src/trace_pd/evaluation/trace_stage4.py`. Aggregates only.

## Pilot reproduction (all visits)

| Model | Metric | Now | Pilot |
|---|---|---|---|
| ppmi | top_differs | 30.7% | 30.7% |
| ppmi | clear_top_differs | 11.4% | 11.4% |
| ppmi | rel_gap_median | 51.8% | 51.8% |
| ppmi | sign_flip | 0.0% | 0.0% |
| ppmi | trace_change_real | 17.3% | 17.3% |
| ppmi | trace_change_real_n | 10997 | 10997 |
| ppmi | locality_10pct | 0.1% | 0.1% |
| ppmi | I_median | 0.030 | 0.030 |
| adni_minus | top_differs | 22.9% | 22.9% |
| adni_minus | clear_top_differs | 8.8% | 8.8% |
| adni_minus | rel_gap_median | 38.6% | 38.6% |
| adni_minus | sign_flip | 0.0% | 0.0% |
| adni_minus | trace_change_real | 7.7% | 7.7% |
| adni_minus | trace_change_real_n | 24099 | 24099 |
| adni_minus | locality_10pct | 1.2% | 1.2% |
| adni_minus | I_median | 0.015 | 0.015 |

## Chain vs direct, compared with seed noise

| Run | Patients | First->last: top differs | ... seed noise | ... difference | Gap pairs: top differs | ... seed noise | ... difference | First->last total disagreement (median) |
|---|---|---|---|---|---|---|---|---|
| ppmi [all] | 88 | 30.7% [20.5, 40.9] | 20.3% [14.9, 25.9] | 10.4% [-0.0, 20.5] | 30.3% [28.0, 32.6] | 25.0% [23.6, 26.5] | 5.3% [2.5, 7.9] | 51.8% |
| adni_minus [all] | 475 | 22.9% [19.2, 26.9] | 11.5% [9.7, 13.2] | 11.4% [7.5, 15.5] | 18.7% [17.0, 20.2] | 9.7% [8.9, 10.6] | 9.0% [7.3, 10.6] | 38.6% |
| adni_minus_lm [all] | 475 | 18.7% [15.4, 22.3] | 11.2% [9.3, 12.9] | 7.6% [4.0, 11.0] | 16.2% [14.8, 17.6] | 9.7% [8.9, 10.5] | 6.5% [5.1, 7.9] | 34.6% |
| adni_minus2 [all] | 475 | 15.4% [12.2, 18.5] | 9.6% [8.0, 11.2] | 5.8% [2.7, 9.0] | 15.2% [14.0, 16.5] | 9.2% [8.5, 10.0] | 6.0% [4.8, 7.1] | 26.7% |
| ppmi [labelled] | 88 | 28.4% [20.5, 38.6] | 23.0% [17.8, 28.5] | 5.5% [-4.0, 15.1] | 23.6% [21.2, 26.1] | 25.8% [23.7, 28.3] | -2.3% [-5.6, 0.9] | 45.4% |
| adni_minus [labelled] | 426 | 20.7% [16.9, 24.9] | 10.9% [9.2, 13.0] | 9.7% [6.2, 13.3] | 14.3% [12.3, 16.5] | 11.1% [9.9, 12.5] | 3.1% [1.2, 5.1] | 32.8% |

## TRACE change rate and locality price

| Run | TRACE vs direct top differs (first->last) | (gap pairs) | (all pairs) | unchanged feature > 10% of pair explanation | > 5% | abs > 0.05 log-odds | 99th pct abs |
|---|---|---|---|---|---|---|---|
| ppmi [all] | 8.0% [3.4, 13.6] | 17.0% [15.6, 18.4] | 16.9% [15.6, 18.0] | 0.14% | 0.66% | 5.72% | 0.100 |
| adni_minus [all] | 6.1% [4.2, 8.4] | 7.1% [6.3, 8.0] | 7.1% [6.3, 7.8] | 1.16% | 3.12% | 9.53% | 0.240 |
| adni_minus_lm [all] | 5.3% [3.4, 7.4] | 6.1% [5.4, 6.8] | 6.6% [5.9, 7.4] | 0.99% | 2.85% | 8.38% | 0.207 |
| adni_minus2 [all] | 4.8% [2.9, 6.7] | 6.1% [5.5, 6.8] | 6.4% [5.7, 7.0] | 1.23% | 3.05% | 8.11% | 0.168 |
| ppmi [labelled] | 11.4% [5.7, 18.2] | 12.5% [11.1, 13.8] | 12.4% [11.1, 13.7] | 0.03% | 0.49% | 3.74% | 0.085 |
| adni_minus [labelled] | 6.1% [4.0, 8.5] | 5.0% [4.1, 6.0] | 6.1% [5.4, 6.9] | 0.21% | 1.10% | 6.34% | 0.179 |

## Trust score I

| Run | median [IQR] | 90th pct | by visits T (n: median) | Spearman(I, first->last disagreement) | Spearman(I, gap-pair disagreement) |
|---|---|---|---|---|---|
| ppmi [all] | 0.030 [0.022, 0.035] | 0.042 | 3-4: 1: 0.013; 5-9: 8: 0.018; 10-14: 23: 0.022; 15-99: 56: 0.033 | 0.38 [0.16, 0.57] | 0.39 [0.16, 0.61] |
| adni_minus [all] | 0.015 [0.010, 0.020] | 0.027 | 3-4: 76: 0.009; 5-9: 203: 0.014; 10-14: 116: 0.015; 15-99: 80: 0.019 | 0.50 [0.42, 0.58] | 0.43 [0.34, 0.51] |
| adni_minus_lm [all] | 0.013 [0.008, 0.017] | 0.023 | 3-4: 76: 0.007; 5-9: 203: 0.013; 10-14: 116: 0.013; 15-99: 80: 0.017 | 0.50 [0.42, 0.58] | 0.41 [0.32, 0.49] |
| adni_minus2 [all] | 0.009 [0.004, 0.015] | 0.020 | 3-4: 76: 0.003; 5-9: 203: 0.007; 10-14: 116: 0.011; 15-99: 80: 0.017 | 0.55 [0.47, 0.62] | 0.51 [0.43, 0.58] |
| ppmi [labelled] | 0.025 [0.019, 0.032] | 0.040 | 3-4: 6: 0.012; 5-9: 21: 0.018; 10-14: 18: 0.029; 15-99: 43: 0.027 | 0.49 [0.28, 0.64] | 0.74 [0.62, 0.84] |
| adni_minus [labelled] | 0.013 [0.007, 0.019] | 0.024 | 3-4: 180: 0.008; 5-9: 199: 0.015; 10-14: 40: 0.018; 15-99: 7: 0.023 | 0.58 [0.51, 0.64] | 0.61 [0.54, 0.67] |

Seed check: max margin difference between retrained seeds = ppmi: 1.06, adni_minus: 1.27, adni_minus_lm: 1.34, adni_minus2: 1.62
