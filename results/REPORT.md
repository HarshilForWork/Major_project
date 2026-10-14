# TRACE: full report (brief Stage 10)



Aggregates only; no patient identifiers. Plain-language summary: `results/SUMMARY_FOR_GUIDE.md`. Log: `results/PROGRESS.md`.



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


# Stage 6: faithfulness

Revert the top-k features (by |credit|) of the later visit to the earlier visit's real values; share of the score change recovered. Patient means, 95% patient-bootstrap CI. eps was OPEN: main = 0.05 (pairs with |score change| < eps excluded), sensitivity 0.01 / 0.1 / 0.25. Direct Baseline Shapley wins on its own pair by construction (upper reference); TRACE's claim is consistency across all pairs.

Figure: `results/figures/fig_faithfulness.png`.

## ppmi (88 patients; eps 0.05: 10738 pairs kept, 311 excluded; AR(1): 9478 pairs, s >= 1 only)

| Method | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| trace | 0.503 [0.462, 0.544] | 0.664 [0.625, 0.701] | 0.765 [0.727, 0.805] | 0.846 [0.815, 0.877] | 0.892 [0.864, 0.918] |
| direct | 0.528 [0.484, 0.571] | 0.666 [0.629, 0.702] | 0.771 [0.732, 0.809] | 0.835 [0.806, 0.865] | 0.886 [0.862, 0.912] |
| chain | 0.464 [0.423, 0.501] | 0.617 [0.581, 0.655] | 0.734 [0.695, 0.772] | 0.809 [0.773, 0.845] | 0.857 [0.828, 0.890] |
| fixed_first | 0.485 [0.439, 0.529] | 0.636 [0.598, 0.674] | 0.749 [0.708, 0.788] | 0.811 [0.780, 0.843] | 0.858 [0.832, 0.883] |
| fixed_last | 0.443 [0.399, 0.485] | 0.627 [0.591, 0.662] | 0.738 [0.703, 0.771] | 0.807 [0.776, 0.839] | 0.856 [0.827, 0.886] |
| population | 0.492 [0.451, 0.534] | 0.649 [0.608, 0.689] | 0.788 [0.751, 0.824] | 0.844 [0.816, 0.872] | 0.893 [0.866, 0.921] |
| path_dep | 0.497 [0.457, 0.536] | 0.663 [0.626, 0.698] | 0.788 [0.750, 0.825] | 0.853 [0.823, 0.882] | 0.902 [0.875, 0.929] |
| ar1 | 0.256 [0.211, 0.298] | 0.389 [0.347, 0.429] | 0.472 [0.433, 0.511] | 0.530 [0.492, 0.572] | 0.567 [0.528, 0.608] |

Paired TRACE minus method (> 0 = TRACE recovers more):

| Method | k=1 | k=3 | k=5 |
|---|---|---|---|
| direct | -0.025 [-0.047, -0.001] | -0.006 [-0.024, 0.011] | 0.005 [-0.006, 0.016] |
| chain | 0.039 [0.008, 0.069] | 0.031 [0.006, 0.059] | 0.035 [0.018, 0.051] |
| fixed_first | 0.018 [-0.012, 0.049] | 0.016 [-0.010, 0.044] | 0.034 [0.018, 0.050] |
| fixed_last | 0.060 [0.030, 0.090] | 0.027 [0.002, 0.052] | 0.035 [0.015, 0.055] |
| population | 0.012 [-0.014, 0.041] | -0.022 [-0.046, -0.000] | -0.001 [-0.018, 0.015] |
| path_dep | 0.007 [-0.019, 0.035] | -0.023 [-0.048, 0.000] | -0.010 [-0.027, 0.007] |
| ar1 | 0.226 [0.177, 0.278] | 0.284 [0.235, 0.333] | 0.322 [0.278, 0.366] |

eps sensitivity (TRACE recovered at k = 1 / 3 / 5; best other method at k = 3):

- eps_0.01: TRACE 0.477 / 0.779 / 0.895 (10966 pairs); best other at k=3: population 0.824
- eps_0.05: TRACE 0.503 / 0.765 / 0.892 (10738 pairs); best other at k=3: path_dep 0.788
- eps_0.1: TRACE 0.484 / 0.761 / 0.887 (10413 pairs); best other at k=3: path_dep 0.781
- eps_0.25: TRACE 0.475 / 0.767 / 0.888 (9495 pairs); best other at k=3: path_dep 0.787

## adni_minus (475 patients; eps 0.05: 23868 pairs kept, 1539 excluded; AR(1): 19904 pairs, s >= 1 only)

| Method | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| trace | 0.706 [0.603, 0.815] | 0.729 [0.685, 0.777] | 0.886 [0.828, 0.942] | 0.895 [0.863, 0.925] | 0.959 [0.934, 0.987] |
| direct | 0.745 [0.649, 0.850] | 0.754 [0.710, 0.800] | 0.898 [0.837, 0.955] | 0.902 [0.872, 0.931] | 0.964 [0.939, 0.991] |
| chain | 0.621 [0.525, 0.722] | 0.720 [0.675, 0.768] | 0.865 [0.792, 0.928] | 0.872 [0.836, 0.906] | 0.941 [0.908, 0.976] |
| fixed_first | 0.687 [0.581, 0.801] | 0.717 [0.675, 0.766] | 0.892 [0.830, 0.948] | 0.892 [0.860, 0.925] | 0.957 [0.927, 0.987] |
| fixed_last | 0.694 [0.587, 0.808] | 0.725 [0.683, 0.772] | 0.891 [0.837, 0.944] | 0.893 [0.859, 0.922] | 0.955 [0.927, 0.986] |
| population | 0.664 [0.560, 0.769] | 0.709 [0.664, 0.757] | 0.857 [0.802, 0.914] | 0.878 [0.845, 0.911] | 0.947 [0.919, 0.977] |
| path_dep | 0.687 [0.581, 0.787] | 0.701 [0.645, 0.756] | 0.840 [0.785, 0.896] | 0.869 [0.837, 0.900] | 0.934 [0.904, 0.966] |
| ar1 | 0.444 [0.369, 0.521] | 0.596 [0.526, 0.684] | 0.661 [0.595, 0.732] | 0.754 [0.681, 0.838] | 0.840 [0.763, 0.944] |

Paired TRACE minus method (> 0 = TRACE recovers more):

| Method | k=1 | k=3 | k=5 |
|---|---|---|---|
| direct | -0.040 [-0.061, -0.019] | -0.012 [-0.027, 0.007] | -0.005 [-0.012, 0.002] |
| chain | 0.084 [0.024, 0.165] | 0.021 [-0.012, 0.049] | 0.018 [-0.005, 0.035] |
| fixed_first | 0.018 [-0.011, 0.049] | -0.006 [-0.032, 0.015] | 0.002 [-0.011, 0.013] |
| fixed_last | 0.011 [-0.031, 0.058] | -0.005 [-0.035, 0.022] | 0.004 [-0.008, 0.016] |
| population | 0.042 [-0.062, 0.170] | 0.029 [-0.015, 0.070] | 0.012 [-0.003, 0.029] |
| path_dep | 0.019 [-0.082, 0.145] | 0.045 [0.005, 0.084] | 0.025 [0.009, 0.042] |
| ar1 | 0.304 [0.229, 0.390] | 0.201 [0.121, 0.287] | 0.130 [0.048, 0.193] |

eps sensitivity (TRACE recovered at k = 1 / 3 / 5; best other method at k = 3):

- eps_0.01: TRACE 0.783 / 0.903 / 0.968 (24409 pairs); best other at k=3: direct 0.916
- eps_0.05: TRACE 0.706 / 0.886 / 0.959 (23868 pairs); best other at k=3: direct 0.898
- eps_0.1: TRACE 0.612 / 0.846 / 0.958 (23273 pairs); best other at k=3: direct 0.862
- eps_0.25: TRACE 0.655 / 0.844 / 0.946 (21893 pairs); best other at k=3: fixed_first 0.856


# TRACE Stage 7: stability

Patient means with patient-bootstrap 95% CIs (1,000 draws). Aggregates only. Units: log-odds.

## ppmi: delete one visit (88 patients with >= 4 visits)

Bound check (Prop. 4): worst max_j|dpsi| / (4R/(T-1)) = 0.0094; violations = 0 (R = 22.90).

AR(1) own Monte Carlo noise (same visit, two seeds), max_j |diff|: mean 0.0163, p95 0.0290 (n = 264).

| Deleted visit | Method | mean over pairs of max_j abs(dA) | mean over pairs of mean_j abs(dA) | TRACE minus method (max metric; < 0 = TRACE moves less) |
|---|---|---|---|---|
| first | trace | 0.013 [0.012, 0.014] | 0.0030 [0.0028, 0.0033] | 0.000 [0.000, 0.000] |
| first | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.014] |
| first | fixed_first | 0.148 [0.139, 0.158] | 0.0309 [0.0287, 0.0332] | -0.135 [-0.144, -0.126] |
| first | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.014] |
| first | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.014] |
| first | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.014] |
| first | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.014] |
| first | ar1 | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.013 [0.012, 0.015] |
| intermediate | trace | 0.012 [0.011, 0.013] | 0.0027 [0.0025, 0.0030] | 0.000 [0.000, 0.000] |
| intermediate | chain | 0.050 [0.047, 0.054] | 0.0094 [0.0087, 0.0102] | -0.038 [-0.042, -0.036] |
| intermediate | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.012 [0.011, 0.013] |
| intermediate | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.012 [0.011, 0.013] |
| intermediate | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.012 [0.011, 0.013] |
| intermediate | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.012 [0.011, 0.013] |
| intermediate | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.012 [0.011, 0.013] |
| intermediate | ar1 | 0.031 [0.026, 0.036] | 0.0044 [0.0037, 0.0052] | -0.019 [-0.024, -0.015] |
| last | trace | 0.014 [0.013, 0.016] | 0.0033 [0.0030, 0.0037] | 0.000 [0.000, 0.000] |
| last | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.016] |
| last | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.016] |
| last | fixed_last | 0.170 [0.150, 0.188] | 0.0359 [0.0312, 0.0403] | -0.156 [-0.174, -0.136] |
| last | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.016] |
| last | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.016] |
| last | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.016] |
| last | ar1 | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.014 [0.013, 0.015] |

## adni_minus: delete one visit (439 patients with >= 4 visits)

Bound check (Prop. 4): worst max_j|dpsi| / (4R/(T-1)) = 0.0184; violations = 0 (R = 26.84).

AR(1) own Monte Carlo noise (same visit, two seeds), max_j |diff|: mean 0.0494, p95 0.1458 (n = 1389).

| Deleted visit | Method | mean over pairs of max_j abs(dA) | mean over pairs of mean_j abs(dA) | TRACE minus method (max metric; < 0 = TRACE moves less) |
|---|---|---|---|---|
| first | trace | 0.045 [0.042, 0.048] | 0.0082 [0.0076, 0.0087] | 0.000 [0.000, 0.000] |
| first | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.045 [0.042, 0.048] |
| first | fixed_first | 0.362 [0.347, 0.379] | 0.0599 [0.0573, 0.0627] | -0.317 [-0.333, -0.303] |
| first | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.045 [0.042, 0.048] |
| first | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.045 [0.042, 0.048] |
| first | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.045 [0.042, 0.048] |
| first | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.045 [0.042, 0.048] |
| first | ar1 | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.046 [0.043, 0.050] |
| intermediate | trace | 0.044 [0.041, 0.047] | 0.0075 [0.0070, 0.0081] | 0.000 [0.000, 0.000] |
| intermediate | chain | 0.163 [0.154, 0.172] | 0.0261 [0.0247, 0.0276] | -0.119 [-0.126, -0.113] |
| intermediate | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.044 [0.041, 0.047] |
| intermediate | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.044 [0.041, 0.047] |
| intermediate | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.044 [0.041, 0.047] |
| intermediate | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.044 [0.041, 0.047] |
| intermediate | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.044 [0.041, 0.047] |
| intermediate | ar1 | 0.136 [0.117, 0.154] | 0.0170 [0.0146, 0.0194] | -0.095 [-0.111, -0.078] |
| last | trace | 0.041 [0.038, 0.045] | 0.0072 [0.0067, 0.0078] | 0.000 [0.000, 0.000] |
| last | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| last | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| last | fixed_last | 0.350 [0.323, 0.376] | 0.0581 [0.0536, 0.0625] | -0.309 [-0.333, -0.283] |
| last | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| last | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| last | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| last | ar1 | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.040 [0.037, 0.044] |

## adni_minus_lm: delete one visit (439 patients with >= 4 visits)

Bound check (Prop. 4): worst max_j|dpsi| / (4R/(T-1)) = 0.0175; violations = 0 (R = 26.36).

| Deleted visit | Method | mean over pairs of max_j abs(dA) | mean over pairs of mean_j abs(dA) | TRACE minus method (max metric; < 0 = TRACE moves less) |
|---|---|---|---|---|
| first | trace | 0.041 [0.038, 0.045] | 0.0073 [0.0068, 0.0078] | 0.000 [0.000, 0.000] |
| first | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| first | fixed_first | 0.326 [0.312, 0.342] | 0.0542 [0.0518, 0.0566] | -0.285 [-0.299, -0.272] |
| first | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| first | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| first | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| first | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.045] |
| intermediate | trace | 0.041 [0.038, 0.044] | 0.0068 [0.0064, 0.0073] | 0.000 [0.000, 0.000] |
| intermediate | chain | 0.152 [0.143, 0.162] | 0.0237 [0.0225, 0.0251] | -0.112 [-0.118, -0.105] |
| intermediate | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.044] |
| intermediate | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.044] |
| intermediate | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.044] |
| intermediate | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.044] |
| intermediate | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.041 [0.038, 0.044] |
| last | trace | 0.039 [0.036, 0.042] | 0.0066 [0.0061, 0.0071] | 0.000 [0.000, 0.000] |
| last | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.039 [0.036, 0.042] |
| last | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.039 [0.036, 0.042] |
| last | fixed_last | 0.327 [0.302, 0.353] | 0.0534 [0.0493, 0.0574] | -0.289 [-0.312, -0.265] |
| last | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.039 [0.036, 0.042] |
| last | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.039 [0.036, 0.042] |
| last | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.039 [0.036, 0.042] |

## adni_minus2: delete one visit (439 patients with >= 4 visits)

Bound check (Prop. 4): worst max_j|dpsi| / (4R/(T-1)) = 0.0148; violations = 0 (R = 27.32).

| Deleted visit | Method | mean over pairs of max_j abs(dA) | mean over pairs of mean_j abs(dA) | TRACE minus method (max metric; < 0 = TRACE moves less) |
|---|---|---|---|---|
| first | trace | 0.030 [0.028, 0.032] | 0.0055 [0.0051, 0.0058] | 0.000 [0.000, 0.000] |
| first | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| first | fixed_first | 0.281 [0.267, 0.296] | 0.0470 [0.0448, 0.0492] | -0.251 [-0.264, -0.238] |
| first | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| first | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| first | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| first | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| intermediate | trace | 0.030 [0.028, 0.032] | 0.0052 [0.0049, 0.0055] | 0.000 [0.000, 0.000] |
| intermediate | chain | 0.111 [0.105, 0.118] | 0.0181 [0.0171, 0.0191] | -0.081 [-0.086, -0.076] |
| intermediate | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| intermediate | fixed_last | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| intermediate | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| intermediate | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| intermediate | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.028, 0.032] |
| last | trace | 0.030 [0.027, 0.032] | 0.0052 [0.0049, 0.0056] | 0.000 [0.000, 0.000] |
| last | chain | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.027, 0.032] |
| last | fixed_first | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.027, 0.032] |
| last | fixed_last | 0.250 [0.232, 0.269] | 0.0423 [0.0393, 0.0454] | -0.221 [-0.238, -0.203] |
| last | direct | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.027, 0.032] |
| last | population | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.027, 0.032] |
| last | path_dep | 0.000 [0.000, 0.000] | 0.0000 [0.0000, 0.0000] | 0.030 [0.027, 0.032] |

## Retraining stability (10 seed models; Spearman across features, same pair)

| Model | Method | Mean Spearman [95% CI] | TRACE minus method, paired (> 0 = TRACE more stable) |
|---|---|---|---|
| ppmi | trace | 0.845 [0.837, 0.853] |  |
| ppmi | direct | 0.834 [0.827, 0.840] | 0.011 [0.008, 0.015] |
| ppmi | chain | 0.804 [0.796, 0.811] | 0.041 [0.037, 0.045] |
| ppmi | fixed_first | 0.819 [0.811, 0.826] | 0.026 [0.023, 0.030] |
| ppmi | fixed_last | 0.815 [0.808, 0.822] | 0.029 [0.024, 0.035] |
| ppmi | population | 0.882 [0.875, 0.888] | -0.037 [-0.041, -0.033] |
| ppmi | path_dep | 0.882 [0.875, 0.888] | -0.037 [-0.042, -0.033] |
| adni_minus | trace | 0.823 [0.816, 0.829] |  |
| adni_minus | direct | 0.859 [0.853, 0.865] | -0.036 [-0.042, -0.031] |
| adni_minus | chain | 0.830 [0.824, 0.835] | -0.007 [-0.012, -0.002] |
| adni_minus | fixed_first | 0.814 [0.808, 0.820] | 0.009 [0.007, 0.011] |
| adni_minus | fixed_last | 0.848 [0.842, 0.854] | -0.025 [-0.031, -0.020] |
| adni_minus | population | 0.842 [0.834, 0.848] | -0.019 [-0.022, -0.015] |
| adni_minus | path_dep | 0.841 [0.834, 0.848] | -0.018 [-0.022, -0.015] |
| adni_minus_lm | trace | 0.820 [0.813, 0.826] |  |
| adni_minus_lm | direct | 0.863 [0.858, 0.868] | -0.043 [-0.049, -0.038] |
| adni_minus_lm | chain | 0.830 [0.824, 0.835] | -0.010 [-0.016, -0.005] |
| adni_minus_lm | fixed_first | 0.812 [0.805, 0.818] | 0.008 [0.006, 0.010] |
| adni_minus_lm | fixed_last | 0.850 [0.844, 0.855] | -0.030 [-0.036, -0.024] |
| adni_minus_lm | population | 0.840 [0.832, 0.847] | -0.020 [-0.024, -0.016] |
| adni_minus_lm | path_dep | 0.837 [0.829, 0.844] | -0.017 [-0.021, -0.013] |
| adni_minus2 | trace | 0.793 [0.786, 0.799] |  |
| adni_minus2 | direct | 0.837 [0.831, 0.843] | -0.044 [-0.051, -0.038] |
| adni_minus2 | chain | 0.804 [0.799, 0.809] | -0.011 [-0.018, -0.006] |
| adni_minus2 | fixed_first | 0.783 [0.776, 0.789] | 0.010 [0.007, 0.013] |
| adni_minus2 | fixed_last | 0.827 [0.820, 0.832] | -0.033 [-0.040, -0.027] |
| adni_minus2 | population | 0.829 [0.822, 0.835] | -0.036 [-0.040, -0.031] |
| adni_minus2 | path_dep | 0.825 [0.818, 0.831] | -0.032 [-0.036, -0.027] |


## Credit given to features that never change for the patient

| Model | trace | chain | direct | fixed_first | fixed_last | population | path_dep |
|---|---|---|---|---|---|---|---|
| ppmi | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 2.88% (max 0.43) | 2.94% (max 0.41) |
| adni_minus | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 2.86% (max 0.46) | 2.32% (max 0.35) |
| adni_minus_lm | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 3.33% (max 0.55) | 2.57% (max 0.45) |
| adni_minus2 | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 0.00% (max 0.00) | 3.99% (max 0.56) | 3.15% (max 0.54) |


# Stage 8: sanity check (patient-level shuffled labels)

Retrained on the same development visits with each patient's baseline label permuted across patients (real visits, shuffled labels; not synthetic data). TRACE recomputed for the same locked-test patients.

| Model | Shuffled-label model: locked-test BA | Spearman of per-feature mean abs(u), real vs shuffled | Per-visit Spearman of tallies (mean) | Total mean abs(u): real / shuffled |
|---|---|---|---|---|
| ppmi | 0.526 | 0.30 | -0.02 | 2.58 / 2.01 |
| adni_minus | 0.516 | 0.46 | -0.11 | 4.13 / 0.69 |

Reading: at the level of individual visits the explanations fall apart (per-visit agreement about 0). The per-feature importance ranking keeps a weak-to-moderate correlation, because features that change more often or more strongly can receive credit even from a model with no real signal.


# Stage 9: descriptive clinical analyses (not causal)

Feature groups: `configs/feature_groups.yaml`, **DRAFT, not yet confirmed**. TRACE explains what the model reacts to, not disease mechanisms. Medication can look important because sicker patients get more medication.

## ppmi

**Model-predicted flips** (score crosses the decision threshold between consecutive visits): 233 (150 towards the positive class, 83 away). Consecutive pairs with both visits labelled: 799; true-label flips: 99; of these also model flips: 18.

| Group | Mean share of a flip's TRACE credit | Mean signed credit, flips up | flips down |
|---|---|---|---|
| motor_daily_living | 35.9% | +0.66 | -0.58 |
| non_motor_questionnaires | 27.6% | +0.32 | -0.29 |
| medication | 14.3% | +0.30 | -0.12 |
| time | 12.1% | +0.31 | -0.22 |
| cognition | 10.1% | +0.07 | -0.13 |
| demographic_static | 0.0% | +0.00 | +0.00 |

Top features by mean share of flip credit: NP2RISE 10.6%; MCATOT 10.1%; LEDD_TOTAL_MG 9.7%; STAI_TOTAL 5.6%; YRS_SINCE_DIAGNOSIS 5.5%; NP2SPCH 5.4%; SCOPA_AUT_TOTAL 4.6%; ESS_TOTAL 4.5%

**Measurement accounting:** 44.5% of the total abs(TRACE tally) comes from comparisons where the feature's recorded/missing status differs (indicative; not sign-separated).

**Within vs between patient** (population SHAP = TRACE within-patient part + between-patient part b). Identity check sum_j b_j(t) = patient mean score - background mean score at every visit: max error 3.6e-08 (tolerance 1e-05): **PASS**. Within-patient share of total credit: 53.9%.

| Group | mean abs(within u) | mean abs(between b) |
|---|---|---|
| demographic_static | 0.000 | 0.428 |
| time | 0.525 | 0.288 |
| medication | 0.506 | 0.386 |
| motor_daily_living | 0.876 | 0.686 |
| non_motor_questionnaires | 0.507 | 0.348 |
| cognition | 0.149 | 0.057 |

| Feature (top 10) | mean abs(within u) | mean abs(between b) | within-patient SD of b |
|---|---|---|---|
| LEDD_TOTAL_MG | 0.290 | 0.194 | 0.059 |
| YRS_SINCE_DIAGNOSIS | 0.281 | 0.117 | 0.034 |
| NP2RISE | 0.223 | 0.160 | 0.029 |
| SEX | 0.000 | 0.350 | 0.052 |
| YRS_SINCE_SYMPTOM_ONSET | 0.170 | 0.097 | 0.038 |
| NP2TURN | 0.141 | 0.086 | 0.023 |
| N_CONMEDS | 0.117 | 0.094 | 0.059 |
| MCATOT | 0.149 | 0.057 | 0.034 |
| NP2SPCH | 0.094 | 0.107 | 0.038 |
| PDTRTMNT | 0.099 | 0.097 | 0.031 |

## adni_minus

**Model-predicted flips** (score crosses the decision threshold between consecutive visits): 1046 (476 towards the positive class, 570 away). Consecutive pairs with both visits labelled: 496; true-label flips: 1; of these also model flips: 0.

| Group | Mean share of a flip's TRACE credit | Mean signed credit, flips up | flips down |
|---|---|---|---|
| cognitive_tests | 56.4% | +2.99 | -3.10 |
| vitals_body | 26.7% | +1.62 | -1.78 |
| mood_behaviour | 13.8% | +0.36 | -0.24 |
| function_daily_living | 2.4% | +0.12 | -0.10 |
| time | 0.7% | +0.01 | +0.01 |
| demographic_static | 0.0% | +0.00 | +0.00 |

Top features by mean share of flip credit: HEIGHT_CM 18.9%; LOGMEM_DELAYED 18.3%; CATEGORY_FLUENCY_ANIMALS 14.5%; NPI_TOTAL 9.9%; TRAILS_B_SEC 6.4%; LOGMEM_IMMEDIATE 6.3%; BOSTON_NAMING 5.7%; WEIGHT_KG 4.3%

**Measurement accounting:** 88.4% of the total abs(TRACE tally) comes from comparisons where the feature's recorded/missing status differs (indicative; not sign-separated).
By ADNI study phase: ADNI1 87.3% (1262 visits); ADNI2 91.8% (2067 visits); ADNI3 84.3% (673 visits); ADNI4 81.9% (318 visits); ADNIGO 88.6% (164 visits).

**Within vs between patient** (population SHAP = TRACE within-patient part + between-patient part b). Identity check sum_j b_j(t) = patient mean score - background mean score at every visit: max error 5.1e-08 (tolerance 1e-05): **PASS**. Within-patient share of total credit: 49.3%.

| Group | mean abs(within u) | mean abs(between b) |
|---|---|---|
| demographic_static | 0.000 | 0.192 |
| time | 0.047 | 0.103 |
| vitals_body | 0.886 | 0.711 |
| mood_behaviour | 0.580 | 0.418 |
| cognitive_tests | 2.205 | 2.355 |
| function_daily_living | 0.251 | 0.308 |

| Feature (top 10) | mean abs(within u) | mean abs(between b) | within-patient SD of b |
|---|---|---|---|
| LOGMEM_DELAYED | 0.731 | 0.731 | 0.157 |
| HEIGHT_CM | 0.605 | 0.491 | 0.108 |
| CATEGORY_FLUENCY_ANIMALS | 0.426 | 0.373 | 0.107 |
| NPI_TOTAL | 0.424 | 0.307 | 0.109 |
| TRAILS_B_SEC | 0.368 | 0.356 | 0.080 |
| MOCA_TOTAL | 0.184 | 0.353 | 0.050 |
| ECOG_PARTNER_TOTAL | 0.193 | 0.242 | 0.045 |
| LOGMEM_IMMEDIATE | 0.185 | 0.191 | 0.056 |
| BOSTON_NAMING | 0.143 | 0.154 | 0.035 |
| WEIGHT_KG | 0.158 | 0.128 | 0.042 |


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
