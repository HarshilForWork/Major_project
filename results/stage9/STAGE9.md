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
