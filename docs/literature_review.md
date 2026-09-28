# Literature Review

The seven papers in the first-review deck (slides 5–6), with verified citations, what each
contributes, and the gap it leaves that TRACE-PD targets.

Citations were checked against arXiv, medRxiv, Crossref and publisher pages. Where the
title we had been using differs from the published title, the published one is given and the
difference is noted.

---

## 1. Conformal prediction for PD clinical decision support

**Diaz-Rincon R., Liang M., Ramirez-Zamora A., Shickel B.** *CASCADE Conformal Prediction:
Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support.* arXiv
preprint, 2026. <https://arxiv.org/abs/2605.20468>
Accepted at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic
Systems. Earlier related work by the same first author: *Uncertainty-Aware Prediction of
Parkinson's Disease Medication Needs: A Two-Stage Conformal Prediction Approach*, MLHC 2025,
arXiv:2508.10284.

> Title in the deck: "CASCADE: Conformal Prediction for Two-Stage Clinical Decision Support
> in PD". The published title doesn't mention PD; PD medication is the application.

| Technique | Advantage | Gap for us |
|---|---|---|
| two-stage conformal: screening classifier → LEDD regression | propagates calibrated uncertainty across stages; statistically guaranteed coverage | applied to **medication dosing**, not subtype; single-visit, no longitudinal tracking |

## 2. 7T MRI for motor-subtype stratification

**Kristoffersen A.L., Unsgård R.G., Fortin M.-A., Kvålsgard I.G., Stige K.E., Doan T.P.,
Berntsen E.M., Tzoulis C., Goa P.E.** *7 Tesla Quantitative MRI and Machine Learning for
Exploratory Motor Subtype Stratification and Diagnosis in Parkinson's Disease.* arXiv
preprint, 2026. <https://arxiv.org/abs/2605.24179>

| Technique | Advantage | Gap for us |
|---|---|---|
| ultra-high-field 7T MRI (MP2RAGE + ASPIRE), several classifiers, 21 HC vs 24 PD | very high-resolution imaging features | only moderate accuracy; exploratory; **no uncertainty calibration**; **imaging-dependent** — 7T is not available where this tool is meant to be used |

> The specific classifier list (KNN / SVM / RF / XGBoost) quoted in the deck couldn't be
> confirmed from the abstract.

## 3. Reliability of LLM clinical reasoning

**Modi M., Krull J.E., Johnson D., Wang X., Gauntner T.D., Li M., Cheng H., Ma A., Zhang P.,
Stover D.G., Li Z., Ma Q.** *Understanding Clinical Reasoning Variability in Medical Large
Language Models: A Mechanistic Interpretability Study.* medRxiv, 2026.
<https://doi.org/10.64898/2026.01.26.26344845> (PMID 41646812)

| Technique | Advantage | Gap for us |
|---|---|---|
| sparse autoencoders; prompt-format sensitivity testing | exposes hidden reliability gaps — accuracy can shift **>50%** on prompt format alone | the reason TRACE-PD's LLM **decides nothing**: it only restates pre-computed facts, behind a validator with a template fallback |

## 4. ML risk + SHAP + LLM explanations

**Yeh Y.-C., Yang H.-Y., Chiu C.-T., Chao A., Chuang Y.-C., Chan W.-S.** *Enhancing large
language model clinical support information with machine learning risk and explainability:
a feasibility study.* Intensive Care Medicine Experimental 14(1):51, 2026.
<https://doi.org/10.1186/s40635-026-00900-w>

> Title in the deck: "Enhancing LLM Clinical Support with ML Risk and SHAP Explainability
> (ICU)". The content matches: XGBoost ICU mortality on MIMIC-IV + SHAP + GPT-4o.

| Technique | Advantage | Gap for us |
|---|---|---|
| XGBoost risk → SHAP attribution → GPT-4o bedside text | a working ML → SHAP → LLM pipeline | ICU mortality, not PD; **no confidence component**; SHAP explains one prediction, not *why confidence changed* between visits |

## 5. Conformal prediction for disease course in MS

**Sreenivasan A.P., Vaivade A., Noui Y., Emami Khoonsari P., Burman J., Spjuth O.,
Kultima K.** *Conformal prediction enables disease course prediction and allows
individualized diagnostic uncertainty in multiple sclerosis.* npj Digital Medicine 8(1):224,
2025. <https://doi.org/10.1038/s41746-025-01616-z>

| Technique | Advantage | Gap for us |
|---|---|---|
| conformal prediction on RRMS → SPMS classification | calibrated, patient-specific confidence; set size reflects certainty | multiple sclerosis, not PD; no stability or transition framing across visits |

## 6. PD motor subtypes from MRI radiomics

**Hui D., Wang X., Xie L., Chen F., Guo Y., Luo Y., He X.** *Automatic differentiation of
Parkinson's disease motor subtypes based on deep learning and radiomics.* Frontiers in
Neurology 16:1650985, 2025. <https://doi.org/10.3389/fneur.2025.1650985>

| Technique | Advantage | Gap for us |
|---|---|---|
| MRI radiomics from 8 brain nuclei, five classifiers, PPMI (135 patients: 92 TD, 43 PIGD) | targets **exactly** TD/PIGD | same-visit imaging only, not earlier prediction; no calibrated confidence; imaging not available at every visit |

## 7. Conformal bands for longitudinal biomarkers

**Tassopoulou V., Stamouli C., Shou H., Pappas G.J., Davatzikos C.**
*Uncertainty-Calibrated Prediction of Randomly-Timed Biomarker Trajectories with Conformal
Bands.* NeurIPS 2025. arXiv:2511.13911 — <https://arxiv.org/abs/2511.13911>

> Title in the deck: "Conformal Bands for Biomarker Trajectories in Alzheimer's/MCI". The
> published title doesn't name the disease; the application is AD progression.

| Technique | Advantage | Gap for us |
|---|---|---|
| group-conditional conformal bands over irregularly timed visits | handles irregular visit timing; flags high-risk patients earlier | Alzheimer's, not PD; continuous biomarker, not a discrete subtype label |

---

## Synthesis — the gap

| | Imaging-free | Calibrated confidence | Longitudinal | PD subtype | Explains *changes* |
|---|---|---|---|---|---|
| 1 CASCADE | ✅ | ✅ | ❌ | ❌ | ❌ |
| 2 7T MRI | ❌ | ❌ | ❌ | ✅ | ❌ |
| 3 LLM variability | — | — | — | — | — |
| 4 ICU ML+SHAP+LLM | ✅ | ❌ | ❌ | ❌ | ❌ |
| 5 MS conformal | — | ✅ | ❌ | ❌ | ❌ |
| 6 Radiomics | ❌ | ❌ | ❌ | ✅ | ❌ |
| 7 AD conformal bands | ❌ (MRI volumes) | ✅ | ✅ | ❌ | ❌ |
| **TRACE-PD (design)** | ✅ | ✅ | ✅ | ✅ | ✅ |

The gaps, in the order the deck presents them:

1. **Imaging-dependent classification** — the subtype-specific work (2, 6) needs MRI.
2. **No calibrated confidence** — subtype classifiers report accuracy or AUC, not a
   per-patient trust measure.
3. **Self-derived subtypes** — clustering approaches invent categories that need their own
   validation. This is exactly what sank TRACE-PD's Phase 1 (see README §3).
4. **Static, single-visit framing** — despite a third to half of patients being reclassified
   within 1–2 years.
5. **No explanation of uncertainty over time** — SHAP/LIME explain one prediction, not why
   confidence moved between visits.

> **Honesty note.** The last row of the synthesis is the *design*. Today only the imaging-free
> PD subtype classifier exists. Calibrated confidence, longitudinal tracking and change
> explanation are designed but not built (README §13).

---

## Paper reference list — status

`reports/paper/TRACE-PD_paper_draft_v7.docx` §7 has **15 references**. Two are complete
(Stebbins 2013, Goetz 2007). **13 still read "Author(s) not specified"**.

| Reference in draft | Status |
|---|---|
| Automatic differentiation … deep learning and radiomics | ✅ resolved — #6 above |
| 7T quantitative MRI … motor subtype stratification | ✅ resolved — #2 above |
| CASCADE … two-stage clinical decision support | ✅ resolved — #1 above |
| Conformal prediction … multiple sclerosis | ✅ resolved — #5 above |
| Conformal bands … Alzheimer's / MCI | ✅ resolved — #7 above |
| Enhancing LLM clinical support … SHAP … ICU | ✅ resolved — #4 above |
| Understanding clinical reasoning variability … LLMs | ✅ resolved — #3 above |
| Auto-classification of PD motor subtypes using ASL-MRI (PMC10670033) | ❌ not yet looked up |
| Multi-modality radiomics (T1 + DTI) … early-stage PD subtypes (PMC11377437) | ❌ not yet looked up |
| Analysis, identification and prediction of PD sub-types … ML (arXiv 2306.04748) | ❌ not yet looked up |
| Stability of MDS-UPDRS motor subtypes over three years … (Front Neurol 2021) | ❌ not yet looked up |
| Consistency and stability of motor subtype classifications in de novo PD (Front Neurosci 2021) | ❌ not yet looked up |
| Efficacy of rehabilitation across PD motor subtypes (PMC8284270) | ❌ not yet looked up |

The 7 resolved citations above **have not yet been written into the .docx**.

**Missing from the draft entirely:**

- **Jankovic J. et al. (1990).** *Variable expression of Parkinson's disease: a base-line
  analysis of the DATATOP cohort.* Neurology 40(10):1529–1534. The original TD/PIGD ratio,
  cited throughout but not listed.
- **Marek K. et al. (2011).** *The Parkinson Progression Marker Initiative (PPMI).* Progress
  in Neurobiology 95(4):629–635. The dataset itself.
- Conformal-prediction method references: Vovk, Gammerman & Shafer (2005); Romano, Sesia &
  Candès (2020, APS).
- Chen & Guestrin (2016), *XGBoost*, KDD.
