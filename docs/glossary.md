# Glossary

For readers from either side — clinicians who don't know the ML terms, engineers who don't
know the clinical ones.

## Clinical

| Term | Meaning |
|---|---|
| **PD** | Parkinson's disease — progressive neurodegeneration from loss of dopamine-producing cells; affects movement, and also cognition, mood, sleep and autonomic function |
| **Motor subtype** | the pattern of motor symptoms that dominates in a given patient |
| **TD** | tremor-dominant subtype — shaking dominates; typically slower decline, better treatment response |
| **PIGD** | postural instability / gait difficulty — balance, walking and freezing problems dominate; faster decline, more falls, weaker response to levodopa and DBS |
| **Indeterminate** | neither dominates — the tremor/PIGD ratio falls between 0.90 and 1.15 |
| **MDS-UPDRS** | Movement Disorder Society Unified Parkinson's Disease Rating Scale — the standard PD rating scale, 50 items in four parts |
| **Part I** | non-motor experiences of daily living (mood, cognition, sleep, pain…) |
| **Part II** | motor experiences of daily living — **reported by the patient** (speech, eating, dressing, walking…) |
| **Part III** | motor examination — **scored by a trained clinician** (tremor, rigidity, gait, balance…); needs certification |
| **Part IV** | motor complications of treatment (not used here) |
| **Stebbins ratio** | mean of 11 tremor items ÷ mean of 5 PIGD items; ≥ 1.15 → TD, ≤ 0.90 → PIGD (Stebbins et al. 2013, adapting Jankovic et al. 1990) |
| **Hoehn & Yahr (NHY)** | 0–5 disease stage; stage ≥ 3 is *defined* by postural instability |
| **Axial symptoms** | trunk-level function — rising from a chair, turning in bed, posture, gait |
| **LEDD** | levodopa-equivalent daily dose — total dopaminergic medication, converted to a common levodopa scale |
| **ON / OFF state** | assessed with medication working (ON) or worn off (OFF); tremor is suppressed when ON |
| **De novo / drug-naive** | newly diagnosed and not yet on PD medication — PPMI's entry criterion |
| **DaTscan** | SPECT imaging of the dopamine transporter; low striatal binding supports PD |
| **SBR** | striatal binding ratio — the quantitative DaTscan measure |
| **LRRK2 / GBA / SNCA** | genes whose variants raise PD risk; PPMI's genetic arm is enriched for them |
| **MoCA** | Montreal Cognitive Assessment — a 30-point cognitive screen |
| **SCOPA-AUT** | autonomic symptom scale (bladder, bowel, blood pressure, sweating…) |
| **GDS-15** | Geriatric Depression Scale, short form |
| **STAI** | State-Trait Anxiety Inventory |
| **ESS** | Epworth Sleepiness Scale — daytime sleepiness |
| **RBDSQ** | REM Sleep Behaviour Disorder Screening Questionnaire — acting out dreams, an early PD marker |
| **PPMI** | Parkinson's Progression Markers Initiative — the longitudinal cohort study this data comes from |
| **LONI IDA** | the archive PPMI data is distributed through |
| **DUA** | Data Use Agreement — the terms PPMI data is released under; no redistribution |

## Data / ML

| Term | Meaning |
|---|---|
| **`PATNO`** | PPMI patient ID |
| **`EVENT_ID`** | PPMI visit code — `BL` baseline, `V01` = 3 months, `V04` = 12 months… |
| **Long format** | one row per patient-visit (6,922 rows) rather than one row per patient |
| **X / Y** | model inputs / the target the model predicts |
| **Label leakage** | the model sees information that directly encodes the answer, so it scores well without learning anything useful |
| **Proxy leakage** | an input isn't part of the label formula but measures nearly the same thing |
| **Feature registry / bucket** | the rule-based assignment of every column to a role — feature, banned, held out, target, key, admin |
| **Grouped CV** | cross-validation where all rows of a patient stay in the same fold, so no patient is in both train and test |
| **`StratifiedGroupKFold`** | grouped CV that also keeps class proportions balanced across folds |
| **Balanced accuracy** | mean of per-class recall; chance = 1 / number of classes. Unlike accuracy, can't be gamed by always predicting the majority class |
| **Macro-F1** | F1 averaged equally across classes |
| **Ablation** | removing feature groups one at a time to see which carry the signal |
| **Positive control** | a deliberate experiment that *should* succeed — here, feeding the banned items back in — to prove the pipeline works |
| **XGBoost / HGB** | gradient-boosted decision-tree libraries; both handle missing values natively |
| **Conformal prediction** | turns model scores into a **prediction set** guaranteed to contain the true label at a chosen rate (e.g. 90%) |
| **Coverage** | how often the true label is inside the set |
| **Marginal vs conditional coverage** | the guarantee averaged over everyone vs holding within each subgroup / class |
| **Calibration split** | held-out data the frozen model scores to set the conformal threshold — never training data |
| **Nonconformity score** | how "strange" the true label looked to the model; the conformal threshold is a quantile of these |
| **`q̂`** | the conformal threshold |
| **APS / LAC** | two nonconformity scores — APS (cumulative, adaptive set sizes) and LAC (1 − probability; a plain cutoff) |
| **Singleton rate** | fraction of patients whose prediction set has exactly one label — a confidence measure |
| **Exchangeability** | the assumption conformal needs — rows are interchangeable; violated by repeated visits of one patient |
| **Mondrian conformal** | class-conditional conformal — a separate threshold per class |
| **CV+ / cross-conformal** | conformal that rotates the calibration role across CV folds to use data efficiently |
| **Transition risk** | the probability the subtype label changes by the next visit |
| **SHAP (φ)** | per-feature contribution to one prediction |
| **Delta attribution (Δφ)** | the change in per-feature contribution between two visits — what moved the confidence |
| **HLD / LLD** | high-level design (components and data flow) vs low-level design (deployment, protocols, latency) |
