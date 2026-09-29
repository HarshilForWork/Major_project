# Architecture (HLD)

![High-Level Design](../reports/figures/fig_HLD_system_architecture.png)

Three subsystems. **Training** (left) and **Inference** (right) are independent and never
call each other — they meet only at the shared **Data Layer** (centre). Training is offline
and batch, re-run per PPMI extract. Inference is online, per patient visit, and never
retrains.

---

## Training subsystem

| Stage | Responsibility |
|---|---|
| **Ingestion & Join** | Pulls 19 PPMI source tables from LONI IDA. 12 visit-level tables outer-joined on `PATNO + EVENT_ID`; 5 static per-patient tables left-joined on `PATNO`; 2 medication logs joined **by date interval**, not visit code. |
| **Preprocessing & Feature Governance** | Restricts to `COHORT == 1`, resolves structural missingness and duplicate visits, produces the 6,922 × 75 curated table. |
| **Label Engine** | Applies the Stebbins ratio to 11 tremor items and 5 PIGD items. TD if ≥ 1.15, PIGD if ≤ 0.90, Indeterminate between. A label is only assigned when all 16 items are present. |
| **Leakage Guard** | Splits every column into permitted X / label-defining Y / banned, writes that split to the Feature Registry, and asserts at runtime that none of the 16 Y-defining items reach a model. |
| **Training Matrix** | One row per patient-visit: permitted X plus subtype Y. |
| **Visit-Pair Builder** | Takes the same rows and pairs each visit with the patient's next visit, producing `LABEL_FLIPPED_NEXT` (4,818 pairs). Only the transition head needs this. |
| **Model Trainer + Calibrator** | Two heads — subtype (single-visit rows) and transition (paired rows). Publishes both to the Model Registry. |

### The fork

The subtype model trains on **single visits**. The transition model trains on **pairs**.
These are separate paths out of the Training Matrix — the subtype model does not pass
through the Visit-Pair Builder.

---

## Data layer

| Store | Contents |
|---|---|
| Raw Data Store | PPMI tables exactly as extracted |
| Curated Dataset | `ppmi_tdpigd_long.csv`, 6,922 × 75 |
| Feature Registry | the X / Y / banned contract — read by both training and inference |
| Model Registry | subtype model, transition model, and their calibrators |
| Patient History Store | per-patient visit record, written by inference, read by the tracker |

The Feature Registry is the mechanism that makes the leakage guarantee hold at serving
time as well as training time: *Feature Assembly* builds the inference feature vector from
the registry, so a banned column cannot enter even if it is present in the request.

---

## Inference subsystem

```
Clinician Workstation
  → Inference API gateway
  → Feature Assembly        (loads contract from Feature Registry)
  → Prediction · XGBoost    (loads subtype model)
  → Conformal Confidence    (probabilities → calibrated prediction set)
  → Longitudinal Tracker    (reads Patient History Store)
  → Transition Risk         (loads transition model)
  → Explanation Orchestrator (→ LLM Provider API)
  → Report Renderer         (writes visit back to Patient History Store)
```

### What each layer contributes

**Prediction** returns a probability vector. Gradient-boosted trees are not calibrated, so
that vector is a score, not a probability a clinician can act on.

**Conformal Confidence** converts it into a prediction *set* at a target coverage (90%).
The signal is the **size** of the set: `{TD}` is confident, `{TD, PIGD}` means the model
can rule out Indeterminate but not choose, `{TD, PIGD, Ind}` means no information. The
threshold `q̂` is an empirical quantile of nonconformity scores measured on a **held-out
calibration split**, never on training data.

**Longitudinal Tracker** is the only stateful component. It produces the label trajectory,
a confidence curve, and a stability index across a patient's visits.

**Transition Risk** answers a different question from conformal confidence:

| | Question | About |
|---|---|---|
| Conformal | How sure am I of the label *right now*? | the model's ignorance |
| Transition risk | Will the true label *change* by the next visit? | the patient actually changing |

Its features include the outputs of the three stages above — predicted label, top-2
probability margin, conformal set size, prior flip count — which is why it sits downstream
of them rather than beside them.

**Explanation Orchestrator** assembles a structured payload of already-computed facts,
runs delta attribution (the change in per-feature contribution between visit *t−1* and *t*,
valid only because the model is frozen), calls an LLM constrained to restating that
payload, then **validates** the output: every number must appear in the payload, every
feature name must be in the registry, no clinical recommendation. On failure it falls back
to a deterministic template, so the report always renders.

---

## Design invariants

1. **Training and inference never share memory or state** — only artifacts in the data layer.
2. **The Feature Registry is a machine-enforced contract**, not documentation.
3. **Inference never retrains.** Models are loaded from the registry; this is what makes
   delta attribution comparable across visits.
4. **The LLM decides nothing.** It renders facts computed upstream, behind a validator.
5. **A completed motor exam always wins.** If the full Part III is done, the Label Engine
   computes ground truth and it overrides the prediction in the report.

---

## Implementation status

Layers 1 and 2 (data layer, label engine, leakage guard, subtype trainer) exist in
`src/trace_pd/`. Conformal confidence, the longitudinal tracker, transition risk and the
explanation orchestrator are designed but not built. The diagram shows the intended
system, not the current one.
