# Clinical Need — Is the Full Motor Exam Actually Done?

Markdown version of `clinical_need_evidence_case.pdf` (10 Sep 2026), plus the discussion that
shaped how we frame the project.

---

## The question

The TD/PIGD label comes from a formula over **item-level** MDS-UPDRS Part II and Part III
scores. **In PPMI those items are collected at essentially every visit**, so inside PPMI the
label is directly computable and there's nothing to predict.

The project only has a clinical justification if the full standardised motor exam is **not**
routinely available outside research cohorts. This document tests that assumption.

---

## Finding 1 — the MDS-UPDRS is a research instrument

| Dimension | Finding |
|---|---|
| Length | 50 items, ~30 minutes in full |
| Part III | 33 scored items from 18 questions, with left/right/body-region breakdowns |
| Rater | MDS runs a formal **certificate programme**; the Part III teaching module alone is ~2 hours |
| Frequency in practice | "not always administered at every clinical visit"; typically every 6 or 12 months, mainly inside research cohorts |
| Telehealth | even fewer ratings done remotely — body parts and tasks are hard to assess on video |

The scale's own literature cites administration time and complexity as the reason it isn't
adopted in ordinary clinics. That's the stated motivation behind abbreviated and remote
versions of it.

## Finding 2 — India (national survey, n = 267 clinicians, 23 states / UTs)

Rajan et al., *J Mov Disord* 2025:

| Metric | Value |
|---|---|
| Initial diagnosis by a movement-disorder specialist | **12.6%** of practices |
| Follow-up by a movement-disorder specialist | 19.5% |
| Follow-up by a neurologist | 85.7% |
| Follow-up by internist-equivalent doctors | 48.9% |
| Movement-disorder specialist "rarely accessible locally" | **79.2%** |
| PD nurse "rarely accessible locally" | 94.4% |
| Respondents practising in urban areas | 92.1% (urban:rural doctor density 3.8 : 1) |
| Patients paying fully out-of-pocket (> 75% of caseload) | 55.4% of practices |

India has roughly **9.5% of the global PD population** (~576,000 people, GBD 2016). A 2015
analysis found no neurologist or neurosurgeon resident in regions covering **934.8 million
people**.

**Implication:** most Indian PD encounters are with clinicians who aren't movement-disorder
specialists and are unlikely to be MDS-UPDRS certified. A label that formally needs certified
Part III rating is, in practice, unavailable to most of these patients.

## Finding 3 — UK: "review" is not "standardised motor exam"

NICE **NG71** recommends reviewing the diagnosis every **6–12 months**. It specifies review
*frequency*, not that a full item-level MDS-UPDRS is scored at each review. The UK
Parkinson's Audit documents variable access to the wider team (e.g. 31% unable or unsure
whether they could access a physiotherapist, pre-pandemic).

Even in a well-resourced system with an explicit guideline, research-grade structured motor
data isn't a by-product of ordinary care.

## Finding 4 — the complication that reshapes the framing

The same Indian survey rules out the naive pitch *"replace the exam with imaging and
genetics"*:

| Investigation | Reported practice in India |
|---|---|
| DaTscan | **94.4%** rarely advise it |
| Genetic testing | **94.9%** rarely advise it |
| CT | 75.7% rarely advise it |
| MRI | 71.5% frequently recommend it |

In low-resource settings **the exam is the cheap input** and imaging/genetics are the
expensive ones. That is why TRACE-PD holds imaging and genetics **out** of X (README §6.3).

---

## The framing that survives

The MDS-UPDRS has two very different halves:

| | Who supplies it | Needs |
|---|---|---|
| **Part II** — motor aspects of daily living | the **patient** (questionnaire) | nothing — self-completed, on paper or remotely |
| **Part III** — motor examination | a **certified clinician** | trained rater, in person, ~30 min |

So the research question becomes:

> *Can the TD / PIGD / Indeterminate classification — defined by the full Stebbins formula —
> be recovered from patient-reported and low-cost inputs alone, without the specialist Part
> III exam? And with what confidence, per patient?*

### How this was tightened later

The evidence case originally proposed keeping the three Part II **formula** items
(`NP2TRMR`, `NP2WALK`, `NP2FREZ`) because they're patient-reported and cheap. **We rejected
that on 10 Sep 2026** (strict leakage policy, README §6.2): even 3 of the 16 formula inputs
is partial label leakage. The 10 Part II items that are **not** in the formula remain in X.

---

## The objection we had to face

> *"But in PPMI, doesn't every patient get the full motor exam at every visit?"*

**Yes.** PPMI does it by protocol. That cuts two ways:

- **It's why we can train at all.** Every visit has a ground-truth label, so we have 5,742
  labelled rows.
- **It weakens a "predict earlier" pitch.** Inside PPMI, the label is available from Visit 1.
  "Earlier" only means something in settings where the exam isn't done.

So the defensible claim isn't *"we predict the subtype earlier than the formula"*. It's:

> **In settings where the full exam isn't done, we estimate what it would say — and say how
> much to trust that estimate.**

The practical use is **triage**. If the model is confident, plan around that subtype. If it
isn't, that's the signal to arrange the full exam.

---

## Limits of this argument

- The India survey is **self-reported** practice, not audited chart data, and 92.1% of
  respondents were urban.
- **No source gives a direct figure** for "% of routine PD visits worldwide with a complete
  item-level MDS-UPDRS". The claim is assembled from adjacent evidence — burden,
  certification, specialist scarcity, guideline wording. That gap should be stated, not
  papered over.
- **PPMI isn't the target population**: early-stage patients from specialist centres in
  high-income settings. External validation is required before any claim about Indian or
  rural use.
- **Part II is patient-reported**, with its own biases (recall, literacy, language). Cheaper
  to collect isn't the same as equally reliable.

---

## Sources

- Rajan R., Kumar H., Radhakrishnan D.M., et al. *The Landscape of Parkinson's Disease
  Treatment in India: A National Cross-Sectional Survey of Clinical Practitioners.* J Mov
  Disord 2025;19(2):212–216. doi:10.14802/jmd.25201
- NICE Guideline NG71, *Parkinson's disease in adults.*
- UK Parkinson's Audit, Parkinson's UK.
- Parkinson's UK written evidence to Parliament (access / waiting times).
- *Abbreviated MDS-UPDRS for Remote Monitoring in PD Identified Using Exhaustive
  Computational Search* (PMC9197633).
- MDS-UPDRS Training / Certificate Program, International Parkinson and Movement Disorder
  Society.
- Ganapathy K. *Distribution of neurologists and neurosurgeons in India …* Neurol India
  2015;63:142–154.
- Item-level completeness figures computed from our own PPMI extract.
