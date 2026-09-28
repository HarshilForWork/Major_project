# Explanation Orchestrator — Design

**Status: designed, not built.**

The name is deliberate: it **orchestrates**, and the LLM only writes. Every number in the
output is computed before the LLM is called, and the LLM decides nothing.

---

## Pipeline

### 1. Evidence assembler

Collects structured facts from the upstream layers into a payload. **The LLM never sees the
raw patient record.**

```json
{
  "visit":      {"patno": "3001", "month": 24},
  "prediction": {"label": "PIGD", "margin_top2": 0.11},
  "conformal":  {"set": ["PIGD"], "size": 1, "coverage": 0.90,
                 "previous_set": ["TD", "PIGD"], "previous_size": 2},
  "trajectory": {"labels": ["TD", "TD", "TD/PIGD", "TD/PIGD", "PIGD"],
                 "n_flips": 1, "stability": 0.62},
  "transition_risk": {"p_flip_next": 0.14, "base_rate": 0.29},
  "delta_drivers": [
    {"feature": "NP2RISE",    "direction": "worsened", "delta_contrib": 0.19},
    {"feature": "NP2TURN",    "direction": "worsened", "delta_contrib": 0.12},
    {"feature": "MCATOT",     "direction": "stable",   "delta_contrib": -0.02}
  ],
  "allowed_terms": ["PIGD", "TD", "Indeterminate", "prediction set"]
}
```

*(Illustrative values.)*

### 2. Delta attribution — the novel part

Standard SHAP explains **one** prediction. Here we compute attributions for the predicted
class at visit *t* and at visit *t − 1*, and take the difference:

```
Δφᵢ = φᵢ(t) − φᵢ(t − 1)
```

Ranked by |Δφ|, this answers *"what changed since last time to move the confidence"*, not
*"what mattered today"*.

It's only valid because **the model is frozen**. It's loaded from the registry and never
retrained online, so the two attribution vectors share a background distribution and are
comparable.

### 3. Prompt builder

A fixed template. The payload is injected **as data**, with an instruction to restate only facts
present in it and to add no clinical advice.

### 4. LLM call

External provider — the dashed edge to *LLM Provider API* in the architecture diagram.

### 5. Validator

This is why the orchestrator is a subsystem and not one API call. It checks the returned text:

- every number appears in the payload
- every feature name is in the registry's permitted list
- no imperative clinical recommendation ("start levodopa", "refer for DBS")
- length and required sections are present

**Fails** → retry once → **fall back to a deterministic template.** The report always renders.
The LLM improves readability; it isn't a dependency.

---

## Example output

> Motor subtype is **PIGD**. The 90% prediction set contains PIGD alone, narrowed from {TD,
> PIGD} at the previous visit — the classification has become more certain. The change is
> driven mainly by worsening on rising from a chair and turning in bed; cognitive scores are
> unchanged. This patient has changed label once across five visits. Estimated probability of
> a further change by the next visit is 14%, below the cohort rate of 29%.
>
> *Computed from 27 permitted features. The 16 MDS-UPDRS items that define the subtype label
> were excluded from the model.*

What it does **not** say: no treatment recommendation, no invented mechanism, no number
missing from the payload. The last line is the **audit trail**. It lets a clinician reviewing
the system see that the leakage control held.

---

## Honest points

**The LLM is the most replaceable component.** A template produces most of that paragraph. The
LLM mainly phrases the *comparison* with the last visit more naturally. If a reviewer objects
to an LLM in a clinical path, the answer is the fallback template, not a defence of the LLM.
Paper #3 in the literature review — accuracy shifting > 50% on prompt format alone — is why
it's fenced this tightly.

**Delta attribution is the most novel claim, and the least validated.** Nobody has shown that
Δφ over consecutive visits faithfully explains a conformal set changing size. SHAP explains
the **probability**; the set size comes from the **nonconformity score**. They're related but
not the same quantity, and the paper should say so.
