# FCX — Formula-Coordinate Explanations

**Our explainability framework for TRACE-PD.** It explains the three models' **predictions**
— the subtype classifier, the conformal sets and the transition-risk model — in the
coordinates of the published Stebbins formula. Because we hold the true exam scores, every
explanation is scored against ground truth.

| | |
|---|---|
| Code | `src/trace_pd/explain/` — `formula.py`, `shapley.py`, `implied_exam.py`, `straddle.py`, `pdn.py` |
| Evaluation | `src/trace_pd/evaluation/evaluate_fcx.py` → `reports/metrics/fcx_results.txt` |
| Figure | `reports/figures/fig_fcx_summary.png` |
| Tests | `tests/test_fcx.py` (7 invariants) |
| Run | `make fcx` |

Everything is our own implementation, including the Shapley estimator. There is no SHAP or
LIME dependency; SHAP-style attribution appears only as the **baseline** we compare against.
Black boxes here are sklearn gradient boosting. FCX only needs `predict` / `predict_proba`,
so it runs unchanged on XGBoost.

![FCX summary](../reports/figures/fig_fcx_summary.png)

---

## 0. The coordinate system

The label is a **known formula** over hidden exam items:

```
ℓ = log(T + ε) − log(P + ε)          T = mean of 11 tremor items, P = mean of 5 gait items
PIGD if ℓ ≤ log 0.90 · TD if ℓ ≥ log 1.15 · Indeterminate otherwise
```

With ε = 0.5 / 16 these zones reproduce **100%** of the stored labels. FCX expresses every
explanation in ℓ, split into a **tremor channel** (log T) and a **gait channel** (log P).
Those are the formula's own quantities and cutoffs.

---

## C1 — Implied-Exam Explanation (explains the subtype classifier)

**Method.**

1. Two concept heads estimate the scores the classifier *behaves as if* it had seen:
   ĥ_T(x) ≈ log T and ĥ_P(x) ≈ log P. The implied ratio is ℓ̂ = ĥ_T − ĥ_P.
2. **Fidelity calibration:** choose the cutoff c* on ℓ̂ so that the implied decision
   reproduces the classifier's decisions.
3. Feature attributions via our own interventional Shapley, on each head. Because ℓ̂ is a
   difference, they split **exactly** into channels: φ_j(ℓ̂) = φ_j(ĥ_T) − φ_j(ĥ_P). Units are
   log-ratio, i.e. distance moved toward or away from the cutoff.

**Identifiability.** The formula uses T and P only through their ratio, and only via three
zones. The classifier's output alone therefore can't identify T and P separately. The heads
are trained with concept supervision (the true scores), and fidelity to the classifier is
**measured**, not assumed.

**Output example:**

> *Predicted PIGD. The model behaves as if tremor ≈ 0.11 and gait ≈ 1.05 (implied ratio 0.13;
> cutoff 1.02). Main drivers: LEDD_TOTAL_MG via the **tremor** channel; NP2TURN via the
> tremor channel; NP2RISE via the gait channel.*

**Results:**

| Metric | Value | Reading |
|---|---|---|
| **Fidelity** — implied decision = classifier decision | **89.6%** | C1 is a faithful account of the classifier |
| Rank corr, classifier P(PIGD) vs implied −ℓ̂ | **0.923** | it tracks the classifier's confidence too |
| Balanced accuracy vs truth: classifier / implied exam | 0.686 / 0.688 | explaining this way costs nothing |
| **Alignment** — implied vs TRUE gait score | r = **0.57** | the model's gait belief is partly right |
| **Alignment** — implied vs TRUE tremor score | r = **0.25** | **the model is nearly blind to tremor** |
| Stability across background / seed | 0.879 (Shapley-on-classifier: 0.880) | as stable as the baseline |
| Error attribution: which channel is wrong on a misclassification | FCX 0.52 · baseline 0.45 · chance 0.50 | ❌ **no method manages it** — reported as a negative result |

**What C1 reveals that plain SHAP can't.**

- The classifier's gait belief tracks the truth (0.57); its tremor belief barely does (0.25).
  Questionnaire data carries almost no tremor information, so **the model's TD calls rest
  mostly on "no gait problems" rather than on seeing tremor.**
- The largest tremor-channel driver is **LEDD** (medication dose). The model has learned
  that medicated patients show less tremor — medication *suppressing* tremor on the exam.
  That's the same artefact as the 25.6% OFF/ON label disagreement (README §9.7). FCX makes it
  visible because the attribution lands in the tremor channel.
- The top gait-channel driver is **NP2RISE** (rising from a chair), an axial item. That's
  consistent with the proxy-leakage concern in README §9.3.

---

## C2 — Cutoff-Straddle Explanation (explains the conformal set)

**Method.**

1. Quantile heads give a half-width per channel: hw_T(x), hw_P(x).
2. The implied-ratio interval is ℓ̂ ± κ (hw_T + hw_P).
3. **Fidelity calibration:** κ is chosen on held-out calibration rows so that "the interval
   straddles the cutoff" reproduces the black-box conformal set's ambiguity.
4. **Blame:**

   ```
   d = |ℓ̂ − cutoff|
   resolves if tremor known   ⇔  d ≥ κ·hw_P
   resolves if gait known     ⇔  d ≥ κ·hw_T
   ```

   Blame goes to the channel whose removal resolves the straddle. If both do, the one with the
   larger half-width. If neither, "both".
5. **Verification:** substitute the TRUE score of the blamed channel and check whether the
   straddle really resolves.

**Output example:**

> *Set {TD, PIGD}: the implied-ratio interval [0.81, 3.18] straddles the cutoff. Responsible
> channel: tremor. Examining tremor would settle it.*

**Results:**

| Metric | Value |
|---|---|
| Black-box LAC coverage / ambiguous sets | 90.1% / 54.8% |
| **Fidelity** — straddle = ambiguous set | **80.8%**, Cohen's κ **0.61** (substantial) |
| Ambiguous sets explained by a straddle | **83.2%** |
| **Verified** — blamed *tremor*: revealing the true tremor score resolves it | **83%** (other channel: 59%) |
| **Verified** — blamed *gait*: revealing the true gait score resolves it | **78%** (other channel: 71%) |
| **vs baseline** on the same straddles: blamed channel really resolves it | **FCX 81%** · grouped Shapley on classifier 68% · random channel 73% |

**Reading:** FCX's "which part of the exam would settle this" is **verified correct 81% of the
time**, beating a SHAP-style grouping (68%) and a random pick (73%). Across all straddles,
revealing tremor resolves 80% and gait 59%. The model is uncertain mostly because it
**can't see tremor**, which matches C1.

> **Caveat.** With κ calibrated for fidelity, the channel intervals are the model's
> *effective* uncertainty, not valid prediction intervals for the true scores (they cover the
> true tremor score only 25% of the time). They explain the conformal set; they aren't a new
> guarantee.

---

## C3 — Proximity / Drift / Noise (explains the transition-risk model)

**Method.**

1. Heads estimate the smoothed underlying ratio η̂(x), its drift to the next visit μ̂(x), and
   visit-to-visit noise σ̂(x). Targets come from the **true** ℓ trajectory, smoothed
   leave-visit-out.
2. A closed-form first-passage surrogate gives the flip probability:

   ```
   p = E_η [ 1 − Σ_zone π_z(η, σ) · π_z(η + μ, σ) ]
   ```

3. It is calibrated to the black box with two parameters: logit g3 ≈ α + β · logit p.
4. **Exact** 3-player Shapley splits each predicted risk into **proximity + drift + noise**.

**Results:**

| Metric | Value |
|---|---|
| Black-box transition model AUROC | **0.527** (out-of-fold upstream features) |
| Fidelity: corr(logit g3, surrogate) / surrogate AUROC | 0.37 / 0.54 |
| **What the model's risk is made of** | **proximity 56% · drift 8% · noise 36%** |
| Noise-vs-drift flip identification (AUROC) | FCX 0.50 · grouped Shapley 0.49 |

**Reading — an important finding, not only an explanation.** Built leak-free, **the transition
model is near chance (AUROC 0.53)**. C3's decomposition shows why: **only 8%** of its predicted
risk comes from real drift. The rest is proximity and noise, which cheap features can barely
estimate.

Rutu's earlier 0.65 used upstream subtype probabilities computed **in-sample**
(`NEXT_STEPS_RUTU.md` §3). This suggests most of that signal was leakage. **C3 can't be
validated against a near-chance model.** It is implemented and tested, and it will become
meaningful once a transition target with real signal exists (e.g. sustained transitions,
`docs/flip_anatomy.md` §6).

---

## Summary

| Component | Explains | Validated? | Headline |
|---|---|---|---|
| **C1** Implied exam | subtype classifier | ✅ fidelity 89.6%, alignment measured | the model is nearly blind to tremor; LEDD acts through the tremor channel |
| **C2** Cutoff straddle | conformal sets | ✅ fidelity κ 0.61; blame verified 81% vs 68% SHAP-style baseline | tells the clinician *which part of the exam* would settle an ambiguous case |
| **C3** Proximity–drift–noise | transition risk | ⚠️ implemented; black box near chance | the transition model has almost no drift signal |

**Novelty positioning** (see `explainability_framework_proposal.md` §3):

- The closest prior work is post-hoc concept bottleneck models (learned head), a 2026
  glioblastoma CBM (trained, not post-hoc) and COCOCO (discrete logic).
- FCX differs by explaining an **unmodified black box** in the coordinates of a **fixed clinical
  formula**. It explains conformal ambiguity as a **cutoff straddle with verified channel
  blame**, and reports **fidelity, alignment and verification** separately.
- Novelty-agent confidence ~0.65. The strongest validated contribution is **C2**.

**Limitations:**

- Error attribution in C1 fails for every method tried.
- C2's intervals are fidelity-calibrated, not coverage-valid.
- C3 awaits a transition model with signal.
- Black boxes are sklearn gradient boosting in this run.
- The label still mixes exam medication states.
