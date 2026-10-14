"""TRACE Stage 10: assemble results/REPORT.md from the per-stage outputs (aggregates only).

Writes results/stage6/STAGE6.md, results/stage8/STAGE8.md, results/stage9/STAGE9.md, appends the
never-changing-credit table to the Stage 7 section, and concatenates everything with
results/REPORT_checklist_deviations.md (hand-written checklist, deviations, open questions).

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_report
"""
import json

from trace_pd import config

R = config.ROOT / "results"
KS = ["1", "2", "3", "4", "5"]
METHODS = ["trace", "direct", "chain", "fixed_first", "fixed_last", "population", "path_dep", "ar1"]


def f3(b):
    return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"


def stage6_md():
    L = ["# Stage 6: faithfulness", "",
         "Revert the top-k features (by |credit|) of the later visit to the earlier visit's real values; "
         "share of the score change recovered. Patient means, 95% patient-bootstrap CI. eps was OPEN: main = 0.05 "
         "(pairs with |score change| < eps excluded), sensitivity 0.01 / 0.1 / 0.25. Direct Baseline Shapley "
         "wins on its own pair by construction (upper reference); TRACE's claim is consistency across all pairs.", "",
         "Figure: `results/figures/fig_faithfulness.png`.", ""]
    for n in ["ppmi", "adni_minus"]:
        r = json.loads((R / "stage6" / f"faithfulness_{n}.json").read_text())
        e = r["eps_0.05"]
        L += [f"## {n} ({r['n_patients']} patients; eps 0.05: {e['trace']['n_pairs']} pairs kept, "
              f"{e['trace']['n_excluded']} excluded; AR(1): {e['ar1']['n_pairs']} pairs, s >= 1 only)", "",
              "| Method | k=1 | k=2 | k=3 | k=4 | k=5 |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            L.append(f"| {m} | " + " | ".join(f3(e[m]["recovered"][k]) for k in KS) + " |")
        L += ["", "Paired TRACE minus method (> 0 = TRACE recovers more):", "",
              "| Method | k=1 | k=3 | k=5 |", "|---|---|---|---|"]
        for m, d in e["trace_minus_method"].items():
            L.append(f"| {m} | {f3(d['1'])} | {f3(d['3'])} | {f3(d['5'])} |")
        L += ["", "eps sensitivity (TRACE recovered at k = 1 / 3 / 5; best other method at k = 3):", ""]
        for eps in ["eps_0.01", "eps_0.05", "eps_0.1", "eps_0.25"]:
            x = r[eps]
            best = max(((m, x[m]["recovered"]["3"]["mean"]) for m in METHODS if m != "trace"), key=lambda t: t[1])
            L.append(f"- {eps}: TRACE {x['trace']['recovered']['1']['mean']:.3f} / {x['trace']['recovered']['3']['mean']:.3f} / "
                     f"{x['trace']['recovered']['5']['mean']:.3f} ({x['trace']['n_pairs']} pairs); best other at k=3: "
                     f"{best[0]} {best[1]:.3f}")
        L.append("")
    (R / "stage6" / "STAGE6.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def stage8_md():
    L = ["# Stage 8: sanity check (patient-level shuffled labels)", "",
         "Retrained on the same development visits with each patient's baseline label permuted across patients "
         "(real visits, shuffled labels; not synthetic data). TRACE recomputed for the same locked-test patients.", "",
         "| Model | Shuffled-label model: locked-test BA | Spearman of per-feature mean abs(u), real vs shuffled | "
         "Per-visit Spearman of tallies (mean) | Total mean abs(u): real / shuffled |", "|---|---|---|---|---|"]
    for n in ["ppmi", "adni_minus"]:
        r = json.loads((R / "stage8" / f"sanity_{n}.json").read_text())
        L.append(f"| {n} | {r['shuffled_test_ba']:.3f} | {r['spearman_mean_abs_u']:.2f} | {r['per_visit_spearman_mean']:.2f} | "
                 f"{r['mean_abs_u_total_real']:.2f} / {r['mean_abs_u_total_shuffled']:.2f} |")
    L += ["", "Reading: at the level of individual visits the explanations fall apart (per-visit agreement about 0). "
          "The per-feature importance ranking keeps a weak-to-moderate correlation, because features that change more "
          "often or more strongly can receive credit even from a model with no real signal.", ""]
    (R / "stage8" / "STAGE8.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def stage9_md():
    L = ["# Stage 9: descriptive clinical analyses (not causal)", "",
         "Feature groups: `configs/feature_groups.yaml`, **DRAFT, not yet confirmed**. TRACE explains what the "
         "model reacts to, not disease mechanisms. Medication can look important because sicker patients get more "
         "medication.", ""]
    for n in ["ppmi", "adni_minus"]:
        r = json.loads((R / "stage9" / f"clinical_{n}.json").read_text())
        fl, m, w = r["flips"], r["measurement"], r["within_between"]
        L += [f"## {n}", "",
              f"**Model-predicted flips** (score crosses the decision threshold between consecutive visits): "
              f"{fl['n_model_flips']} ({fl['n_up']} towards the positive class, {fl['n_down']} away). "
              f"Consecutive pairs with both visits labelled: {fl['label_pairs_both_labelled']}; true-label flips: "
              f"{fl['label_flips']}; of these also model flips: {fl['label_flips_also_model_flips']}.", "",
              "| Group | Mean share of a flip's TRACE credit | Mean signed credit, flips up | flips down |", "|---|---|---|---|"]
        for g, v in sorted(fl["mean_share_by_group"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {g} | {100 * v:.1f}% | {fl['mean_signed_credit_up_by_group'].get(g, float('nan')):+.2f} | "
                     f"{fl['mean_signed_credit_down_by_group'].get(g, float('nan')):+.2f} |")
        L += ["", "Top features by mean share of flip credit: " + "; ".join(
            f"{k} {100 * v:.1f}%" for k, v in list(fl["mean_share_by_feature"].items())[:8]), "",
              f"**Measurement accounting:** {100 * m['share_abs_u_from_missingness_differences']:.1f}% of the total "
              f"abs(TRACE tally) comes from comparisons where the feature's recorded/missing status differs (indicative; "
              f"not sign-separated)."]
        if m["by_phase"]:
            L.append("By ADNI study phase: " + "; ".join(f"{k} {100 * v['share']:.1f}% ({v['visits']} visits)"
                                                         for k, v in m["by_phase"].items()) + ".")
        L += ["", f"**Within vs between patient** (population SHAP = TRACE within-patient part + between-patient part b). "
              f"Identity check sum_j b_j(t) = patient mean score - background mean score at every visit: max error "
              f"{w['identity_max_error']:.1e} (tolerance {w['identity_tolerance']:.0e}): "
              f"**{'PASS' if w['identity_pass'] else 'FAIL'}**. Within-patient share of total credit: "
              f"{100 * w['share_within']:.1f}%.", "",
              "| Group | mean abs(within u) | mean abs(between b) |", "|---|---|---|"]
        for g, v in w["by_group"].items():
            L.append(f"| {g} | {v['mean_abs_within_u']:.3f} | {v['mean_abs_between_b']:.3f} |")
        L += ["", "| Feature (top 10) | mean abs(within u) | mean abs(between b) | within-patient SD of b |", "|---|---|---|---|"]
        for k, v in list(w["by_feature"].items())[:10]:
            L.append(f"| {k} | {v['mean_abs_within_u']:.3f} | {v['mean_abs_between_b']:.3f} | {v['within_patient_sd_of_b']:.3f} |")
        L.append("")
    (R / "stage9" / "STAGE9.md").write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def never_md():
    d = json.loads((R / "stage7" / "never_changing_credit.json").read_text())
    L = ["## Credit given to features that never change for the patient", "",
         "| Model | " + " | ".join(["trace", "chain", "direct", "fixed_first", "fixed_last", "population", "path_dep"]) + " |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in d.items():
        L.append(f"| {n} | " + " | ".join(
            f"{100 * r[m]['share']['mean']:.2f}% (max {r[m]['max_abs']:.2f})" for m in
            ["trace", "chain", "direct", "fixed_first", "fixed_last", "population", "path_dep"]) + " |")
    return "\n".join(L) + "\n"


def main():
    parts = ["# TRACE: full report (brief Stage 10)", "",
             "Aggregates only; no patient identifiers. Plain-language summary: `results/SUMMARY_FOR_GUIDE.md`. "
             "Log: `results/PROGRESS.md`.", ""]
    for p in [R / "gate" / "GATE.md", R / "stage4" / "STAGE4.md", R / "stage5" / "STAGE5.md"]:
        parts.append(p.read_text(encoding="utf-8"))
    parts.append(stage6_md())
    parts.append((R / "stage7" / "STAGE7.md").read_text(encoding="utf-8"))
    parts.append(never_md())
    parts.append(stage8_md())
    parts.append(stage9_md())
    tail = R / "REPORT_checklist_deviations.md"
    if tail.exists():
        parts.append(tail.read_text(encoding="utf-8"))
    (R / "REPORT.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("results/REPORT.md written")


if __name__ == "__main__":
    main()
