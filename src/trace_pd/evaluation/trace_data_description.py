"""Data description for both cohorts (PPMI; ADNI with the primary model M_minus), computed from
the locked processed data, the data dictionaries and the saved TRACE models. Aggregates only.

Sources of truth: data/processed/*_dictionary.csv (buckets, pct_present), the PPMI column
descriptions file, preprocess.py / preprocess_adni.py (label rules, source columns),
trace_run.MODELS (model inputs), models/trace/*.json (saved models), configs/feature_groups.yaml
(feature groups: DRAFT).

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_data_description
Out:  results/data_description/DATA_DESCRIPTION.md
"""
import numpy as np
import pandas as pd
import yaml

from trace_pd import config
from trace_pd.evaluation import trace_run as tr

OUT = config.ROOT / "results" / "data_description"

# ADNI has no description file; these are read off preprocess_adni.py (source table.column).
ADNI_DESC = {
    "RID": "Patient identifier (key)", "VISCODE2": "ADNI visit code (key)",
    "VISIT_MONTH": "Months since baseline (visit time)", "EXAMDATE": "Exam date",
    "LABEL": "Clinical diagnosis CN / MCI / Dementia (DXSUM.DIAGNOSIS)",
    "AGE_AT_VISIT": "Age at visit (derived from PTDEMOG birth date and exam date)",
    "SEX": "Sex (PTDEMOG.PTGENDER)", "HANDED": "Handedness (PTDEMOG.PTHAND)",
    "EDUCATION_YEARS": "Years of education (PTDEMOG.PTEDUCAT)",
    "MMSE_TOTAL": "Mini-Mental State Exam total (MMSE.MMSCORE)", "MOCA_TOTAL": "MoCA total (MOCA.MOCA)",
    "GDS_TOTAL": "Geriatric Depression Scale total (GDSCALE.GDTOTAL)",
    "FAQ_TOTAL": "Functional Activities Questionnaire total (FAQ.FAQTOTAL)",
    "NPI_TOTAL": "Neuropsychiatric Inventory-Q total (NPIQ.NPISCORE)",
    "ECOG_PATIENT_TOTAL": "Everyday Cognition, self-report total (ECOGPT)",
    "ECOG_PARTNER_TOTAL": "Everyday Cognition, study-partner total (ECOGSP)",
    "LOGMEM_IMMEDIATE": "Logical Memory immediate recall (NEUROBAT.LIMMTOTAL)",
    "LOGMEM_DELAYED": "Logical Memory delayed recall (NEUROBAT.LDELTOTAL)",
    "BOSTON_NAMING": "Boston Naming Test (NEUROBAT.BNTTOTAL)",
    "CATEGORY_FLUENCY_ANIMALS": "Category fluency, animals (NEUROBAT.CATANIMSC)",
    "TRAILS_A_SEC": "Trail Making A, seconds (NEUROBAT.TRAASCOR)",
    "TRAILS_B_SEC": "Trail Making B, seconds (NEUROBAT.TRABSCOR)",
    "DIGIT_SPAN": "Digit span score (NEUROBAT.DIGITSCOR)",
    "WEIGHT_KG": "Weight, kg (VITALS; unit-converted, -1 sentinel -> missing)",
    "HEIGHT_CM": "Height, cm (VITALS; unit-converted; carried forward/back within patient)",
    "BP_SYSTOLIC": "Systolic BP (VITALS.VSBPSYS)", "BP_DIASTOLIC": "Diastolic BP (VITALS.VSBPDIA)",
    "PULSE": "Pulse (VITALS.VSPULSE)",
    "CDGLOBAL": "CDR global score", "CDRSB": "CDR sum of boxes",
}
GROUP_LABEL = {"demographic_static": "demographic / static", "time": "time", "medication": "medication",
               "motor_daily_living": "motor daily living (MDS-UPDRS II)", "non_motor_questionnaires": "non-motor questionnaires",
               "cognition": "cognition", "vitals_body": "vitals / body", "mood_behaviour": "mood / behaviour",
               "cognitive_tests": "cognitive tests", "function_daily_living": "function / daily living"}


def pct_missing(df, c):
    return 100 * df[c].isna().mean()


def load_raw(cohort):
    """Processed table exactly as on disk (missingness of text columns preserved)."""
    p = config.LONG_TABLE if cohort == "ppmi" else config.ADNI_LONG_TABLE
    return pd.read_csv(p, low_memory=False)


def cohort_section(cohort, model, slide_rows, issues):
    raw = load_raw(cohort)
    dd = pd.read_csv(config.DICTIONARY if cohort == "ppmi" else config.ADNI_DICTIONARY)
    desc = (pd.read_csv(config.ROOT / "data/processed/ppmi_tdpigd_column_descriptions.csv").set_index("column")["description"].to_dict()
            if cohort == "ppmi" else ADNI_DESC)
    groups = yaml.safe_load(open(config.ROOT / "configs/feature_groups.yaml", encoding="utf-8"))[model]
    grp_of = {f: g for g, fs in groups.items() for f in fs}
    c = tr.COHORT[cohort]
    dev, test, F_all = tr.load_cohort(cohort)
    feats = tr.features(model)
    b = tr.get_model(model)
    idc = c["id"]

    # --- checks against the dictionary
    bucket = dd.set_index("column")["bucket"].to_dict()
    banned = sorted(dd.loc[dd.bucket == "BANNED_LABEL_DEFINING", "column"])
    cheap = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
    not_cheap = [f for f in feats if bucket.get(f) != "CHEAP_FEATURE"]
    if not_cheap:
        issues.append(f"{cohort}: model inputs not in CHEAP_FEATURE: {not_cheap}")
    overlap = sorted(set(feats) & set(banned))
    n_model_feats = b.num_features()
    if n_model_feats != len(feats):
        issues.append(f"{cohort}: saved model has {n_model_feats} inputs but the code lists {len(feats)}")
    model_excluded = sorted(set(cheap) - set(feats))
    for col in dd.column:                                          # dictionary pct_present vs data
        if col in raw:
            d = abs((100 - dd.set_index("column").loc[col, "pct_present"]) - pct_missing(raw, col))
            if d > 0.1:
                issues.append(f"{cohort}: {col} % missing differs from dictionary by {d:.2f} points")
        else:
            issues.append(f"{cohort}: dictionary column {col} not in the data file")
    extra = sorted(set(raw.columns) - set(dd.column))
    if extra:
        issues.append(f"{cohort}: data columns missing from the dictionary: {extra}")

    # --- model input / output
    tr_rows = dev[dev.LABEL.isin([c["neg"], c["pos"]])]
    te_rows = test[test.LABEL.isin([c["neg"], c["pos"]])]
    n_locked = len(pd.read_csv(config.INTERIM / ("locked_test_patients.csv" if cohort == "ppmi" else "locked_test_patients_adni.csv")))
    lab_counts = raw["LABEL"].value_counts(dropna=False)
    ba, _ = tr.locked_test_ba(model, b)
    n_analysis = len(tr.analysis_patients(cohort))
    L = []
    title = "PPMI (Parkinson's)" if cohort == "ppmi" else "ADNI (Alzheimer's), primary model M_minus"
    L += [f"## {title}", "",
          f"Cohort file: {raw[idc].nunique():,} patients, {len(raw):,} visits. "
          f"Visit-level label counts: " + ", ".join(f"{k if isinstance(k, str) else 'no label'} {v:,}" for k, v in lab_counts.items()) + ".", "",
          "### 1. Model input / output", ""]
    if cohort == "ppmi":
        L += ["- **Target (Y):** `LABEL`, the motor subtype at a visit (Stebbins et al. 2013). TREMOR_SCORE = mean of 11 tremor "
              "items, PIGD_SCORE = mean of 5 gait/balance items (all items required). Ratio >= 1.15 -> TD, <= 0.90 -> PIGD, "
              "otherwise Indeterminate; if PIGD_SCORE = 0, TD when tremor > 0, else Indeterminate.",
              "- **Model task:** binary TD vs PIGD; Indeterminate and unlabelled visits are not used for training. "
              "**PIGD is coded 1**, TD 0. Output: raw log-odds of PIGD; predicted PIGD when > 0 (probability 0.5)."]
    else:
        L += ["- **Target (Y):** `LABEL`, the clinical diagnosis at a visit, from DXSUM.DIAGNOSIS (CN / MCI / Dementia).",
              "- **Model task:** binary CN vs Dementia; MCI and unlabelled visits are not used for training. "
              "**Dementia is coded 1**, CN 0. Output: raw log-odds of dementia; predicted Dementia when > 0."]
    L += [f"- **Inputs (X):** {len(feats)} features (saved model expects {n_model_feats}); listed in section 3.",
          f"- **Training data:** development split only (locked test patients excluded): {len(tr_rows):,} labelled visits from "
          f"{tr_rows[idc].nunique():,} patients ({c['neg']} {int((tr_rows.LABEL == c['neg']).sum()):,} / {c['pos']} "
          f"{int((tr_rows.LABEL == c['pos']).sum()):,}); balanced sample weights; XGBoost, 300 trees, depth 3, seed 42.",
          f"- **Locked test set:** {n_locked:,} patients ({test[idc].nunique():,} with visits in the file); "
          f"{len(te_rows):,} labelled {c['neg']}/{c['pos']} test visits; balanced accuracy {ba:.3f}. "
          f"TRACE analyses use the {n_analysis} test patients with >= 3 visits.", ""]

    # --- slide table
    L += ["### 2. Slide table", "", "| Column | Role | Description | % missing |", "|---|---|---|---|"]
    for col, role in slide_rows:
        cols = col if isinstance(col, tuple) else (col,)
        name = " / ".join(f"`{x}`" for x in cols)
        d = "; ".join(desc.get(x, "") for x in cols)
        miss = " / ".join(f"{pct_missing(raw, x):.1f}%" for x in cols)
        L.append(f"| {name} | {role} | {d} | {miss} |")
    L.append("")

    # --- feature list
    L += ["### 3. Model input features", "",
          "Groups are the DRAFT in `configs/feature_groups.yaml` (not yet confirmed). % missing = share of all visits in the cohort file "
          "with no value; training-visit missingness in the last column.", "",
          "| # | Feature | Group | Description | % missing (all visits) | % missing (training visits) |", "|---|---|---|---|---|---|"]
    order = sorted(feats, key=lambda f: (list(groups).index(grp_of[f]), f))
    for i, f in enumerate(order, 1):
        L.append(f"| {i} | `{f}` | {GROUP_LABEL.get(grp_of[f], grp_of[f])} | {desc.get(f, '')} | {pct_missing(raw, f):.1f}% | "
                 f"{100 * tr_rows[f].isna().mean():.1f}% |")
    L.append("")

    # --- banned
    L += ["### 4. Banned and excluded columns", "",
          f"**Dictionary bucket BANNED_LABEL_DEFINING ({len(banned)}):** " + ", ".join(f"`{x}`" for x in banned) + ".", ""]
    if model_excluded:
        L.append(f"**Excluded from this model by the model definition (dictionary bucket CHEAP_FEATURE, dropped in "
                 f"`trace_run.MODELS['{model}']`):** " + ", ".join(f"`{x}`" for x in model_excluded) + ".")
        L.append("")
    others = dd.loc[~dd.bucket.isin(["CHEAP_FEATURE", "BANNED_LABEL_DEFINING"])].groupby("bucket")["column"].apply(list)
    L.append("**Other non-input buckets:** " + "; ".join(f"{k} ({len(v)}): " + ", ".join(v) for k, v in others.items()) + ".")
    L += ["", f"**Check:** banned columns among model inputs = {overlap if overlap else 'none'}; every input is in CHEAP_FEATURE = "
          f"{'yes' if not not_cheap else 'NO'}; saved model input count = {n_model_feats} = code list {len(feats)}.", ""]
    return "\n".join(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    issues = []
    ppmi_rows = [("LABEL", "Target"), ("PATNO", "Key (patient)"), ("VISIT_MONTH", "Key/Time"),
                 ("NP2RISE", "Predictor: motor daily living"), ("NP1PTOT", "Predictor: non-motor questionnaire"),
                 ("LEDD_TOTAL_MG", "Predictor: medication"), ("MCATOT", "Predictor: cognition"),
                 ("YRS_SINCE_DIAGNOSIS", "Predictor: time"), (("SEX", "HANDED"), "Predictor: demographic"),
                 ("NP3RTARU", "Banned: one of the 11 tremor items that define Y"),
                 ("NP2WALK", "Banned: one of the 5 PIGD items that define Y"),
                 ("NP3TOT", "Banned: Part III total, contains the label items (likewise NP2PTOT, NHY)")]
    adni_rows = [("LABEL", "Target"), ("RID", "Key (patient)"), ("VISIT_MONTH", "Key/Time"),
                 ("LOGMEM_DELAYED", "Predictor: cognitive test"),
                 ("ECOG_PARTNER_TOTAL", "Predictor: function"), ("NPI_TOTAL", "Predictor: mood/behaviour"),
                 ("HEIGHT_CM", "Predictor: vitals/body"), ("AGE_AT_VISIT", "Predictor: time"),
                 (("SEX", "HANDED"), "Predictor: demographic"),
                 ("CDGLOBAL", "Banned: CDR exam, which the diagnosis is informed by (CDR 0 -> CN 89.7% of the time)"),
                 ("MMSE_TOTAL", "Excluded in M_minus: an ADNI diagnostic criterion (MMSE ranges gate CN / dementia)"),
                 ("FAQ_TOTAL", "Excluded in M_minus: functional impairment, part of the dementia diagnosis")]
    body = [cohort_section("ppmi", "ppmi", ppmi_rows, issues), cohort_section("adni", "adni_minus", adni_rows, issues)]
    head = ["# Data description: PPMI and ADNI (TRACE)", "",
            "Computed from the locked processed data, the data dictionaries and the saved models by "
            "`src/trace_pd/evaluation/trace_data_description.py`. Aggregates only; no patient identifiers.", ""]
    notes = ["## Things that do not match the dictionaries or the project docs", ""]
    notes += [f"- {x}" for x in issues] if issues else ["- Automated checks: no differences between dictionary % present and the data, "
                                                         "no undocumented columns, every model input is in CHEAP_FEATURE, no banned column is an input."]
    notes += [
        "- **ADNI MMSE_TOTAL and FAQ_TOTAL are not in the dictionary's banned bucket.** They are CHEAP_FEATURE in "
        "`adni_dx_dictionary.csv`; they are excluded only by the M_minus model definition in code. The dictionary's only "
        "BANNED_LABEL_DEFINING columns are the 8 CDR columns.",
        "- **PPMI GENETIC_COHORT** is described in CLAUDE.md as banned (a recruitment flag that alone scored 0.629), but the "
        "dictionary puts it in the ADMIN bucket, not BANNED_LABEL_DEFINING. It is not a model input either way.",
        "- **ADNI HEIGHT_CM is carried forward/back within each patient** during preprocessing (preprocess_adni.py, step 7b). "
        "This is a small exception to the 'missing stays missing' rule and should be disclosed.",
        "- **ADNI has no column-description file** (PPMI has one); ADNI descriptions above were taken from preprocess_adni.py.",
        "- **Feature groups are a draft** (`configs/feature_groups.yaml`, not yet confirmed).",
        "- **Older slides say 5,742 labelled PPMI rows;** the current data has 5,638 (30 Sep 2026 preprocessing fix).",
        ""]
    (OUT / "DATA_DESCRIPTION.md").write_text("\n".join(head + body + notes), encoding="utf-8")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
