"""
01_preprocess_tdpigd.py
=======================
Preprocessing for the TD / PIGD / Indeterminate motor-subtype project
(renewed approach, supersedes the earlier K-Means slow/moderate/fast pipeline).

WHAT THIS SCRIPT PRODUCES
-------------------------
  ppmi_tdpigd_long.csv      One row per patient per scheduled visit. Contains
                            the computed TD/PIGD/Indeterminate label, the
                            transition (label-flip) target, and every candidate
                            predictor. LONG format on purpose -- the per-round
                            wide tables (Round 1 = BL only, Round 2 = BL+6mo,
                            etc.) are built from this file downstream, not here.
  ppmi_tdpigd_dictionary.csv  Machine-readable data dictionary: every column,
                            its role, its feature bucket, and its missingness.
  ppmi_tdpigd_runlog.txt    Full provenance log of this run (row counts at each
                            filtering step, label distribution, flip rates).

KEY DESIGN DECISIONS (all deliberate -- see the methods doc for rationale)
-------------------------------------------------------------------------
1. LABEL = published formula, not clustering.
   TD/PIGD/Indeterminate is computed per patient-visit using the Stebbins et al.
   (2013, Movement Disorders) MDS-UPDRS formulation of the Jankovic (1990)
   tremor/PIGD ratio. Nothing is clustered or invented.

2. PD COHORT = COHORT == 1 (n=439), not APPRDX == 1 (n=242).
   COHORT==1 is the PPMI Parkinson's disease cohort. It comprises APPRDX==1
   (sporadic PD, n=242) plus APPRDX==5 (genetic-cohort PD, n=197). Both are PD.
   A GENETIC_COHORT flag is retained so the genetic subgroup can be adjusted
   for, or excluded, in any downstream analysis.

3. NO minimum follow-up filter.
   The previous pipeline dropped patients with <3 years of follow-up because the
   old label was a progression *rate* needing a long trajectory. The TD/PIGD
   label is computed independently at each visit, so that filter is obsolete and
   has been removed. This is the main reason the usable sample roughly doubles.

4. NO blanket median imputation.
   The previous pipeline median-imputed every numeric column, which flattened
   DaTscan striatal binding ratio to a constant 0.68 across all subtypes and
   destroyed a real signal. Missing values are therefore LEFT AS NaN here.
   XGBoost handles NaN natively via default-direction learning; any model that
   cannot should impute inside its own cross-validation folds, never here
   (imputing before splitting leaks test-fold information into training).

5. LEAKAGE CONTROL is explicit and machine-enforced.
   The 13 MDS-UPDRS Part III items that define the label are tagged
   bucket="BANNED_LABEL_DEFINING" in the data dictionary, along with NP3TOT,
   NP2PTOT and NHY (Hoehn & Yahr), each of which encodes the same information
   indirectly:
       - NP3TOT  sums the tremor and gait items into a single total
       - NP2PTOT sums the three Part II formula items into a single total
       - NHY     stage >=3 is *defined* by postural instability, i.e. it
                 restates the PIGD construct
   Downstream code must select features by bucket, never by hand.

6. ON/OFF medication state.
   MDS-UPDRS Part III is sometimes recorded twice at a visit (PDSTATE = OFF and
   ON). Dopaminergic medication suppresses tremor and can therefore flip a
   patient's computed category. One row per patient-visit is kept, preferring
   the OFF-state assessment (untreated severity). PDSTATE_USED records which
   state was actually used for every retained row.

USAGE
-----
    python3 01_preprocess_tdpigd.py

Inputs are the raw PPMI CSVs in ./ppmi_raw/ as downloaded from LONI IDA
(16 Aug 2026 extract).
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
_RAW       = _ROOT / "data" / "raw" / "ppmi_csv"
_PROCESSED = _ROOT / "data" / "processed"
_METRICS   = _ROOT / "reports" / "metrics"
_FIGURES   = _ROOT / "reports" / "figures"
for _d in (_PROCESSED, _METRICS, _FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


RAW_DIR = str(_RAW)
OUT_LONG = str(_PROCESSED / "ppmi_tdpigd_long.csv")
OUT_DICT = str(_PROCESSED / "ppmi_tdpigd_dictionary.csv")
OUT_LOG  = str(_METRICS / "preprocess_runlog.txt")

# ---------------------------------------------------------------------------
# Visit code -> months since baseline.
# Non-scheduled codes (SC screening, LOG, ED early-discontinuation, ST
# symptomatic-therapy, U01/U02 unscheduled, R15-R22 remote/telephone) have no
# fixed position on the study timeline and are dropped rather than force-mapped.
# ---------------------------------------------------------------------------
VISIT_MONTHS = {
    "BL": 0, "V01": 3, "V02": 6, "V03": 9, "V04": 12, "V05": 15,
    "V06": 18, "V07": 21, "V08": 24, "V09": 30, "V10": 36, "V11": 42,
    "V12": 48, "V13": 54, "V14": 60, "V15": 66, "V16": 72, "V17": 78,
    "V18": 84, "V19": 90, "V20": 96,
}

# ---------------------------------------------------------------------------
# STEBBINS ET AL. (2013) TD / PIGD DEFINITION
#
#   tremor_score = mean of 11 items:
#       MDS-UPDRS 2.10  tremor (patient-reported)          -> NP2TRMR
#       MDS-UPDRS 3.15a/b postural tremor of hands R/L      -> NP3PTRMR/L
#       MDS-UPDRS 3.16a/b kinetic tremor of hands R/L       -> NP3KTRMR/L
#       MDS-UPDRS 3.17a-e rest tremor amplitude RUE/LUE/
#                         RLE/LLE/lip-jaw                   -> NP3RTARU/ALU/
#                                                              ARL/ALL/ALJ
#       MDS-UPDRS 3.18  constancy of rest tremor            -> NP3RTCON
#
#   pigd_score = mean of 5 items:
#       MDS-UPDRS 2.12  walking and balance (patient-rep.)  -> NP2WALK
#       MDS-UPDRS 2.13  freezing (patient-reported)         -> NP2FREZ
#       MDS-UPDRS 3.10  gait                                -> NP3GAIT
#       MDS-UPDRS 3.11  freezing of gait                    -> NP3FRZGT
#       MDS-UPDRS 3.12  postural stability                  -> NP3PSTBL
#
#   ratio = tremor_score / pigd_score
#       ratio >= 1.15  -> TD
#       ratio <= 0.90  -> PIGD
#       otherwise      -> INDETERMINATE
#
#   Edge cases (defined explicitly here, since the ratio is undefined):
#       pigd_score == 0 and tremor_score  > 0 -> TD (pure tremor, no PIGD)
#       pigd_score == 0 and tremor_score == 0 -> INDETERMINATE (asymptomatic on
#                                                both constructs; 0/0)
# ---------------------------------------------------------------------------
TREMOR_P2 = ["NP2TRMR"]
TREMOR_P3 = ["NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
             "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ",
             "NP3RTCON"]
PIGD_P2 = ["NP2WALK", "NP2FREZ"]
PIGD_P3 = ["NP3GAIT", "NP3FRZGT", "NP3PSTBL"]

TREMOR_ITEMS = TREMOR_P2 + TREMOR_P3
PIGD_ITEMS = PIGD_P2 + PIGD_P3
LABEL_DEFINING_P3 = TREMOR_P3 + PIGD_P3          # never allowed as predictors
LABEL_DEFINING_P2 = TREMOR_P2 + PIGD_P2          # allowed only in Variant A
BANNED_DERIVED = ["NP3TOT", "NP2PTOT", "NHY"]    # encode the label indirectly

TD_CUTOFF = 1.15
PIGD_CUTOFF = 0.90

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def load(fname, **kw):
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raw input: {path}")
    return pd.read_csv(path, low_memory=False, **kw)


def keep(df, cols, rename=None):
    """Subset to `cols` that actually exist, optionally renaming."""
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log(f"    NOTE: columns absent from source, skipped: {missing}")
    out = df[present].copy()
    if rename:
        out = out.rename(columns=rename)
    return out


def item_sum(df, items, new_col):
    """Sum raw item values into a crude total.

    NOTE: this is a plain arithmetic sum of raw item codes, NOT the official
    reverse-coded clinical scoring for GDS-15 / STAI / RBDSQ. It is retained
    only as a monotonic severity proxy and is labelled as such in the data
    dictionary. Do not report these as validated instrument totals.
    """
    cols = [c for c in items if c in df.columns]
    df[new_col] = (df[cols].apply(pd.to_numeric, errors="coerce")
                   .sum(axis=1, min_count=1))
    return df


log("=" * 78)
log("01_preprocess_tdpigd.py")
log(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
log(f"Raw input directory: {os.path.abspath(RAW_DIR)}")
log("=" * 78)

# ===========================================================================
# STEP 1  Cohort definition
# ===========================================================================
log("\n[STEP 1] Cohort definition")
cohort_raw = load("Subject_Cohort_History_16Aug2026.csv")
cohort = cohort_raw.drop_duplicates(subset="PATNO")[["PATNO", "APPRDX", "COHORT"]].copy()
log(f"  Subjects in cohort file: {cohort['PATNO'].nunique()}")
log(f"  COHORT value counts:\n{cohort['COHORT'].value_counts().to_string()}")

pd_cohort = cohort[cohort["COHORT"] == 1].copy()
pd_cohort["GENETIC_COHORT"] = (pd_cohort["APPRDX"] == 5).astype(int)
log(f"  PD cohort (COHORT==1): {len(pd_cohort)} patients "
    f"({(pd_cohort.GENETIC_COHORT == 0).sum()} sporadic APPRDX==1, "
    f"{(pd_cohort.GENETIC_COHORT == 1).sum()} genetic-cohort APPRDX==5)")
pd_ids = set(pd_cohort["PATNO"])

# ===========================================================================
# STEP 2  MDS-UPDRS Part III -- item level, ON/OFF de-duplicated
# ===========================================================================
log("\n[STEP 2] MDS-UPDRS Part III (item level, ON/OFF de-duplication)")
p3_raw = load("MDS-UPDRS_Part_III_16Aug2026.csv")
log(f"  Raw Part III rows: {len(p3_raw)}")

p3_cols = ["PATNO", "EVENT_ID", "INFODT", "PDSTATE", "PDTRTMNT"] + \
          LABEL_DEFINING_P3 + ["NP3TOT", "NHY"]
p3 = keep(p3_raw, p3_cols)

# PPMI codes "unable to rate" (UR) in Part III as the NUMBER 101, not only as the
# string "UR". Left as-is, 101 enters the tremor / PIGD means as a real score and
# silently forces a PIGD label (bug found 30 Sep 2026: 104 labelled rows, 98 of
# them PIGD). Treat it as missing, so the completeness rule leaves the visit
# unlabelled. Valid item scores are 0-4; NHY is 0-5.
_ur_cols = LABEL_DEFINING_P3 + ["NHY"]
p3[_ur_cols] = p3[_ur_cols].apply(pd.to_numeric, errors="coerce")
_n_ur = int((p3[_ur_cols] == 101).sum().sum())
p3[_ur_cols] = p3[_ur_cols].mask(p3[_ur_cols] == 101)
log(f"  Part III cells coded 101 (unable to rate) -> NaN: {_n_ur}")

dups_before = p3.duplicated(subset=["PATNO", "EVENT_ID"]).sum()
# Prefer the OFF-medication assessment: sort so OFF sorts first, keep first.
p3["_off_pref"] = (p3["PDSTATE"] != "OFF").astype(int)
p3 = (p3.sort_values(["PATNO", "EVENT_ID", "_off_pref"])
        .drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
        .drop(columns="_off_pref")
        .rename(columns={"PDSTATE": "PDSTATE_USED"}))
log(f"  Duplicate PATNO+EVENT_ID rows before de-dup: {dups_before}")
log(f"  Rows after keeping one (OFF-preferred) per patient-visit: {len(p3)}")
log(f"  PDSTATE_USED distribution:\n"
    f"{p3['PDSTATE_USED'].value_counts(dropna=False).to_string()}")

# ===========================================================================
# STEP 3  MDS-UPDRS Part II -- item level (patient-reported)
# ===========================================================================
log("\n[STEP 3] MDS-UPDRS Part II (item level, patient-reported)")
p2_raw = load("MDS_UPDRS_Part_II__Patient_Questionnaire_16Aug2026.csv")
P2_NON_FORMULA = ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES",
                  "NP2HYGN", "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2RISE"]
p2 = keep(p2_raw, ["PATNO", "EVENT_ID"] + LABEL_DEFINING_P2 +
                  P2_NON_FORMULA + ["NP2PTOT"])
p2 = p2.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
log(f"  Part II rows retained: {len(p2)}")

# ===========================================================================
# STEP 4  Remaining clinical instruments (candidate cheap predictors)
# ===========================================================================
log("\n[STEP 4] Other clinical instruments")

updrs1 = keep(load("MDS-UPDRS_Part_I_16Aug2026.csv"),
              ["PATNO", "EVENT_ID", "NP1RTOT"])
updrs1pq = keep(load("MDS-UPDRS_Part_I_Patient_Questionnaire_16Aug2026.csv"),
                ["PATNO", "EVENT_ID", "NP1PTOT"])
moca = keep(load("Montreal_Cognitive_Assessment__MoCA__16Aug2026.csv"),
            ["PATNO", "EVENT_ID", "MCATOT"])

scopa = load("SCOPA-AUT_16Aug2026.csv")
# SCOPA-AUT items are scored 0-3; PPMI codes "not applicable" (e.g. the sexual-
# function or catheter items) as 9. Summed raw, each 9 added 9 points of fake
# autonomic burden (bug found 30 Sep 2026: 24,513 raw cells). Treat as missing.
_scau = [f"SCAU{i}" for i in range(1, 26)]
_scau = [c for c in _scau if c in scopa.columns]
scopa[_scau] = scopa[_scau].apply(pd.to_numeric, errors="coerce")
log(f"  SCOPA-AUT cells coded 9 (not applicable) -> NaN: {int((scopa[_scau] == 9).sum().sum())}")
scopa[_scau] = scopa[_scau].mask(scopa[_scau] == 9)
scopa = item_sum(scopa, _scau, "SCOPA_AUT_TOTAL")
scopa = keep(scopa, ["PATNO", "EVENT_ID", "SCOPA_AUT_TOTAL"])

gds = load("Geriatric_Depression_Scale__Short_Version__16Aug2026.csv")
gds = item_sum(gds, ["GDSSATIS", "GDSDROPD", "GDSEMPTY", "GDSBORED", "GDSGSPIR",
                     "GDSAFRAD", "GDSHAPPY", "GDSHLPLS", "GDSHOME", "GDSMEMRY",
                     "GDSALIVE", "GDSWRTLS", "GDSENRGY", "GDSHOPLS",
                     "GDSBETER"], "GDS_TOTAL")
gds = keep(gds, ["PATNO", "EVENT_ID", "GDS_TOTAL"])

stai = load("State-Trait_Anxiety_Inventory_for_Adults_16Aug2026.csv")
stai = item_sum(stai, [f"STAIAD{i}" for i in range(1, 41)], "STAI_TOTAL")
stai = keep(stai, ["PATNO", "EVENT_ID", "STAI_TOTAL"])

ess = load("Epworth_Sleepiness_Scale_16Aug2026.csv")
ess = item_sum(ess, [f"ESS{i}" for i in range(1, 9)], "ESS_TOTAL")
ess = keep(ess, ["PATNO", "EVENT_ID", "ESS_TOTAL"])

rbd = load("REM_Sleep_Behavior_Disorder_Screening_Questionnaire_16Aug2026.csv")
rbd = item_sum(rbd, ["DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR",
                     "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN",
                     "DRMREMEM", "SLPDSTRB"], "RBDSQ_TOTAL")
rbd = keep(rbd, ["PATNO", "EVENT_ID", "RBDSQ_TOTAL"])

age = keep(load("Age_at_visit_16Aug2026.csv"), ["PATNO", "EVENT_ID", "AGE_AT_VISIT"])

# --- Medication: MUST be joined by DATE INTERVAL, not by EVENT_ID ----------
# The medication logs are running records, not visit-linked: their EVENT_ID
# only ever takes the values LOG / ED / SC, so a PATNO+EVENT_ID join against
# scheduled visits (BL, V01, ...) matches nothing and silently yields an
# all-missing column. (This bug was present in the earlier pipeline, where
# LEDD_TOTAL_MG was 100% NaN.) Instead, each medication record carries
# STARTDT and STOPDT (MM/YYYY; blank STOPDT = still ongoing), so for every
# patient-visit we sum the LEDD of all medications active on that visit date.
log("  Medication: building LEDD via date-interval join (not EVENT_ID)")
ledd_raw = load("LEDD_Concomitant_Medication_Log_16Aug2026.csv")
ledd_raw["_ledd"] = pd.to_numeric(ledd_raw["LEDD"], errors="coerce")
ledd_raw["_start"] = pd.to_datetime(ledd_raw["STARTDT"], format="%m/%Y", errors="coerce")
ledd_raw["_stop"] = pd.to_datetime(ledd_raw["STOPDT"], format="%m/%Y", errors="coerce")
log(f"    LEDD log rows: {len(ledd_raw)}; numeric LEDD parsed: "
    f"{ledd_raw['_ledd'].notna().mean()*100:.1f}%")
_LEDD_LOG = ledd_raw[["PATNO", "_ledd", "_start", "_stop"]].dropna(subset=["_start"])

conmed_raw = load("Concomitant_Medication_Log_16Aug2026.csv")
conmed_raw["_start"] = pd.to_datetime(conmed_raw["STARTDT"], format="%m/%Y", errors="coerce")
conmed_raw["_stop"] = pd.to_datetime(conmed_raw["STOPDT"], format="%m/%Y", errors="coerce")
_CONMED_LOG = conmed_raw[["PATNO", "_start", "_stop"]].dropna(subset=["_start"])


def active_meds_at_visits(visits, med_log, value_col=None, out_name="OUT"):
    """For each (PATNO, visit date), aggregate medication records whose
    [STARTDT, STOPDT] interval covers that date. A missing STOPDT is treated
    as still ongoing. Returns one row per PATNO+EVENT_ID."""
    v = visits[["PATNO", "EVENT_ID", "_visit_date"]].dropna(subset=["_visit_date"])
    m = v.merge(med_log, on="PATNO", how="inner")
    active = (m["_start"] <= m["_visit_date"]) & (
        m["_stop"].isna() | (m["_stop"] >= m["_visit_date"]))
    m = m[active]
    if value_col:
        agg = (m.groupby(["PATNO", "EVENT_ID"], as_index=False)[value_col]
               .sum().rename(columns={value_col: out_name}))
    else:
        agg = (m.groupby(["PATNO", "EVENT_ID"]).size()
               .reset_index(name=out_name))
    return agg

# Imaging (resource-dependent bucket)
sbr = keep(load("Xing_Core_Lab_-_Quant_SBR_16Aug2026.csv"),
           ["PATNO", "EVENT_ID", "CAUDATE_REF_CWM", "PUTAMEN_REF_CWM",
            "STRIATUM_REF_CWM"])

# ===========================================================================
# STEP 5  Static (per-patient) tables
# ===========================================================================
log("\n[STEP 5] Static per-patient tables")
demo = keep(load("Demographics_16Aug2026.csv"), ["PATNO", "SEX", "HANDED"])
demo = demo.drop_duplicates(subset="PATNO")

visread = keep(load("Xing_Core_Lab_-_Visual_Read_16Aug2026.csv"),
               ["PATNO", "DATSCAN_VISINTRP"]).drop_duplicates(subset="PATNO")

genetics = keep(load("iu_genetic_consensus_20251025_16Aug2026.csv"),
                ["PATNO", "APOE", "LRRK2", "GBA", "VPS35", "SNCA", "PRKN",
                 "PARK7", "PINK1"]).drop_duplicates(subset="PATNO")

# Symptom / diagnosis dates -> disease duration at each visit
pd_hist = keep(load("PD_Diagnosis_History_16Aug2026.csv"),
               ["PATNO", "EVENT_ID", "SXDT", "PDDXDT"])
# Take the earliest non-null record per patient as the definitive dates
pd_hist_static = (pd_hist.sort_values(["PATNO", "EVENT_ID"])
                  .groupby("PATNO", as_index=False)
                  .agg({"SXDT": "first", "PDDXDT": "first"}))

# ===========================================================================
# STEP 6  Join everything
# ===========================================================================
log("\n[STEP 6] Joining")
visit_tables = [p3, p2, updrs1, updrs1pq, moca, scopa, gds, stai, ess, rbd,
                age, sbr]
df = visit_tables[0]
for t in visit_tables[1:]:
    t = t.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
    df = df.merge(t, on=["PATNO", "EVENT_ID"], how="outer")
log(f"  After visit-level outer joins: {len(df)} rows, {df.shape[1]} cols")

for st in [demo, visread, genetics, pd_hist_static,
           pd_cohort[["PATNO", "APPRDX", "COHORT", "GENETIC_COHORT"]]]:
    df = df.merge(st, on="PATNO", how="left")
log(f"  After static left joins: {len(df)} rows, {df.shape[1]} cols")

# ===========================================================================
# STEP 7  Filters: PD cohort, scheduled visits only
# ===========================================================================
log("\n[STEP 7] Filtering")
n0 = df["PATNO"].nunique()
df = df[df["PATNO"].isin(pd_ids)].copy()
log(f"  Restrict to PD cohort (COHORT==1): {n0} -> {df['PATNO'].nunique()} patients, {len(df)} rows")

df["VISIT_MONTH"] = df["EVENT_ID"].map(VISIT_MONTHS)
n_unsched = df["VISIT_MONTH"].isna().sum()
unsched_codes = sorted(df.loc[df["VISIT_MONTH"].isna(), "EVENT_ID"].dropna().unique().tolist())
df = df[df["VISIT_MONTH"].notna()].copy()
df["VISIT_MONTH"] = df["VISIT_MONTH"].astype(int)
log(f"  Dropped {n_unsched} rows with non-scheduled visit codes: {unsched_codes}")
log(f"  Remaining: {len(df)} rows, {df['PATNO'].nunique()} patients")

# Explicit missing-data codes -> NaN (no imputation; see design decision 4)
df = df.replace({"UR": np.nan, "NA": np.nan, "": np.nan})

# ===========================================================================
# STEP 8  Disease duration
# ===========================================================================
log("\n[STEP 8] Disease duration")


def parse_ppmi_date(s):
    """PPMI dates are typically MM/YYYY."""
    return pd.to_datetime(s, format="%m/%Y", errors="coerce")


df["_sx_date"] = parse_ppmi_date(df["SXDT"]) if "SXDT" in df.columns else pd.NaT
df["_dx_date"] = parse_ppmi_date(df["PDDXDT"]) if "PDDXDT" in df.columns else pd.NaT
df["_visit_date"] = parse_ppmi_date(df["INFODT"]) if "INFODT" in df.columns else pd.NaT
log(f"  Visit date (INFODT) parsed for {df['_visit_date'].notna().mean()*100:.1f}% of rows")

df["YRS_SINCE_SYMPTOM_ONSET"] = (df["_visit_date"] - df["_sx_date"]).dt.days / 365.25
df["YRS_SINCE_DIAGNOSIS"] = (df["_visit_date"] - df["_dx_date"]).dt.days / 365.25
for c in ["YRS_SINCE_SYMPTOM_ONSET", "YRS_SINCE_DIAGNOSIS"]:
    # Negative durations indicate inconsistent source dates -> treat as missing
    bad = (df[c] < 0).sum()
    if bad:
        log(f"  {c}: {bad} negative values set to NaN (inconsistent source dates)")
    df.loc[df[c] < 0, c] = np.nan
    log(f"  {c}: {df[c].notna().mean()*100:.1f}% present, "
        f"median {df[c].median():.2f} yrs")

# --- Medication features, joined by date interval (see STEP 4 note) --------
log("\n[STEP 8b] Medication burden at each visit (date-interval join)")
ledd_at_visit = active_meds_at_visits(df, _LEDD_LOG, value_col="_ledd",
                                      out_name="LEDD_TOTAL_MG")
conmed_at_visit = active_meds_at_visits(df, _CONMED_LOG, out_name="N_CONMEDS")
df = df.merge(ledd_at_visit, on=["PATNO", "EVENT_ID"], how="left")
df = df.merge(conmed_at_visit, on=["PATNO", "EVENT_ID"], how="left")
# A patient with no active PD medication legitimately has LEDD = 0, not NaN,
# provided we know their visit date (i.e. the interval test was evaluable).
evaluable = df["_visit_date"].notna()
df.loc[evaluable & df["LEDD_TOTAL_MG"].isna(), "LEDD_TOTAL_MG"] = 0.0
df.loc[evaluable & df["N_CONMEDS"].isna(), "N_CONMEDS"] = 0.0
log(f"  LEDD_TOTAL_MG present: {df['LEDD_TOTAL_MG'].notna().mean()*100:.1f}% "
    f"| median {df['LEDD_TOTAL_MG'].median():.0f} mg "
    f"| {(df['LEDD_TOTAL_MG'] == 0).sum()} rows on no PD medication")
log(f"  N_CONMEDS present: {df['N_CONMEDS'].notna().mean()*100:.1f}% "
    f"| median {df['N_CONMEDS'].median():.0f} concomitant meds")
log("  LEDD by visit month (median, should rise over time as PD progresses):")
log(df.groupby("VISIT_MONTH")["LEDD_TOTAL_MG"].median().head(9).to_string())

df = df.drop(columns=["_sx_date", "_dx_date", "_visit_date"])

# ===========================================================================
# STEP 9  Compute the TD / PIGD / Indeterminate label (Stebbins 2013)
# ===========================================================================
log("\n[STEP 9] Computing TD/PIGD label (Stebbins et al. 2013)")

for c in TREMOR_ITEMS + PIGD_ITEMS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

have_all_tremor = df[TREMOR_ITEMS].notna().all(axis=1)
have_all_pigd = df[PIGD_ITEMS].notna().all(axis=1)
computable = have_all_tremor & have_all_pigd
log(f"  Rows with all 11 tremor items present : {have_all_tremor.sum()}")
log(f"  Rows with all  5 PIGD  items present  : {have_all_pigd.sum()}")
log(f"  Rows with a fully computable label    : {computable.sum()} "
    f"({computable.mean()*100:.1f}% of {len(df)} rows)")

# Means are computed on complete items only; rows lacking any item get NaN and
# therefore no label, rather than a label derived from a partial numerator.
df["TREMOR_SCORE"] = np.where(computable, df[TREMOR_ITEMS].mean(axis=1), np.nan)
df["PIGD_SCORE"] = np.where(computable, df[PIGD_ITEMS].mean(axis=1), np.nan)

with np.errstate(divide="ignore", invalid="ignore"):
    ratio = df["TREMOR_SCORE"] / df["PIGD_SCORE"]
df["TD_PIGD_RATIO"] = ratio


def classify(row):
    t, p = row["TREMOR_SCORE"], row["PIGD_SCORE"]
    if pd.isna(t) or pd.isna(p):
        return np.nan
    if p == 0:
        # ratio undefined; see edge-case rules in the header comment
        return "TD" if t > 0 else "INDETERMINATE"
    r = t / p
    if r >= TD_CUTOFF:
        return "TD"
    if r <= PIGD_CUTOFF:
        return "PIGD"
    return "INDETERMINATE"


df["LABEL"] = df.apply(classify, axis=1)

n_zero_pigd = ((df["PIGD_SCORE"] == 0) & computable).sum()
log(f"  Edge case -- PIGD score == 0 (ratio undefined): {n_zero_pigd} rows")
log(f"  Label distribution (visit level):\n"
    f"{df['LABEL'].value_counts(dropna=False).to_string()}")
log(f"  Label distribution (%):\n"
    f"{(df['LABEL'].value_counts(normalize=True)*100).round(1).to_string()}")

# ===========================================================================
# STEP 10  Transition (label-flip) target
# ===========================================================================
log("\n[STEP 10] Transition / label-flip target")
df = df.sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True)

labelled = df["LABEL"].notna()
df["NEXT_LABEL"] = df.groupby("PATNO")["LABEL"].shift(-1)
df["NEXT_VISIT_MONTH"] = df.groupby("PATNO")["VISIT_MONTH"].shift(-1)
df["MONTHS_TO_NEXT_VISIT"] = df["NEXT_VISIT_MONTH"] - df["VISIT_MONTH"]

both_known = df["LABEL"].notna() & df["NEXT_LABEL"].notna()
df["LABEL_FLIPPED_NEXT"] = np.where(
    both_known, (df["LABEL"] != df["NEXT_LABEL"]).astype(float), np.nan)

flip_rate = df.loc[both_known, "LABEL_FLIPPED_NEXT"].mean()
log(f"  Consecutive visit pairs with both labels known: {both_known.sum()}")
log(f"  Visit-to-visit flip rate: {flip_rate*100:.1f}%")
log("  Flip rate by current label:")
log(df.loc[both_known].groupby("LABEL")["LABEL_FLIPPED_NEXT"]
    .agg(["mean", "size"]).assign(mean=lambda x: (x["mean"]*100).round(1))
    .rename(columns={"mean": "flip_%", "size": "n_pairs"}).to_string())

# Per-patient: did this patient EVER change label across their visits?
per_pat = (df[labelled].groupby("PATNO")["LABEL"]
           .agg(n_visits="size", n_distinct="nunique"))
ever_changed = (per_pat["n_distinct"] > 1)
multi = per_pat["n_visits"] > 1
log(f"  Patients with >=2 labelled visits: {multi.sum()}")
log(f"  Of those, ever changed label: {(ever_changed & multi).sum()} "
    f"({(ever_changed & multi).sum()/max(multi.sum(),1)*100:.1f}%)")

# ===========================================================================
# STEP 11  Feature buckets + data dictionary
# ===========================================================================
log("\n[STEP 11] Feature buckets / data dictionary")

KEYS = ["PATNO", "EVENT_ID", "VISIT_MONTH", "INFODT"]
TARGETS = ["LABEL", "NEXT_LABEL", "LABEL_FLIPPED_NEXT", "TREMOR_SCORE",
           "PIGD_SCORE", "TD_PIGD_RATIO", "NEXT_VISIT_MONTH",
           "MONTHS_TO_NEXT_VISIT"]
# STRICT LEAKAGE POLICY (decided 10 Sep 2026):
# No column that participates in computing Y may be used for training.
# That includes the three PATIENT-REPORTED Part II formula items
# (NP2TRMR, NP2WALK, NP2FREZ) -- they are cheap to collect, but they are
# still label-defining, so they are banned outright. The "Variant A"
# option (allowing them, with disclosure) is therefore NOT in force.
BANNED = LABEL_DEFINING_P3 + LABEL_DEFINING_P2 + BANNED_DERIVED
FORMULA_P2 = []  # intentionally empty under the strict policy
RESOURCE = ["CAUDATE_REF_CWM", "PUTAMEN_REF_CWM", "STRIATUM_REF_CWM",
            "DATSCAN_VISINTRP", "APOE", "LRRK2", "GBA", "VPS35", "SNCA",
            "PRKN", "PARK7", "PINK1"]
ADMIN = ["APPRDX", "COHORT", "GENETIC_COHORT", "PDSTATE_USED", "SXDT", "PDDXDT"]

BUCKET_NOTES = {
    "KEY": "Identifier / join key. Not a predictor.",
    "TARGET": "Outcome or label-derived quantity. Never a predictor.",
    "BANNED_LABEL_DEFINING": "Participates in computing Y (a Part II or Part "
                             "III formula item, or a total/stage encoding "
                             "them). MUST NEVER be used as a model input, "
                             "regardless of how cheap it is to collect.",
    "CHEAP_FEATURE": "Patient-reported or low-cost clinical measure, plausibly "
                     "available in a non-specialist setting. Core input set.",
    "RESOURCE_DEPENDENT_FEATURE": "Imaging or genotyping. Rarely available in "
                                  "low-resource settings (~94% of surveyed "
                                  "Indian clinicians rarely order these). Use "
                                  "only in the 'with-imaging/genetics' variant.",
    "ADMIN": "Provenance / bookkeeping. Not a predictor.",
}


def bucket_of(col):
    if col in KEYS:
        return "KEY"
    if col in TARGETS:
        return "TARGET"
    if col in BANNED:
        return "BANNED_LABEL_DEFINING"
    if col in FORMULA_P2:
        return "FORMULA_PART2_PATIENT_REPORTED"
    if col in RESOURCE:
        return "RESOURCE_DEPENDENT_FEATURE"
    if col in ADMIN:
        return "ADMIN"
    return "CHEAP_FEATURE"


dictionary = pd.DataFrame({
    "column": df.columns,
    "bucket": [bucket_of(c) for c in df.columns],
    "dtype": [str(df[c].dtype) for c in df.columns],
    "pct_present": [round(df[c].notna().mean() * 100, 1) for c in df.columns],
    "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
dictionary["bucket_meaning"] = dictionary["bucket"].map(BUCKET_NOTES)
dictionary = dictionary.sort_values(["bucket", "column"]).reset_index(drop=True)

log("  Columns per bucket:")
log(dictionary["bucket"].value_counts().to_string())

# Sanity assertion: no banned column may ever be classed as a usable feature.
usable = set(dictionary.loc[dictionary.bucket.isin(
    ["CHEAP_FEATURE", "RESOURCE_DEPENDENT_FEATURE"]), "column"])
overlap = set(BANNED) & usable
assert not overlap, f"LEAKAGE GUARD FAILED -- banned cols in feature buckets: {overlap}"

# Strict policy: EVERY column feeding the Y computation must be banned.
all_label_inputs = set(TREMOR_ITEMS) | set(PIGD_ITEMS)
escaped = all_label_inputs & usable
assert not escaped, f"STRICT GUARD FAILED -- Y-defining items usable as features: {escaped}"
log(f"  Leakage guard: PASSED ({len(BANNED)} banned columns)")
log(f"  Strict guard:  PASSED (all {len(all_label_inputs)} Y-defining items "
    f"excluded, including the 3 patient-reported Part II items)")
log(f"  Usable feature columns remaining: {len(usable)}")

# ===========================================================================
# STEP 12  Per-round availability (informational; reshape happens downstream)
# ===========================================================================
log("\n[STEP 12] Per-round patient availability (label computable at each visit)")
lab = df[df["LABEL"].notna()]
avail = {mo: set(lab.loc[lab.VISIT_MONTH == mo, "PATNO"]) for mo in [0, 6, 12, 24]}
rounds = {
    "Round 1 (BL)": avail[0],
    "Round 2 (BL+6)": avail[0] & avail[6],
    "Round 3 (BL+6+12)": avail[0] & avail[6] & avail[12],
    "Round 4 (BL+6+12+24)": avail[0] & avail[6] & avail[12] & avail[24],
    "ALT annual-only (BL+12)": avail[0] & avail[12],
    "ALT annual-only (BL+12+24)": avail[0] & avail[12] & avail[24],
}
for k, v in rounds.items():
    log(f"  {k:<30}: {len(v)} patients")

# ===========================================================================
# STEP 13  Save
# ===========================================================================
log("\n[STEP 13] Writing outputs")
df.to_csv(OUT_LONG, index=False)
dictionary.to_csv(OUT_DICT, index=False)
log(f"  {OUT_LONG}: {df.shape[0]} rows x {df.shape[1]} cols")
log(f"  {OUT_DICT}: {len(dictionary)} column definitions")

dup_keys = df.duplicated(subset=["PATNO", "EVENT_ID"]).sum()
log(f"  Duplicate PATNO+EVENT_ID rows in output: {dup_keys}")
assert dup_keys == 0, "Duplicate patient-visit keys in output!"

log(f"\nRun finished: {datetime.now().isoformat(timespec='seconds')}")
with open(OUT_LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_log_lines) + "\n")
print(f"\nProvenance log written to {OUT_LOG}")
