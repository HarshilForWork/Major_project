"""
preprocess_adni.py
===================
Preprocessing for the ADNI (Alzheimer's Disease Neuroimaging Initiative) second
cohort. Built to the same discipline as preprocess.py (the PPMI pipeline), so the
same leakage policy, feature-bucket machinery and AR(1)-SHAP code can run on
either cohort -- that side-by-side result is what turns AR(1)-SHAP from "a PPMI
script" into a framework.

WHAT THIS SCRIPT PRODUCES
--------------------------
  adni_dx_long.csv        One row per patient per visit. Long format, same
                           convention as ppmi_tdpigd_long.csv.
  adni_dx_dictionary.csv  Same shape as the PPMI dictionary: every column, its
                           bucket, its missingness.
  adni_dx_preprocess_runlog.txt   Provenance log.

SOURCE DATA
-----------
ADNI's current download (the ADNIMERGE2 R package) ships ~235 separate raw
tables, not one merged file. `notebooks/convert_adnimerge.py` converts the 15
tables this pipeline needs into data/raw/adni_csv/*.csv. See that script and
HANDOFF_AR1_SHAP.md for how they were obtained.

KEY DESIGN DECISIONS
---------------------
1. LABEL = the clinical diagnosis (CN / MCI / Dementia), from DXSUM.DIAGNOSIS.
   Unlike PPMI's TD/PIGD label, this is NOT a deterministic formula -- it's a
   site clinician's consensus call, informed by (not computed from) the CDR
   exam, MMSE, informant interview and the neuropsych battery. This changes the
   task's framing from PPMI's "can cheap data reproduce a formula" to "can cheap
   data + history estimate the diagnosis a full clinical work-up would reach,
   between visits where that full work-up isn't repeated". It also means
   patients genuinely CONVERT over time (CN -> MCI -> Dementia), which is a
   cleaner fit for a "transition / time-to-confidence" story than PPMI's label
   wobble was.

2. BANNED_LABEL_DEFINING = the entire CDR exam (8 columns): the six domain
   ratings (CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE), the global
   score CDGLOBAL, and the sum-of-boxes CDRSB. This was verified empirically,
   not assumed: on 13,043 visits with both a diagnosis and a CDR score,
   CDGLOBAL=0.0 is 91% CN, CDGLOBAL=1.0+ is >92% Dementia, and CDGLOBAL=0.5 is
   the ambiguous band split between MCI and mild Dementia -- structurally the
   same near-deterministic relationship, with the same kind of ambiguous middle
   zone, that PPMI's Stebbins ratio has around its two cutoffs. CDR is also used
   as a literal eligibility/enrollment-group criterion in the ADNI protocol.
   This is the ADNI equivalent of banning NP3TOT, NHY and the 16 Stebbins items.

3. RESOURCE_DEPENDENT_FEATURE = APOE_GENOTYPE (from APOERES). A one-time
   genetic test, not routinely available outside a research cohort -- same
   policy as PPMI's held-out genotype columns. It's a static, baseline-only
   attribute (ADNI draws it once), so it's forward-filled across a patient's
   visits like PPMI's static demographic columns.

4. MMSE total and the Logical Memory delayed-recall score (LDELTOTAL) ARE kept
   as CHEAP_FEATUREs, not banned, even though both are also used by the ADNI
   protocol to help assign a diagnosis. The empirical check shows why this is a
   softer case than CDR: LDELTOTAL's group means are well separated (CN 13.6,
   MCI 8.0, Dementia 1.7) but the within-group spread is wide (sd 3-5) and the
   bands overlap a great deal, unlike CDR's near-complete separation. This is
   the ADNI analogue of PPMI's NP2RISE/NP2TURN axial items: plausible proxy
   information, disclosed rather than banned. State this plainly in any write-up
   that uses ADNI, the same way PPMI's README discloses it (see its Sec. 9.3).

5. VISIT TIMELINE = VISCODE2, not EXAMDATE. ADNI re-codes every visit's raw,
   phase-specific code (e.g. "v06" in ADNI2) into a VISCODE2 that is already a
   consistent "months since this patient's very first baseline" value (bl=0,
   m06=6, ..., m84=84, ...), stable across a patient switching protocol phases
   (ADNI1 -> ADNIGO -> ADNI2 -> ADNI3 -> ADNI4). Verified on a patient who
   switched phases: EXAMDATE keeps advancing in step with VISCODE2's number the
   whole way through. This is used directly as VISIT_MONTH, exactly like PPMI's
   visit-code -> month mapping, and it also sidesteps that EXAMDATE itself is
   missing on ~15% of REGISTRY rows. Codes that don't parse to a fixed month
   (screening, screening-MRI, unscheduled, screen-fail) are dropped, the same
   policy as PPMI dropping its unscheduled/log/remote visit codes.

6. VITALS sentinel code. VSWEIGHT and VSHEIGHT use -1 for "not recorded" mixed
   in with real measurements (115 and 22 rows respectively) -- the same kind of
   trap as PPMI's Part III code 101 ("unable to rate"). Converted to NaN before
   use. Units are mixed (pounds/kilograms, inches/centimeters) and are
   normalised to kilograms and centimeters; a measurement with an unknown unit
   and an implausible value for either unit is set to NaN rather than guessed.

7. NO imputation outside a fold -- same policy as PPMI. NaN is left as NaN.

8. AGE_AT_VISIT is derived, not measured per visit. PTDEMOG gives only a
   patient's birth YEAR (privacy-limited, like most clinical registries) and a
   baseline exam date is available for every patient from REGISTRY. Age at any
   later visit = (baseline exam year - birth year) + VISIT_MONTH / 12. This is
   deterministic and needs no per-visit measurement, the same role PPMI's
   AGE_AT_VISIT / YRS_SINCE_DIAGNOSIS "clock" features play for AR(1)-SHAP.

9. "sc" (screening) PROMOTED TO BASELINE for 742 patients who have no separate
   "bl" entry at all -- their screening visit IS their only anchor in time.
   Checked before deciding: of 2,752 patients with a screening-visit diagnosis,
   2,010 also have a redundant "bl" diagnosis (screening dropped as noise, same
   as before), but 742 have NO "bl" diagnosis whatsoever -- dropping "sc"
   unconditionally would have silently removed all 742 of them from the labelled
   cohort. Rule: "sc" becomes VISIT_MONTH=0 only for a patient with no other
   month-0 entry; otherwise it's dropped as a redundant pre-baseline visit, the
   same as before.

10. DUPLICATE (RID, VISCODE2) rows. A handful of source tables (1-3 rows each)
   have an exact duplicate key, almost always a corrected re-entry. Kept the
   last one by `update_stamp` (the most recent correction), then asserted zero
   duplicate keys remain in the final table -- same invariant PPMI enforces on
   PATNO+EVENT_ID.

Run:  PYTHONPATH=src python -m trace_pd.data.preprocess_adni
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

from trace_pd import config

OUT_LONG = config.ADNI_LONG_TABLE
OUT_DICT = config.ADNI_DICTIONARY
OUT_LOG = config.PROCESSED / "adni_dx_preprocess_runlog.txt"

_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def read(name):
    return pd.read_csv(config.RAW_ADNI / f"{name}.csv", low_memory=False)


def dedup_latest(df, keys, order_col="update_stamp"):
    """Keep the most recently updated row for each key; everything else in this
    script assumes (RID, VISCODE2) is unique per source table."""
    if order_col in df.columns:
        df = df.sort_values(order_col)
    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="last")
    dropped = before - len(df)
    if dropped:
        log(f"    dedup on {keys}: dropped {dropped} duplicate rows, kept the latest")
    return df


# ===========================================================================
# STEP 1  Visit backbone: every (RID, VISCODE2) with a fixed month number
# ===========================================================================
log("=" * 78)
log("ADNI preprocessing")
log("=" * 78)
log(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
log("\n[STEP 1] Visit backbone from REGISTRY")

reg = read("REGISTRY")
log(f"  REGISTRY: {len(reg)} rows, {reg.RID.nunique()} unique RID")


def parse_viscode2(v):
    """'bl' -> 0, 'm06' -> 6, 'm252' -> 252; screening/unscheduled codes -> NaN.
    Verified on a patient who crossed ADNI1 -> ADNI2: EXAMDATE keeps pace with
    this parsed number throughout, i.e. it is a consistent months-since-the-
    patient's-own-baseline value, not reset at each protocol phase."""
    v = str(v)
    if v == "bl":
        return 0
    m = re.match(r"^m(\d+)$", v)
    return int(m.group(1)) if m else np.nan


reg["VISIT_MONTH"] = reg["VISCODE2"].map(parse_viscode2)

# Promote "sc" (screening) to VISIT_MONTH=0 for patients with no other month-0
# entry -- see decision 9. Must have a real exam date to anchor a timeline on.
has_month0 = set(reg.loc[reg["VISIT_MONTH"] == 0, "RID"])
sc_only = reg["VISCODE2"].eq("sc") & reg["EXAMDATE"].notna() & ~reg["RID"].isin(has_month0)
n_promoted = int(sc_only.sum())
reg.loc[sc_only, "VISIT_MONTH"] = 0
log(f"  Promoted {n_promoted} screening visits to VISIT_MONTH=0 "
    f"(these patients have no separate baseline visit; see decision 9)")

n_before = len(reg)
reg = reg[reg["VISIT_MONTH"].notna()].copy()
reg["VISIT_MONTH"] = reg["VISIT_MONTH"].astype(int)
log(f"  Dropped {n_before - len(reg)} rows with a non-fixed-timeline code "
    f"(screening, screening-MRI, screen-fail, unscheduled, etc.)")

reg = dedup_latest(reg, ["RID", "VISCODE2"])
backbone = reg[["RID", "VISCODE2", "VISIT_MONTH", "EXAMDATE", "ORIGPROT", "COLPROT"]].copy()
backbone = backbone.sort_values(["RID", "VISIT_MONTH"]).reset_index(drop=True)
log(f"  Backbone: {len(backbone)} patient-visits, {backbone.RID.nunique()} patients")

# ===========================================================================
# STEP 2  Baseline exam date per patient (for deriving age; see decision 8)
# ===========================================================================
log("\n[STEP 2] Baseline exam date per patient")
base_date = (backbone[backbone.VISIT_MONTH == 0]
             .dropna(subset=["EXAMDATE"])
             .drop_duplicates("RID")
             .set_index("RID")["EXAMDATE"])
base_date = pd.to_datetime(base_date, errors="coerce")
log(f"  Baseline exam date known for {base_date.notna().sum()} / {backbone.RID.nunique()} patients")

# ===========================================================================
# STEP 3  Label: diagnosis per visit, from DXSUM
# ===========================================================================
log("\n[STEP 3] Label (DIAGNOSIS) from DXSUM")
dx = read("DXSUM")
dx = dedup_latest(dx, ["RID", "VISCODE2"])
dx["LABEL"] = dx["DIAGNOSIS"].where(dx["DIAGNOSIS"].isin(["CN", "MCI", "Dementia"]))
log(f"  DXSUM: {len(dx)} rows; label distribution:\n{dx['LABEL'].value_counts(dropna=False).to_string()}")

df = backbone.merge(dx[["RID", "VISCODE2", "LABEL"]], on=["RID", "VISCODE2"], how="left")

# ===========================================================================
# STEP 4  Banned label-defining columns: the full CDR exam
# ===========================================================================
log("\n[STEP 4] CDR exam -- banned as label-defining (see decision 2)")
cdr = read("CDR")
cdr = dedup_latest(cdr, ["RID", "VISCODE2"])
CDR_COLS = ["CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE", "CDGLOBAL", "CDRSB"]
df = df.merge(cdr[["RID", "VISCODE2"] + CDR_COLS], on=["RID", "VISCODE2"], how="left")

# Empirical verification, re-run here (not just asserted in the docstring) so the
# claim is checked against whatever extract is actually loaded.
chk = df.dropna(subset=["LABEL", "CDGLOBAL"])
cn_at_0 = (chk.loc[chk.CDGLOBAL == 0, "LABEL"] == "CN").mean()
dem_at_1plus = (chk.loc[chk.CDGLOBAL >= 1, "LABEL"] == "Dementia").mean()
log(f"  Verification: CDGLOBAL=0 -> CN {cn_at_0 * 100:.1f}% of the time; "
    f"CDGLOBAL>=1 -> Dementia {dem_at_1plus * 100:.1f}% of the time (n={len(chk)})")
assert cn_at_0 > 0.85 and dem_at_1plus > 0.85, (
    "CDR no longer looks label-defining in this extract -- re-check the ban before trusting it")

# ===========================================================================
# STEP 5  Demographics (static) + derived AGE_AT_VISIT
# ===========================================================================
log("\n[STEP 5] Demographics (PTDEMOG, static per patient)")
ptd = read("PTDEMOG")
ptd = ptd.dropna(subset=["RID"]).drop_duplicates("RID", keep="last")
demo = pd.DataFrame({
    "RID": ptd["RID"],
    "SEX": ptd["PTGENDER"],
    "HANDED": ptd["PTHAND"],
    "EDUCATION_YEARS": ptd["PTEDUCAT"],
    "_BIRTH_YEAR": pd.to_datetime(ptd["PTDOBYY"], errors="coerce").dt.year,
}).set_index("RID")

df = df.merge(demo, on="RID", how="left")
baseline_year = df["RID"].map(base_date).dt.year
df["AGE_AT_VISIT"] = (baseline_year - df["_BIRTH_YEAR"]) + df["VISIT_MONTH"] / 12.0
df = df.drop(columns=["_BIRTH_YEAR"])
log(f"  AGE_AT_VISIT computed for {df['AGE_AT_VISIT'].notna().sum()} / {len(df)} rows")
implausible = ((df["AGE_AT_VISIT"] < 40) | (df["AGE_AT_VISIT"] > 110)) & df["AGE_AT_VISIT"].notna()
df.loc[implausible, "AGE_AT_VISIT"] = np.nan
log(f"  Set {int(implausible.sum())} implausible ages (<40 or >110) to missing")

# ===========================================================================
# STEP 6  Genetics (static, held out -- RESOURCE_DEPENDENT_FEATURE)
# ===========================================================================
log("\n[STEP 6] APOE genotype (APOERES, static per patient, held out)")
apoe = read("APOERES").dropna(subset=["RID"]).drop_duplicates("RID", keep="last")
df = df.merge(apoe[["RID", "GENOTYPE"]].rename(columns={"GENOTYPE": "APOE_GENOTYPE"}), on="RID", how="left")

# ===========================================================================
# STEP 7  Cheap features -- cognition, function, mood/behaviour, vitals
# ===========================================================================
log("\n[STEP 7] Cheap, routinely-collectable features")


def merge_cols(df, table_name, rename_map):
    t = read(table_name)
    t = dedup_latest(t, ["RID", "VISCODE2"])
    cols = {"RID": "RID", "VISCODE2": "VISCODE2", **rename_map}
    t = t[list(cols)].rename(columns=cols)
    out = df.merge(t, on=["RID", "VISCODE2"], how="left")
    log(f"  + {table_name}: {list(rename_map.values())}")
    return out


df = merge_cols(df, "MMSE", {"MMSCORE": "MMSE_TOTAL"})
df = merge_cols(df, "MOCA", {"MOCA": "MOCA_TOTAL"})
df = merge_cols(df, "GDSCALE", {"GDTOTAL": "GDS_TOTAL"})
df = merge_cols(df, "FAQ", {"FAQTOTAL": "FAQ_TOTAL"})
df = merge_cols(df, "NPIQ", {"NPISCORE": "NPI_TOTAL"})
df = merge_cols(df, "ECOGPT", {"EcogPtTotal": "ECOG_PATIENT_TOTAL"})
df = merge_cols(df, "ECOGSP", {"EcogSPTotal": "ECOG_PARTNER_TOTAL"})
df = merge_cols(df, "NEUROBAT", {
    "LIMMTOTAL": "LOGMEM_IMMEDIATE", "LDELTOTAL": "LOGMEM_DELAYED",
    "BNTTOTAL": "BOSTON_NAMING", "CATANIMSC": "CATEGORY_FLUENCY_ANIMALS",
    "TRAASCOR": "TRAILS_A_SEC", "TRABSCOR": "TRAILS_B_SEC", "DIGITSCOR": "DIGIT_SPAN",
})

# --- vitals: fix the -1 "not recorded" sentinel, then normalise units --------
log("\n[STEP 7b] Vitals (VSWEIGHT/VSHEIGHT sentinel fix -- see decision 6)")
vit = read("VITALS")
vit = dedup_latest(vit, ["RID", "VISCODE2"])
n_sentinel = int((vit["VSWEIGHT"] < 0).sum() + (vit["VSHEIGHT"] < 0).sum())
vit.loc[vit["VSWEIGHT"] < 0, "VSWEIGHT"] = np.nan
vit.loc[vit["VSHEIGHT"] < 0, "VSHEIGHT"] = np.nan
log(f"  Converted {n_sentinel} sentinel (-1, 'not recorded') values to missing")

weight_kg = np.where(vit["VSWTUNIT"] == "pounds", vit["VSWEIGHT"] * 0.453592, vit["VSWEIGHT"])
height_cm = np.where(vit["VSHTUNIT"] == "inches", vit["VSHEIGHT"] * 2.54, vit["VSHEIGHT"])
# unit missing: only keep if the raw value is already a plausible kg/cm figure,
# otherwise it's ambiguous (could be lb or in) and is set to missing rather than guessed
unit_missing_wt = vit["VSWTUNIT"].isna()
weight_kg = np.where(unit_missing_wt & ((weight_kg < 30) | (weight_kg > 250)), np.nan, weight_kg)
unit_missing_ht = vit["VSHTUNIT"].isna()
height_cm = np.where(unit_missing_ht & ((height_cm < 100) | (height_cm > 220)), np.nan, height_cm)
weight_kg = pd.Series(weight_kg)
height_cm = pd.Series(height_cm)
# Post-conversion plausibility clamp: even with a stated unit, a handful of
# entries convert to an impossible adult figure (e.g. a height that converts to
# 490cm) -- almost certainly a wrong unit tag on the source CRF, not a real
# measurement. These are set to missing rather than guessed at, same policy as
# the age clamp in step 5.
n_bad_wt = int(((weight_kg < 30) | (weight_kg > 250)).sum())
n_bad_ht = int(((height_cm < 130) | (height_cm > 220)).sum())
weight_kg = weight_kg.where((weight_kg >= 30) & (weight_kg <= 250))
height_cm = height_cm.where((height_cm >= 130) & (height_cm <= 220))
log(f"  Cleared {n_bad_wt} implausible weights and {n_bad_ht} implausible heights "
    f"after unit conversion (likely wrong unit tag on the source form)")

vit["WEIGHT_KG"] = weight_kg.to_numpy()
vit["HEIGHT_CM"] = height_cm.to_numpy()

# Height is near-static in adults and is recorded at only a minority of visits
# (often once, at baseline) -- carried across a patient's other visits rather
# than left missing, the same reasoning as PPMI's static demographic columns.
vit = vit.sort_values(["RID", "VISCODE2"])
vit["HEIGHT_CM"] = vit.groupby("RID")["HEIGHT_CM"].transform(lambda s: s.ffill().bfill())
log(f"  WEIGHT_KG: {vit['WEIGHT_KG'].notna().sum()} / {len(vit)} usable; "
    f"HEIGHT_CM: {vit['HEIGHT_CM'].notna().sum()} / {len(vit)} usable after per-patient carry-forward")

df = df.merge(vit[["RID", "VISCODE2", "WEIGHT_KG", "HEIGHT_CM", "VSBPSYS", "VSBPDIA", "VSPULSE"]]
              .rename(columns={"VSBPSYS": "BP_SYSTOLIC", "VSBPDIA": "BP_DIASTOLIC", "VSPULSE": "PULSE"}),
              on=["RID", "VISCODE2"], how="left")

CHEAP_FEATURES = [
    "SEX", "HANDED", "EDUCATION_YEARS", "AGE_AT_VISIT",
    "MMSE_TOTAL", "MOCA_TOTAL", "LOGMEM_IMMEDIATE", "LOGMEM_DELAYED",
    "BOSTON_NAMING", "CATEGORY_FLUENCY_ANIMALS", "TRAILS_A_SEC", "TRAILS_B_SEC", "DIGIT_SPAN",
    "FAQ_TOTAL", "ECOG_PATIENT_TOTAL", "ECOG_PARTNER_TOTAL",
    "GDS_TOTAL", "NPI_TOTAL",
    "WEIGHT_KG", "HEIGHT_CM", "BP_SYSTOLIC", "BP_DIASTOLIC", "PULSE",
]
log(f"\n  {len(CHEAP_FEATURES)} cheap features assembled")

# ===========================================================================
# STEP 8  Transition target: will the diagnosis change at the next visit?
# ===========================================================================
log("\n[STEP 8] Transition target (LABEL_FLIPPED_NEXT), same construction as PPMI")
df = df.sort_values(["RID", "VISIT_MONTH"]).reset_index(drop=True)
g = df.groupby("RID")
df["NEXT_LABEL"] = g["LABEL"].shift(-1)
df["NEXT_VISIT_MONTH"] = g["VISIT_MONTH"].shift(-1)
df["MONTHS_TO_NEXT_VISIT"] = df["NEXT_VISIT_MONTH"] - df["VISIT_MONTH"]
has_pair = df["LABEL"].notna() & df["NEXT_LABEL"].notna()
df["LABEL_FLIPPED_NEXT"] = np.where(has_pair, (df["LABEL"] != df["NEXT_LABEL"]).astype(float), np.nan)
n_pairs = int(has_pair.sum())
flip_rate = df.loc[has_pair, "LABEL_FLIPPED_NEXT"].mean()
log(f"  {n_pairs} labelled consecutive-visit pairs; overall flip rate {flip_rate * 100:.1f}%")
log(df.loc[has_pair].groupby("LABEL")["LABEL_FLIPPED_NEXT"].mean().mul(100).round(1).to_string())

# ===========================================================================
# STEP 9  Duplicate-key check (same invariant as PPMI's PATNO+EVENT_ID)
# ===========================================================================
log("\n[STEP 9] Duplicate-key check")
dup_keys = df.duplicated(subset=["RID", "VISCODE2"]).sum()
log(f"  Duplicate RID+VISCODE2 rows: {dup_keys}")
assert dup_keys == 0, "Duplicate patient-visit keys in output!"

# ===========================================================================
# STEP 10  Dictionary + leakage guard
# ===========================================================================
log("\n[STEP 10] Data dictionary and leakage guard")

KEY = ["RID", "VISCODE2", "VISIT_MONTH", "EXAMDATE"]
ADMIN = ["ORIGPROT", "COLPROT"]
TARGET = ["LABEL", "NEXT_LABEL", "LABEL_FLIPPED_NEXT", "NEXT_VISIT_MONTH", "MONTHS_TO_NEXT_VISIT"]
BANNED = CDR_COLS
RESOURCE_DEPENDENT = ["APOE_GENOTYPE"]

BUCKET_NOTES = {
    "CHEAP_FEATURE": "Routinely collectable without a specialist work-up. Used for training.",
    "RESOURCE_DEPENDENT_FEATURE": "Needs a lab test (genotyping). Held out, not used for training.",
    "BANNED_LABEL_DEFINING": "Part of the CDR exam the diagnosis is informed by / enrollment-gated on. Never a feature.",
    "TARGET": "The label, or computed from it.",
    "KEY": "Identifiers and timeline.",
    "ADMIN": "Provenance / bookkeeping. Not a predictor.",
}


def bucket_of(c):
    if c in CHEAP_FEATURES:
        return "CHEAP_FEATURE"
    if c in RESOURCE_DEPENDENT:
        return "RESOURCE_DEPENDENT_FEATURE"
    if c in BANNED:
        return "BANNED_LABEL_DEFINING"
    if c in TARGET:
        return "TARGET"
    if c in KEY:
        return "KEY"
    if c in ADMIN:
        return "ADMIN"
    raise ValueError(f"Column {c} isn't assigned to a bucket -- fix the dictionary logic")


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

usable = set(dictionary.loc[dictionary.bucket.isin(
    ["CHEAP_FEATURE", "RESOURCE_DEPENDENT_FEATURE"]), "column"])
overlap = set(BANNED) & usable
assert not overlap, f"LEAKAGE GUARD FAILED -- banned cols in feature buckets: {overlap}"
log(f"  Leakage guard: PASSED ({len(BANNED)} banned CDR columns, "
    f"{len(CHEAP_FEATURES)} usable cheap features)")

# ===========================================================================
# STEP 11  Save
# ===========================================================================
log("\n[STEP 11] Writing outputs")
df.to_csv(OUT_LONG, index=False)
dictionary.to_csv(OUT_DICT, index=False)
log(f"  {OUT_LONG}: {df.shape[0]} rows x {df.shape[1]} cols, {df.RID.nunique()} patients")
log(f"  {OUT_DICT}: {len(dictionary)} column definitions")
log(f"  Labelled rows: {df['LABEL'].notna().sum()} ({df['LABEL'].notna().mean() * 100:.1f}%)")
log(df["LABEL"].value_counts(dropna=False).to_string())

log(f"\nRun finished: {datetime.now().isoformat(timespec='seconds')}")
with open(OUT_LOG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_log_lines) + "\n")
print(f"\nProvenance log written to {OUT_LOG}")
