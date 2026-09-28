"""
02_baseline_model_tdpigd.py
===========================
Baseline sanity-check model for TD / PIGD / Indeterminate prediction.

PURPOSE
-------
Answer one question: with the STRICT feature policy in force (no column that
participates in computing Y may be used as an input), is there any learnable
signal at all? This is a feasibility check, not a tuned final model.

STRICT FEATURE POLICY (enforced, not assumed)
---------------------------------------------
Features are selected exclusively from the CHEAP_FEATURE bucket of
ppmi_tdpigd_dictionary.csv. All 16 Y-defining MDS-UPDRS items are excluded --
including the three PATIENT-REPORTED Part II items (NP2TRMR, NP2WALK,
NP2FREZ) -- as are NP3TOT, NP2PTOT and NHY, which encode them indirectly.
The script re-verifies this against the label formula and aborts if violated.

EVALUATION DESIGN
-----------------
* Round 1 only (baseline visit), one row per patient -> no within-patient
  repetition, so no grouped-split subtlety for the headline number.
* Also evaluated on ALL visits pooled, where GroupKFold(PATNO) is mandatory
  because the same patient contributes ~16 rows; a row-wise split would leak.
* 5-fold cross-validation, stratified where possible, grouped by patient.
* Class weights applied (TD outnumbers PIGD ~2:1, Indeterminate is ~11%).
* Compared against two honest reference points:
    - MAJORITY: always predict TD (the do-nothing baseline)
    - LOGISTIC: multinomial logistic regression on median-imputed, scaled data
  A boosted model that cannot beat MAJORITY has found nothing.

NOTE ON THE MODEL
-----------------
Uses sklearn HistGradientBoostingClassifier rather than XGBoost (the XGBoost
wheel could not be installed in this environment). Same algorithm family
(histogram-based gradient boosting) with native NaN support, so conclusions
about feasibility transfer; swap in XGBoost for the final paper run.

Run:  python3 02_baseline_model_tdpigd.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
_RAW       = _ROOT / "data" / "raw" / "ppmi_csv"
_PROCESSED = _ROOT / "data" / "processed"
_METRICS   = _ROOT / "reports" / "metrics"
_FIGURES   = _ROOT / "reports" / "figures"
for _d in (_PROCESSED, _METRICS, _FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


LONG = str(_PROCESSED / "ppmi_tdpigd_long.csv")
DICT = str(_PROCESSED / "ppmi_tdpigd_dictionary.csv")
OUT_TXT = str(_METRICS / "baseline_model_results.txt")

TREMOR_ITEMS = ["NP2TRMR", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
                "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ",
                "NP3RTCON"]
PIGD_ITEMS = ["NP2WALK", "NP2FREZ", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"]
CLASSES = ["TD", "PIGD", "INDETERMINATE"]

_lines = []


def log(m=""):
    print(m)
    _lines.append(str(m))


df = pd.read_csv(LONG, low_memory=False)
dd = pd.read_csv(DICT)

# ---------------------------------------------------------------------------
# Feature selection -- by bucket only, then hard-verified
# ---------------------------------------------------------------------------
FEATURES = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
banned = set(TREMOR_ITEMS) | set(PIGD_ITEMS) | {"NP3TOT", "NP2PTOT", "NHY"}
violation = banned & set(FEATURES)
assert not violation, f"STRICT POLICY VIOLATED: {violation}"

# Encode the few object-dtype features numerically
X_all = df[FEATURES].copy()
for c in X_all.columns:
    if X_all[c].dtype == object:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

log("=" * 74)
log("BASELINE MODEL -- TD / PIGD / INDETERMINATE")
log("=" * 74)
log(f"\nStrict policy verified: 0 of the 16 Y-defining items are in the feature set.")
log(f"Features used ({len(FEATURES)}): {FEATURES}")


def evaluate(X, y, groups, title, n_splits=5):
    log("\n" + "=" * 74)
    log(title)
    log("=" * 74)
    log(f"n = {len(y)} rows, {len(np.unique(groups))} patients")
    log("Class distribution: " +
        ", ".join(f"{k} {v} ({v/len(y)*100:.1f}%)"
                  for k, v in y.value_counts().items()))

    if len(np.unique(groups)) == len(y):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
        log("CV: StratifiedKFold (one row per patient, no grouping needed)")
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y, groups=groups)
        log("CV: StratifiedGroupKFold grouped by PATNO (prevents patient leakage)")

    preds = {"HGB": np.empty(len(y), dtype=object),
             "LOGISTIC": np.empty(len(y), dtype=object),
             "MAJORITY": np.empty(len(y), dtype=object)}

    for tr, te in split_iter:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]
        w = compute_sample_weight("balanced", ytr)

        hgb = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
        hgb.fit(Xtr, ytr, sample_weight=w)
        preds["HGB"][te] = hgb.predict(Xte)

        lr = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               multi_class="multinomial"))
        lr.fit(Xtr, ytr)
        preds["LOGISTIC"][te] = lr.predict(Xte)

        preds["MAJORITY"][te] = ytr.value_counts().idxmax()

    rows = []
    for name in ["MAJORITY", "LOGISTIC", "HGB"]:
        p = preds[name].astype(str)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y, p),
            "balanced_acc": balanced_accuracy_score(y, p),
            "macro_F1": f1_score(y, p, average="macro"),
            "weighted_F1": f1_score(y, p, average="weighted"),
        })
    res = pd.DataFrame(rows).set_index("model").round(3)
    log("\nOut-of-fold performance:")
    log(res.to_string())

    log("\nPer-class detail (HGB):")
    log(classification_report(y, preds["HGB"].astype(str),
                              labels=[c for c in CLASSES if c in y.unique()],
                              digits=3, zero_division=0))
    cm_labels = [c for c in CLASSES if c in y.unique()]
    cm = confusion_matrix(y, preds["HGB"].astype(str), labels=cm_labels)
    log("Confusion matrix (rows = true, cols = predicted): " + str(cm_labels))
    log(pd.DataFrame(cm, index=cm_labels, columns=cm_labels).to_string())
    return res


# ---------------------------------------------------------------------------
# Experiment 1 -- Round 1: baseline visit only
# ---------------------------------------------------------------------------
bl = df[(df.VISIT_MONTH == 0) & df.LABEL.notna()].copy()
r1 = evaluate(X_all.loc[bl.index], bl["LABEL"], bl["PATNO"].values,
              "EXPERIMENT 1 -- Round 1: baseline visit only (one row per patient)")

# ---------------------------------------------------------------------------
# Experiment 2 -- all visits pooled (grouped CV mandatory)
# ---------------------------------------------------------------------------
allv = df[df.LABEL.notna()].copy()
r2 = evaluate(X_all.loc[allv.index], allv["LABEL"], allv["PATNO"].values,
              "EXPERIMENT 2 -- all scheduled visits pooled")

# ---------------------------------------------------------------------------
# Experiment 3 -- binary TD vs PIGD (drop the unstable Indeterminate band)
# ---------------------------------------------------------------------------
bin_df = df[df.LABEL.isin(["TD", "PIGD"]) & df.LABEL.notna()].copy()
r3 = evaluate(X_all.loc[bin_df.index], bin_df["LABEL"], bin_df["PATNO"].values,
              "EXPERIMENT 3 -- binary TD vs PIGD, all visits "
              "(Indeterminate dropped: it is a 0.90-1.15 ratio band, "
              "76% unstable visit-to-visit)")

# ---------------------------------------------------------------------------
# Experiment 4 -- leakage positive control (SANITY CHECK ONLY, NOT A RESULT)
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("EXPERIMENT 4 -- LEAKAGE POSITIVE CONTROL (not a result; diagnostic only)")
log("=" * 74)
log("Deliberately FEEDS the banned Y-defining items back in. If the pipeline is")
log("wired correctly this must score near-perfectly, confirming that the low")
log("scores above reflect a genuinely hard task rather than a broken pipeline.")
leak_feats = FEATURES + [c for c in TREMOR_ITEMS + PIGD_ITEMS if c in df.columns]
Xl = df[leak_feats].apply(pd.to_numeric, errors="coerce")
yl = allv["LABEL"]
sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
pl = np.empty(len(yl), dtype=object)
Xl_sub = Xl.loc[allv.index]
for tr, te in sgk.split(Xl_sub, yl, groups=allv["PATNO"].values):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                       learning_rate=0.06, random_state=42)
    m.fit(Xl_sub.iloc[tr], yl.iloc[tr],
          sample_weight=compute_sample_weight("balanced", yl.iloc[tr]))
    pl[te] = m.predict(Xl_sub.iloc[te])
log(f"\nWith Y-defining items included -> accuracy {accuracy_score(yl, pl.astype(str)):.3f}, "
    f"macro-F1 {f1_score(yl, pl.astype(str), average='macro'):.3f}")
log("(Expected ~1.0. This is exactly the meaningless result the strict policy avoids.)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log("\n" + "=" * 74)
log("SUMMARY -- balanced accuracy / macro-F1 vs. the majority-class baseline")
log("=" * 74)
summary = pd.DataFrame({
    "Exp 1: Round 1 (BL only)": r1["balanced_acc"],
    "Exp 2: all visits": r2["balanced_acc"],
    "Exp 3: TD vs PIGD binary": r3["balanced_acc"],
}).T
log("\nBalanced accuracy:")
log(summary.round(3).to_string())
summary_f1 = pd.DataFrame({
    "Exp 1: Round 1 (BL only)": r1["macro_F1"],
    "Exp 2: all visits": r2["macro_F1"],
    "Exp 3: TD vs PIGD binary": r3["macro_F1"],
}).T
log("\nMacro F1:")
log(summary_f1.round(3).to_string())

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print(f"\nResults written to {OUT_TXT}")
