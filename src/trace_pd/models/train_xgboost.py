"""
02_baseline_model_xgb.py
========================
XGBoost version of the TD / PIGD / Indeterminate baseline.

Identical experimental design to 02_baseline_model_tdpigd.py, but uses
XGBClassifier as the boosted model. Handles the two things XGBoost needs
that sklearn does not:

  1. LABEL ENCODING -- XGBoost requires integer class labels, so LabelEncoder
     is fitted once on the full label set and predictions are decoded back to
     the original strings before scoring. Encoding once (not per fold) keeps
     the class<->integer mapping stable across folds.
  2. SAMPLE WEIGHTS -- passed per fold via sample_weight, computed on the
     TRAINING fold only, so no test-fold class information leaks into the
     weighting.

Falls back to sklearn HistGradientBoostingClassifier automatically if xgboost
is not importable, so the script runs either way.

STRICT FEATURE POLICY (enforced)
--------------------------------
No column that participates in computing Y may be used as an input. All 16
Y-defining MDS-UPDRS items are excluded, INCLUDING the three patient-reported
Part II items (NP2TRMR, NP2WALK, NP2FREZ), plus NP3TOT, NP2PTOT and NHY which
encode them indirectly. The script asserts this and aborts on violation.

Run:  python 02_baseline_model_xgb.py
Requires: pandas, numpy, scikit-learn, xgboost (optional)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
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


try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

LONG = str(_PROCESSED / "ppmi_tdpigd_long.csv")
DICT = str(_PROCESSED / "ppmi_tdpigd_dictionary.csv")
OUT_TXT = str(_METRICS / "baseline_model_results_xgb.txt")

TREMOR_ITEMS = ["NP2TRMR", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
                "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ",
                "NP3RTCON"]
PIGD_ITEMS = ["NP2WALK", "NP2FREZ", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"]
CLASSES = ["TD", "PIGD", "INDETERMINATE"]

_lines = []


def log(m=""):
    print(m)
    _lines.append(str(m))


def make_booster(n_classes):
    """Boosted model, XGBoost if available, else sklearn equivalent."""
    if HAVE_XGB:
        params = dict(max_depth=3, n_estimators=300, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      min_child_weight=5, random_state=42, n_jobs=-1,
                      tree_method="hist")
        if n_classes > 2:
            params.update(objective="multi:softprob", num_class=n_classes,
                          eval_metric="mlogloss")
        else:
            params.update(objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(**params)
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=300, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42)


df = pd.read_csv(LONG, low_memory=False)
dd = pd.read_csv(DICT)

FEATURES = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
banned = set(TREMOR_ITEMS) | set(PIGD_ITEMS) | {"NP3TOT", "NP2PTOT", "NHY"}
violation = banned & set(FEATURES)
assert not violation, f"STRICT POLICY VIOLATED: {violation}"

X_all = df[FEATURES].copy()
for c in X_all.columns:
    if X_all[c].dtype == object:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

log("=" * 74)
log(f"BASELINE MODEL -- {'XGBoost' if HAVE_XGB else 'HistGradientBoosting (xgboost not found)'}")
log("=" * 74)
log(f"Strict policy verified: none of the 16 Y-defining items are inputs.")
log(f"Features ({len(FEATURES)}): {FEATURES}")


def evaluate(X, y, groups, title, n_splits=5):
    log("\n" + "=" * 74)
    log(title)
    log("=" * 74)

    # Encode labels ONCE so the integer mapping is identical in every fold.
    le = LabelEncoder().fit(y)
    y_enc = pd.Series(le.transform(y), index=y.index)
    n_classes = len(le.classes_)
    log(f"n = {len(y)} rows, {len(np.unique(groups))} patients, "
        f"{n_classes} classes: {list(le.classes_)}")
    log("Class distribution: " +
        ", ".join(f"{k} {v} ({v/len(y)*100:.1f}%)" for k, v in y.value_counts().items()))

    if len(np.unique(groups)) == len(y):
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=42).split(X, y_enc)
        log("CV: StratifiedKFold (one row per patient)")
    else:
        split_iter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                          random_state=42).split(X, y_enc, groups=groups)
        log("CV: StratifiedGroupKFold grouped by PATNO (prevents patient leakage)")

    preds = {k: np.empty(len(y), dtype=object) for k in ["BOOST", "LOGISTIC", "MAJORITY"]}

    for tr, te in split_iter:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr_enc, ytr_raw = y_enc.iloc[tr], y.iloc[tr]
        # weights from the TRAINING fold only
        w = compute_sample_weight("balanced", ytr_enc)

        booster = make_booster(n_classes)
        booster.fit(Xtr, ytr_enc, sample_weight=w)
        preds["BOOST"][te] = le.inverse_transform(booster.predict(Xte).astype(int))

        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
        lr.fit(Xtr, ytr_raw)
        preds["LOGISTIC"][te] = lr.predict(Xte)

        preds["MAJORITY"][te] = ytr_raw.value_counts().idxmax()

    rows = []
    for name in ["MAJORITY", "LOGISTIC", "BOOST"]:
        p = preds[name].astype(str)
        rows.append({"model": name,
                     "accuracy": accuracy_score(y, p),
                     "balanced_acc": balanced_accuracy_score(y, p),
                     "macro_F1": f1_score(y, p, average="macro"),
                     "weighted_F1": f1_score(y, p, average="weighted")})
    res = pd.DataFrame(rows).set_index("model").round(3)
    log("\nOut-of-fold performance:")
    log(res.to_string())

    lbls = [c for c in CLASSES if c in y.unique()]
    log("\nPer-class detail (boosted model):")
    log(classification_report(y, preds["BOOST"].astype(str), labels=lbls,
                              digits=3, zero_division=0))
    cm = confusion_matrix(y, preds["BOOST"].astype(str), labels=lbls)
    log("Confusion matrix (rows = true, cols = predicted):")
    log(pd.DataFrame(cm, index=lbls, columns=lbls).to_string())

    # Feature importance, where the model exposes it
    fitted = make_booster(n_classes)
    fitted.fit(X, y_enc, sample_weight=compute_sample_weight("balanced", y_enc))
    if hasattr(fitted, "feature_importances_"):
        imp = (pd.Series(fitted.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(12))
        log("\nTop 12 features by importance (fitted on all rows, indicative only):")
        log(imp.round(4).to_string())
    return res


bl = df[(df.VISIT_MONTH == 0) & df.LABEL.notna()]
r1 = evaluate(X_all.loc[bl.index], bl["LABEL"], bl["PATNO"].values,
              "EXPERIMENT 1 -- Round 1: baseline visit only")

allv = df[df.LABEL.notna()]
r2 = evaluate(X_all.loc[allv.index], allv["LABEL"], allv["PATNO"].values,
              "EXPERIMENT 2 -- all scheduled visits pooled")

binm = df[df.LABEL.isin(["TD", "PIGD"])]
r3 = evaluate(X_all.loc[binm.index], binm["LABEL"], binm["PATNO"].values,
              "EXPERIMENT 3 -- binary TD vs PIGD (Indeterminate dropped)")

log("\n" + "=" * 74)
log("SUMMARY -- balanced accuracy (3-class chance = 0.333, binary = 0.500)")
log("=" * 74)
log(pd.DataFrame({"Exp 1: Round 1 (BL only)": r1["balanced_acc"],
                  "Exp 2: all visits": r2["balanced_acc"],
                  "Exp 3: TD vs PIGD": r3["balanced_acc"]}).T.round(3).to_string())
log("\nMacro F1:")
log(pd.DataFrame({"Exp 1: Round 1 (BL only)": r1["macro_F1"],
                  "Exp 2: all visits": r2["macro_F1"],
                  "Exp 3: TD vs PIGD": r3["macro_F1"]}).T.round(3).to_string())

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print(f"\nResults written to {OUT_TXT}")
