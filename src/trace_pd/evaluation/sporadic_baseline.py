"""Hardest honest test: sporadic PD only, baseline visit only.

Removes the two things that inflate the pooled numbers:
  * GENETIC_COHORT -- a recruitment-arm flag that alone scores 0.629
  * repeated visits -- ~17 correlated rows per patient

What remains is the actual clinical use case: a new sporadic patient, first
visit, routine data only. Binary TD vs PIGD (Indeterminate dropped).
"""
from pathlib import Path as _Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

_ROOT = _Path(__file__).resolve().parents[3]
_PROCESSED = _ROOT / "data" / "processed"
_METRICS = _ROOT / "reports" / "metrics"
_METRICS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(_PROCESSED / "ppmi_tdpigd_long.csv", low_memory=False)
dd = pd.read_csv(_PROCESSED / "ppmi_tdpigd_dictionary.csv")
FEATURES = sorted(c for c in dd.loc[dd.bucket == "CHEAP_FEATURE", "column"]
                  if c != "GENETIC_COHORT")

sub = df[(df.GENETIC_COHORT == 0) & (df.VISIT_MONTH == 0)
         & df.LABEL.isin(["TD", "PIGD"])].copy()
# MoCA is an annual-visit instrument and is 0% present at baseline for this
# subset -- drop any column with no observed values rather than impute nothing.
empty = [c for c in FEATURES if sub[c].notna().sum() == 0]
FEATURES = [c for c in FEATURES if c not in empty]
X, y = sub[FEATURES], (sub.LABEL == "PIGD").astype(int)

out = [f"SPORADIC PD, BASELINE VISIT ONLY -- binary TD vs PIGD",
       f"n = {len(sub)} patients (TD {int((y==0).sum())}, PIGD {int((y==1).sum())}), "
       f"{len(FEATURES)} features (GENETIC_COHORT removed; all-missing dropped: {empty})",
       "CV: StratifiedKFold x5 (one row per patient, so grouping is implicit)",
       "chance balanced accuracy = 0.500", ""]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name in ["LOGISTIC", "HGB"]:
    ba, f1 = [], []
    for tr, te in skf.split(X, y):
        if name == "HGB":
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               learning_rate=0.06, random_state=42)
            m.fit(X.iloc[tr], y.iloc[tr],
                  sample_weight=compute_sample_weight("balanced", y.iloc[tr]))
        else:
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        ba.append(balanced_accuracy_score(y.iloc[te], p))
        f1.append(f1_score(y.iloc[te], p, average="macro"))
    out.append(f"{name:9s} balanced_acc {np.mean(ba):.3f} ± {np.std(ba):.3f}   "
               f"macro_F1 {np.mean(f1):.3f}")

text = "\n".join(out)
print(text)
(_METRICS / "sporadic_baseline_results.txt").write_text(text + "\n", encoding="utf-8")
