"""
03_ablation_tdpigd.py
=====================
Ablation study on the binary TD vs PIGD task, to find out WHAT is actually
carrying the signal in the baseline model -- and whether any of it is
something we should not be relying on.

Three concerns motivated this, all raised by the baseline feature importances:

  A. GENETIC_COHORT ranked #2 (0.095). This is a PPMI RECRUITMENT ARM flag,
     not a clinical measurement. The genetic cohort is deliberately enriched
     for LRRK2/GBA/SNCA carriers, so the model may be exploiting "which study
     arm was this patient enrolled into" -- information that does not exist
     for a new patient walking into a clinic. If performance depends on it,
     the headline number is not transferable.

  B. NP2RISE (#1, 0.141) and NP2TURN (#3, 0.073) are AXIAL MOBILITY items
     (rising from a chair; turning in bed). They are not in the TD/PIGD
     formula, so they pass the strict policy -- but they measure the same
     axial-motor construct that the PIGD score measures, just via
     patient report instead of examiner rating. A reviewer may call this
     proxy leakage. We should know how much of the result rests on them.

  C. LEDD_TOTAL_MG / PDTRTMNT are medication features. Dopaminergic
     medication suppresses tremor, and prescribing is itself influenced by
     phenotype, so these carry a confounding / reverse-causality risk.

Each ablation removes one group and re-measures. Design is otherwise
identical to the baseline: StratifiedGroupKFold by PATNO, balanced class
weights from the training fold only, 5 folds.

Run:  python 03_ablation_tdpigd.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

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
    def booster():
        return XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             min_child_weight=5, random_state=42, n_jobs=-1,
                             tree_method="hist", objective="binary:logistic",
                             eval_metric="logloss")
    BACKEND = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def booster():
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    BACKEND = "HistGradientBoosting (xgboost not installed)"

df = pd.read_csv(_PROCESSED / "ppmi_tdpigd_long.csv", low_memory=False)
dd = pd.read_csv(_PROCESSED / "ppmi_tdpigd_dictionary.csv")
FEATURES = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())

AXIAL = ["NP2RISE", "NP2TURN"]
MEDS = ["LEDD_TOTAL_MG", "PDTRTMNT", "N_CONMEDS"]
COHORT_FLAG = ["GENETIC_COHORT"]
ADL_ALL = [c for c in FEATURES if c.startswith("NP2")]

b = df[df.LABEL.isin(["TD", "PIGD"])].copy()
ALL_COLS = sorted(list(set(FEATURES + COHORT_FLAG + AXIAL + MEDS + ADL_ALL)))
X_full = b[ALL_COLS].apply(pd.to_numeric, errors="coerce")
y = b["LABEL"]
groups = b["PATNO"].values
le = LabelEncoder().fit(y)
y_enc = pd.Series(le.transform(y), index=y.index)   # PIGD=0, TD=1

_lines = []
def log(m=""):
    print(m)
    _lines.append(str(m))


def run(cols, name):
    X = X_full[cols]
    oof = np.zeros(len(y)); oof_p = np.zeros(len(y))
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in sgk.split(X, y_enc, groups=groups):
        m = booster()
        m.fit(X.iloc[tr], y_enc.iloc[tr],
              sample_weight=compute_sample_weight("balanced", y_enc.iloc[tr]))
        oof[te] = m.predict(X.iloc[te])
        oof_p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return {
        "configuration": name,
        "n_feat": len(cols),
        "balanced_acc": round(balanced_accuracy_score(y_enc, oof), 3),
        "macro_F1": round(f1_score(y_enc, oof, average="macro"), 3),
        "ROC_AUC": round(roc_auc_score(y_enc, oof_p), 3),
    }


log("=" * 78)
log(f"ABLATION -- binary TD vs PIGD  |  backend: {BACKEND}")
log("=" * 78)
log(f"n = {len(y)} visit-rows, {b.PATNO.nunique()} patients "
    f"(TD {(y=='TD').sum()}, PIGD {(y=='PIGD').sum()})")
log("Chance-level balanced accuracy = 0.500")

results = [
    run(FEATURES, "FULL (all 26 clean cheap features)"),
    run(FEATURES + COHORT_FLAG, "Ref: + GENETIC_COHORT (recruitment flag)"),
    run([c for c in FEATURES if c not in AXIAL],
        "A) drop axial items (NP2RISE, NP2TURN)"),
    run([c for c in FEATURES if c not in MEDS],
        "B) drop medication features"),
    run([c for c in FEATURES if c not in AXIAL + MEDS],
        "C) drop axial and medication features"),
    run([c for c in FEATURES if c not in ADL_ALL],
        "D) drop ALL Part II ADL items (no patient-reported motor at all)"),
    run(ADL_ALL, "E) ONLY Part II ADL items"),
    run(AXIAL, "F) ONLY the 2 axial items"),
    run(COHORT_FLAG, "G) ONLY GENETIC_COHORT (sanity floor)"),
]
res = pd.DataFrame(results).set_index("configuration")
log("\n" + res.to_string())

full = res.loc["FULL (all 26 clean cheap features)", "balanced_acc"]
log("\n" + "=" * 78)
log("INTERPRETATION")
log("=" * 78)
for name, row in res.iterrows():
    if name.startswith("FULL"):
        continue
    delta = row["balanced_acc"] - full
    log(f"  {name:<62} {row['balanced_acc']:.3f}  ({delta:+.3f})")

with open(_METRICS / "ablation_results.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print("\nWritten to ablation_results.txt")
