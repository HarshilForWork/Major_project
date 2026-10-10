"""Model ceiling check (1 Oct 2026): can the binary TD-vs-PIGD model go past ~0.70?

Dev patients only (locked test untouched), grouped 5-fold CV, nothing logged.
Run:  python notebooks/model_ceiling_check.py      (~1 min)
Result summary: HANDOFF_AR1_SHAP.md section 7.

Which levers can move the binary TD-vs-PIGD model? Dev patients only, grouped 5-fold CV.
All experiments are scored on the SAME rows: labelled TD/PIGD visits that have a previous
scheduled visit (the project's longitudinal scope), unless stated."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from trace_pd import config
from trace_pd.evaluation.splits import split_dev_test
from trace_pd.explain import ar1, formula as FM
from trace_pd.models.backend import make_classifier, fit_classifier

df = pd.read_csv(config.LONG_TABLE, low_memory=False)
dd = pd.read_csv(config.DICTIONARY)
F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
df[F] = df[F].apply(pd.to_numeric, errors="coerce")
dev, _ = split_dev_test(df)
dev = dev.sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True)
_, _, DYN = ar1.feature_roles(F)

# ---- history features (no labels, no exam items) ---------------------------------
P = ar1.build_pairs(dev, F)
P = P.merge(dev[["PATNO", "VISIT_MONTH", "PDSTATE_USED", "TREMOR_SCORE", "PIGD_SCORE"]],
            on=["PATNO", "VISIT_MONTH"], how="left")
g = dev.groupby("PATNO")
hist = pd.DataFrame({"PATNO": dev.PATNO, "VISIT_MONTH": dev.VISIT_MONTH})
for f in DYN:
    s = dev[f]
    cs = s.fillna(0).groupby(dev.PATNO).cumsum(); cn = s.notna().groupby(dev.PATNO).cumsum()
    hist["MEAN_" + f] = (cs / cn.replace(0, np.nan))                  # mean of visits up to t
    hist["MAX_" + f] = s.groupby(dev.PATNO).cummax()
hist["VISIT_IDX"] = g.cumcount()
# previous visit's exam-derived label (the exam WAS done at t-1 in PPMI)
hist["PREV_LABEL_PIGD"] = (g.LABEL.shift(1) == "PIGD").astype(float).where(g.LABEL.shift(1).isin(["TD", "PIGD"]))
lT, lP = FM.log_scores(dev.TREMOR_SCORE, dev.PIGD_SCORE); lT, lP = pd.Series(lT), pd.Series(lP)
hist["PREV_ELL"] = (lT - lP).where(dev.LABEL.notna()).groupby(dev.PATNO).shift(1)
# smoothed ("denoised") target: mean log-ratio over t-1, t, t+1
ell = (lT - lP).where(dev.LABEL.notna())
sm = pd.concat([ell.groupby(dev.PATNO).shift(1), ell, ell.groupby(dev.PATNO).shift(-1)], axis=1).mean(axis=1)
hist["SMOOTH_ZONE"] = pd.Series(FM.zone(sm)).where(sm.notna())
P = P.merge(hist, on=["PATNO", "VISIT_MONTH"], how="left")

base = P[P.LABEL.isin(["TD", "PIGD"])].reset_index(drop=True)
y = (base.LABEL == "PIGD").astype(int).to_numpy()
grp = base.PATNO.to_numpy()

PREV = ["PREV_" + f for f in DYN]
DELTA = []
for f in DYN:
    base["DELTA_" + f] = base[f] - base["PREV_" + f]; DELTA.append("DELTA_" + f)
MEAN = ["MEAN_" + f for f in DYN]; MAXF = ["MAX_" + f for f in DYN]


def cv(cols, rows=None, target=None, model="xgb"):
    r = np.ones(len(base), bool) if rows is None else rows
    yy = (y if target is None else target)[r]; X = base.loc[r, cols].to_numpy(float); gg = grp[r]
    ok = ~np.isnan(yy.astype(float)); X, yy, gg = X[ok], yy[ok].astype(int), gg[ok]
    pred = np.zeros(len(yy)); prob = np.zeros(len(yy))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, yy, gg):
        m = fit_classifier(make_classifier(model, 2), X[tr], yy[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]; pred[te] = (prob[te] >= 0.5)
    return balanced_accuracy_score(yy, pred), roc_auc_score(yy, prob), len(yy)


def show(name, res):
    ba, auc, n = res
    print(f"{name:<62} bal.acc {ba:.3f}  AUROC {auc:.3f}  n={n}", flush=True)


print(f"rows: {len(base)} visits with a previous visit, {base.PATNO.nunique()} dev patients\n")
print("A. CHEAP FEATURES ONLY (deployable as the project defines it)")
show("A0 current visit, 26 features (project baseline)", cv(F))
show("A1 + previous visit values", cv(F + PREV))
show("A2 + previous values + changes since last visit", cv(F + PREV + DELTA))
show("A3 + patient's running mean and worst-so-far", cv(F + MEAN + MAXF + ["VISIT_IDX"]))
show("A4 everything in A1-A3", cv(F + PREV + DELTA + MEAN + MAXF + ["VISIT_IDX"]))

print("\nB. CLEANER LABELS (same features as A0)")
off = (base.PDSTATE_USED == "OFF").to_numpy()
show("B1 only visits examined OFF medication", cv(F, rows=off))
show("B2 A4 features, OFF-medication visits only", cv(F + PREV + DELTA + MEAN + MAXF + ["VISIT_IDX"], rows=off))
smz = base.SMOOTH_ZONE.to_numpy()
sm_rows = np.isin(smz, [0, 2])
sm_y = np.where(smz == 0, 1, np.where(smz == 2, 0, np.nan))
show("B3 smoothed target (avg of t-1,t,t+1 ratio), A0 features", cv(F, rows=sm_rows, target=sm_y))
show("B4 smoothed target, A4 features", cv(F + PREV + DELTA + MEAN + MAXF + ["VISIT_IDX"], rows=sm_rows, target=sm_y))

print("\nC. USING THE PREVIOUS VISIT'S EXAM (not cheap-only: needs the exam at t-1)")
show("C1 previous label only", cv(["PREV_LABEL_PIGD"]))
show("C2 previous exam ratio only", cv(["PREV_ELL"]))
show("C3 A4 + previous exam ratio", cv(F + PREV + DELTA + MEAN + MAXF + ["VISIT_IDX", "PREV_ELL"]))

print("\nD. CEILING: how well does the real exam predict itself at the next visit?")
r = base.PREV_LABEL_PIGD.notna().to_numpy()
print(f"   label at t-1 used as the prediction for t: bal.acc "
      f"{balanced_accuracy_score(y[r], base.PREV_LABEL_PIGD.to_numpy()[r].astype(int)):.3f}  (n={r.sum()})")
