"""Fidelity probe for Implied-Exam Explanation (IEE).
Can model-implied tremor / gait scores, computed from the SAME 26 cheap features,
reproduce the classifier's decisions?  If they can't, IEE isn't a faithful
explanation of the classifier and the idea dies here.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/processed/ppmi_tdpigd_long.csv", low_memory=False)
dd = pd.read_csv(ROOT/"data/processed/ppmi_tdpigd_dictionary.csv")
F = sorted(dd.loc[dd.bucket=="CHEAP_FEATURE","column"])
b = df[df.LABEL.isin(["TD","PIGD"])].reset_index(drop=True)
X = b[F].apply(pd.to_numeric, errors="coerce"); y = (b.LABEL=="PIGD").astype(int).to_numpy(); g = b.PATNO.to_numpy()
eps = 0.5/16
logT = np.log(b.TREMOR_SCORE + eps).to_numpy(); logP = np.log(b.PIGD_SCORE + eps).to_numpy()
p_cls = np.zeros(len(b)); lT = np.zeros(len(b)); lP = np.zeros(len(b))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g):
    c = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)
    c.fit(X.iloc[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr])); p_cls[te] = c.predict_proba(X.iloc[te])[:,1]
    rT = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42).fit(X.iloc[tr], logT[tr])
    rP = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42).fit(X.iloc[tr], logP[tr])
    lT[te] = rT.predict(X.iloc[te]); lP[te] = rP.predict(X.iloc[te])
logr_hat = lT - lP
cls_pred = (p_cls >= 0.5).astype(int)
# implied-exam decision: threshold the implied log-ratio. Binary task -> one cutoff; pick it on training folds' median
# simplest faithful choice: the cutoff that maximises agreement with the classifier (reported), and the formula cutoff
for name, cut in [("formula midpoint log(1.0)", 0.0), ("formula PIGD cutoff log(0.90)", np.log(0.90)), ("formula TD cutoff log(1.15)", np.log(1.15))]:
    iee_pred = (logr_hat <= cut).astype(int)
    print(f"{name:32s} agreement with classifier {np.mean(iee_pred==cls_pred)*100:5.1f}% | "
          f"IEE bal-acc vs truth {balanced_accuracy_score(y, iee_pred):.3f}")
best = max(np.quantile(logr_hat, np.linspace(.05,.95,91)), key=lambda c: np.mean((logr_hat<=c)==cls_pred))
iee_pred = (logr_hat <= best).astype(int)
print(f"{'best-agreement cutoff':32s} agreement with classifier {np.mean(iee_pred==cls_pred)*100:5.1f}% | IEE bal-acc {balanced_accuracy_score(y, iee_pred):.3f}")
print(f"\nclassifier bal-acc vs truth: {balanced_accuracy_score(y, cls_pred):.3f}")
print(f"rank corr (classifier P(PIGD), implied -log r): {pd.Series(p_cls).corr(pd.Series(-logr_hat), method='spearman'):.3f}")
print(f"\nVERIFIABILITY — implied vs TRUE hidden scores (out-of-fold):")
print(f"  corr(implied log T, true log T) = {np.corrcoef(lT, logT)[0,1]:.3f}")
print(f"  corr(implied log P, true log P) = {np.corrcoef(lP, logP)[0,1]:.3f}")
print(f"  corr(implied log r, true log r) = {np.corrcoef(logr_hat, logT-logP)[0,1]:.3f}")
# which channel carries the classifier's decision?
print(f"\nwhich channel does the classifier track?  corr(P(PIGD), implied log P) = {np.corrcoef(p_cls, lP)[0,1]:+.3f} | "
      f"corr(P(PIGD), implied log T) = {np.corrcoef(p_cls, lT)[0,1]:+.3f}")
