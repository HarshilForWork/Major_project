"""Feasibility probe: Definitional vs Informational Ambiguity (DIA).

Question: when the conformal set is ambiguous ({TD, PIGD}), is that because the
cheap features lack information (a full exam would settle it), or because the
patient's TRUE item scores sit so close to the Stebbins threshold that even a
repeat full exam would give a different label (the ambiguity is in the label)?

We can check this because we hold the true 16 item scores.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/ppmi_tdpigd_long.csv", low_memory=False)
dd = pd.read_csv(ROOT / "data/processed/ppmi_tdpigd_dictionary.csv")
FEATS = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
T = ["NP2TRMR","NP3PTRMR","NP3PTRML","NP3KTRMR","NP3KTRML","NP3RTARU","NP3RTALU","NP3RTARL","NP3RTALL","NP3RTALJ","NP3RTCON"]
P = ["NP2WALK","NP2FREZ","NP3GAIT","NP3FRZGT","NP3PSTBL"]
d = df[df.LABEL.notna()].reset_index(drop=True)
H = d[T + P].to_numpy(float)
rng = np.random.default_rng(0)

def stebbins(Hm):                               # vectorised label: 0 TD, 1 PIGD, 2 IND
    t, p = Hm[..., :11].mean(-1), Hm[..., 11:].mean(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p > 0, t / p, np.where(t > 0, np.inf, np.nan))
    lab = np.full(r.shape, 2)
    lab[r >= 1.15] = 0; lab[r <= 0.90] = 1
    return lab
MAP = {"TD": 0, "PIGD": 1, "INDETERMINATE": 2}
y3 = d.LABEL.map(MAP).to_numpy()
assert (stebbins(H) == y3).all(), "formula re-implementation mismatch"

# ---- 1. fragility rho under an item-level rater-noise kernel ----------------
def fragility(q, B=300):
    shift = rng.choice([-1, 1], size=(B,) + H.shape) * (rng.random((B,) + H.shape) < q)
    Hn = np.clip(H[None] + shift, 0, 4)
    return (stebbins(Hn) != y3[None]).mean(0)
rho = {q: fragility(q) for q in (0.1, 0.2, 0.3)}
one_pt = np.zeros(len(H), bool)                  # any single +-1 change flips the label
for j in range(16):
    for s in (-1, 1):
        Hn = H.copy(); Hn[:, j] = np.clip(Hn[:, j] + s, 0, 4)
        one_pt |= stebbins(Hn) != y3
print(f"visits: {len(d)} | one-point fragile: {one_pt.mean()*100:.1f}%")
for q, r in rho.items():
    print(f"  rater noise q={q}: mean rho {r.mean():.3f} | rho>=0.25: {(r>=.25).mean()*100:.1f}% | rho>=0.5: {(r>=.5).mean()*100:.1f}%")

# ---- 2. OOF LAC conformal sets on binary TD vs PIGD --------------------------
b = d[d.LABEL.isin(["TD", "PIGD"])].copy(); bi = b.index.to_numpy()
X = b[FEATS].apply(pd.to_numeric, errors="coerce"); y = (b.LABEL == "PIGD").astype(int).to_numpy()
g = b.PATNO.to_numpy(); size = np.zeros(len(b), int); cover = np.zeros(len(b), bool); p1 = np.zeros(len(b))
outer = StratifiedGroupKFold(5, shuffle=True, random_state=42)
for tr_cal, te in outer.split(X, y, g):
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=7)
    a, c = next(inner.split(X.iloc[tr_cal], y[tr_cal], g[tr_cal])); tr, cal = tr_cal[a], tr_cal[c]
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)
    m.fit(X.iloc[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    pc = m.predict_proba(X.iloc[cal]); s = 1 - pc[np.arange(len(cal)), y[cal]]
    n = len(s); qhat = np.sort(s)[min(n, int(np.ceil((n + 1) * 0.9))) - 1]
    pt = m.predict_proba(X.iloc[te]); inset = pt >= 1 - qhat
    inset[inset.sum(1) == 0, pt[inset.sum(1) == 0].argmax(1)] = True
    size[te] = inset.sum(1); cover[te] = inset[np.arange(len(te)), y[te]]; p1[te] = pt[:, 1]
print(f"\nLAC binary: coverage {cover.mean()*100:.1f}% | singleton {np.mean(size==1)*100:.1f}% | ambiguous {np.mean(size==2)*100:.1f}%")

# ---- 3. does ambiguity track fragility? --------------------------------------
r = rho[0.2][bi]; tert = pd.qcut(r.rank(method="first") if hasattr(r,"rank") else pd.Series(r).rank(method="first"), 3, labels=["low", "mid", "high"])
t = pd.DataFrame({"tertile": tert, "ambig": size == 2, "cover": cover, "rho": r, "onept": one_pt[bi]})
print("\nBy fragility tertile (q=0.2):")
print(t.groupby("tertile", observed=True).agg(rho_mean=("rho","mean"), ambiguous_pct=("ambig", lambda x: x.mean()*100),
                               coverage_pct=("cover", lambda x: x.mean()*100), n=("rho","size")).round(3).to_string())
amb = size == 2
print(f"\nAmong AMBIGUOUS sets: definitional (rho>=0.25) {np.mean(r[amb]>=.25)*100:.1f}% | informational {np.mean(r[amb]<.25)*100:.1f}%")
print(f"Among SINGLETON sets: rho>=0.25 {np.mean(r[~amb]>=.25)*100:.1f}%   <- confident calls on labels a repeat exam could flip")
print(f"Singleton accuracy: robust (rho<.25) {cover[(~amb)&(r<.25)].mean()*100:.1f}% vs fragile (rho>=.25) {cover[(~amb)&(r>=.25)].mean()*100:.1f}%")

# ---- 4. can fragility be predicted from cheap features alone? ----------------
Xa = d[FEATS].apply(pd.to_numeric, errors="coerce"); z = (rho[0.2] >= 0.25).astype(int); ga = d.PATNO.to_numpy()
oof = np.zeros(len(d))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xa, z, ga):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)
    m.fit(Xa.iloc[tr], z[tr], sample_weight=compute_sample_weight("balanced", z[tr])); oof[te] = m.predict_proba(Xa.iloc[te])[:, 1]
print(f"\nPredict fragility (rho>=0.25, base {z.mean()*100:.1f}%) from 26 cheap features: AUROC {roc_auc_score(z, oof):.3f}")

# ---- 5. are label flips explained by fragility? ------------------------------
f = d.LABEL_FLIPPED_NEXT.notna().to_numpy(); yf = d.LABEL_FLIPPED_NEXT.to_numpy()[f].astype(int)
print(f"\nNext-visit flip (n={f.sum()}, base {yf.mean()*100:.1f}%):")
print(f"  flip rate | one-point fragile {yf[one_pt[f]].mean()*100:.1f}%  vs robust {yf[~one_pt[f]].mean()*100:.1f}%")
for q in (0.1, 0.2, 0.3):
    print(f"  AUROC of rho(q={q}) ALONE for predicting the flip: {roc_auc_score(yf, rho[q][f]):.3f}")
print("  (for reference: the 40-feature transition model scored ~0.65)")
