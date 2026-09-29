"""Can C3 (proximity / drift / noise) be validated?  Three experiments.

C3 could not be validated against the cheap-feature transition model because that model
is near chance (AUROC ~0.53): a correct explanation of a model that knows nothing also
scores 0.5 on every correctness test. This script builds black boxes WITH signal, and a
planted-mechanism check.

  E1  CHEAP            transition model on cheap features (the deployment setting) -- reference
  E2  EXAM-INFORMED    the full exam WAS done at visit t ("will today's subtype hold?");
                       today's tremor / gait scores and exam medication state are inputs.
                       Not leakage: the target is the NEXT visit's label.
  E3  SUSTAINED        as E2, but the target is a SUSTAINED flip (the new label still holds
                       one visit later) instead of any flip
  E4  PLANTED          black boxes trained on ONE mechanism's features only:
                         proximity-only (today's ratio & distance to cutoff)
                         drift-only     (between-visit deltas, visit count)
                         noise-only     (medication / exam-state features)
                       PASS if C3 loads each planted model mainly on the planted component.

For each black box: AUROC, C3 fidelity (R^2 of logit), component shares, and the
ground-truth test -- telling noise flips from drift flips -- vs grouped Shapley on the
black box.

Run:  python src/trace_pd/evaluation/validate_c3.py [--backend xgb|hgb]
Output: reports/metrics/c3_validation[_xgb].txt
"""
from pathlib import Path as _Path
import sys
_ROOT = _Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from trace_pd.explain import formula as FM
from trace_pd.explain.shapley import shapley_mc
from trace_pd.explain.pdn import PDNExplainer, logit
from trace_pd.models.backend import make_classifier, fit_classifier, backend_from_argv

BACKEND = backend_from_argv(sys.argv)
PROC, MET = _ROOT / "data/processed", _ROOT / "reports/metrics"
SEED, N_BG, N_PERM = 42, 100, 16
LOG = []
def log(s=""): print(s, flush=True); LOG.append(s)
def fit_clf(X, y): return fit_classifier(make_classifier(BACKEND, n_classes=len(np.unique(y))), X, y)

# ------------------------------------------------------------------ data
df = pd.read_csv(PROC / "ppmi_tdpigd_long.csv", low_memory=False).sort_values(["PATNO", "VISIT_MONTH"])
dd = pd.read_csv(PROC / "ppmi_tdpigd_dictionary.csv")
FEATS = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
df[FEATS] = df[FEATS].apply(pd.to_numeric, errors="coerce")
df["logT"], df["logP"] = FM.log_scores(df.TREMOR_SCORE, df.PIGD_SCORE)
df["ell"] = df.logT - df.logP
d = df[df.LABEL.notna()].copy().reset_index(drop=True)
g = d.groupby("PATNO")
d["eta"] = (g.ell.shift(1) + g.ell.shift(-1)) / 2                      # leave-visit-out smoothing
d["noise_abs"] = (d.ell - d.eta).abs()
d["eta_next"] = g.eta.shift(-1); d["mu"] = d.eta_next - d.eta
DC = ["NP2RISE", "NP2TURN", "LEDD_TOTAL_MG", "MCATOT", "NP1RTOT", "NP1PTOT", "GDS_TOTAL", "SCOPA_AUT_TOTAL"]
for c in DC: d["D_" + c] = d[c] - g[c].shift(1)
d["PRIOR_VISITS"] = g.cumcount()
d["DIST_CUT"] = FM.distance_to_cutoff(d.ell)
d["STATE_ON"] = (d.PDSTATE_USED == "ON").astype(float).where(d.PDSTATE_USED.notna())
d["L2"] = g.LABEL.shift(-2)
d["SUSTAINED"] = np.where(d.LABEL_FLIPPED_NEXT.isna() | d.L2.isna(), np.nan,
                          ((d.LABEL_FLIPPED_NEXT == 1) & (d.L2 == d.NEXT_LABEL)).astype(float))
# leak-free upstream subtype probabilities
Xa = d[FEATS].to_numpy(float); ya = d.LABEL.map({"TD": 0, "PIGD": 1, "INDETERMINATE": 2}).to_numpy()
oof = np.zeros((len(d), 3))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xa, ya, d.PATNO):
    oof[te] = fit_clf(Xa[tr], ya[tr]).predict_proba(Xa[te])
d["P_TD"], d["P_PIGD"], d["P_IND"] = oof.T
sp = np.sort(oof, 1); d["P_MARGIN"] = sp[:, -1] - sp[:, -2]

UP = ["P_TD", "P_PIGD", "P_IND", "P_MARGIN"]
DRIFT_F = ["D_" + c for c in DC] + ["PRIOR_VISITS"]
NOISE_F = ["LEDD_TOTAL_MG", "N_CONMEDS", "PDTRTMNT", "D_LEDD_TOTAL_MG"]
EXAM_F = ["ell", "logT", "logP", "DIST_CUT", "STATE_ON"]
CHEAP_SET = FEATS + DRIFT_F + UP
EXAM_SET = CHEAP_SET + EXAM_F

# ------------------------------------------------------------------ one experiment
def run(name, feats, target, head_feats=None, groups_def=None):
    """Train black box on `feats`; C3 heads on `head_feats` (default = feats)."""
    head_feats = feats if head_feats is None else head_feats
    t = d[d[target].notna()].reset_index(drop=True)
    X, XH = t[feats].to_numpy(float), t[head_feats].to_numpy(float)
    y = t[target].astype(int).to_numpy(); G = t.PATNO.to_numpy()
    gp, fid = np.zeros(len(t)), np.zeros(len(t))
    phi = np.zeros((len(t), 3)); base_phi = np.full((len(t), 3), np.nan); W = []
    if groups_def is None:
        groups_def = dict(P=[c for c in feats if c not in DRIFT_F + NOISE_F],
                          D=[c for c in feats if c in DRIFT_F], N=[c for c in feats if c in NOISE_F])
    grp = [[feats.index(c) for c in groups_def[k]] for k in "PDN"]
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, G)):
        m = fit_clf(X[tr], y[tr]); gp[te] = m.predict_proba(X[te])[:, 1]
        pdn = PDNExplainer().fit(XH[tr], t.eta.to_numpy()[tr], t.mu.to_numpy()[tr], t.noise_abs.to_numpy()[tr])
        pdn.calibrate_fidelity(XH[tr], m.predict_proba(X[tr])[:, 1]); W.append(pdn.b)
        e = pdn.explain(XH[te]); fid[te] = e["full"]
        phi[te] = np.c_[e["phi_P"], e["phi_D"], e["phi_N"]]
        if all(len(q) for q in grp):
            bg = X[np.random.default_rng(fold).choice(tr, N_BG, replace=False)]
            f = lambda Z, mm=m: logit(mm.predict_proba(Z)[:, 1])
            base_phi[te], _ = shapley_mc(f, X[te], bg, n_perm=N_PERM, groups=grp, rng=np.random.default_rng(fold))
    z = logit(gp)
    auc = roc_auc_score(y, gp); r2 = 1 - np.var(z - fid) / np.var(z)
    sh = np.abs(phi).mean(0); sh = sh / sh.sum() * 100
    fl = (y == 1) & t.eta.notna().to_numpy() & t.eta_next.notna().to_numpy()
    noise_flip = (FM.zone(t.eta.to_numpy()[fl]) == FM.zone(t.eta_next.to_numpy()[fl]))
    au_fcx = roc_auc_score(noise_flip, phi[fl, 0] + phi[fl, 2] - phi[fl, 1]) if noise_flip.any() and (~noise_flip).any() else np.nan
    au_b = (roc_auc_score(noise_flip, base_phi[fl, 0] + base_phi[fl, 2] - base_phi[fl, 1])
            if np.isfinite(base_phi[fl]).all() and noise_flip.any() else np.nan)
    wmean = {c: np.mean([w[c] for w in W]) for c in "PDN"}
    log(f"\n[{name}]  n={len(t):,}  positives {y.mean()*100:.1f}%")
    log(f"  black-box AUROC            {auc:.3f}")
    log(f"  C3 fidelity                R^2 {r2:.3f} | corr {np.corrcoef(z, fid)[0,1]:.3f}")
    log(f"  component weights b_P/b_D/b_N  {wmean['P']:+.2f} / {wmean['D']:+.2f} / {wmean['N']:+.2f}")
    log(f"  risk made of               proximity {sh[0]:.0f}% | drift {sh[1]:.0f}% | noise {sh[2]:.0f}%")
    log(f"  noise-vs-drift flips (AUROC, {fl.sum()} flips, {noise_flip.mean()*100:.0f}% noise):  "
        f"C3 {au_fcx:.3f} | grouped Shapley {au_b:.3f}")
    return dict(auc=auc, r2=r2, shares=sh, w=wmean, au_fcx=au_fcx, au_b=au_b)

log("=" * 78); log(f"C3 VALIDATION  |  black box: {'XGBoost' if BACKEND == 'xgb' else 'sklearn HGB'}"); log("=" * 78)
r1 = run("E1 CHEAP  -- deployment setting, any flip", CHEAP_SET, "LABEL_FLIPPED_NEXT")
r2 = run("E2 EXAM-INFORMED  -- full exam done at visit t, any flip", EXAM_SET, "LABEL_FLIPPED_NEXT")
r3 = run("E3 EXAM-INFORMED  -- SUSTAINED flip", EXAM_SET, "SUSTAINED")

log("\n" + "-" * 78); log("E4 PLANTED MECHANISMS  (heads always see the full exam-informed set)"); log("-" * 78)
PLANT = {"proximity-only": ["ell", "DIST_CUT", "logT", "logP"],
         "drift-only": DRIFT_F,
         "noise-only": NOISE_F + ["STATE_ON"]}
res4 = {}
for k, fs in PLANT.items():
    res4[k] = run(f"E4 planted {k}", fs, "LABEL_FLIPPED_NEXT", head_feats=EXAM_SET,
                  groups_def=dict(P=fs if k == "proximity-only" else [], D=fs if k == "drift-only" else [],
                                  N=fs if k == "noise-only" else []))
log("\nPLANTED TEST -- share of explained risk per component (rows = planted mechanism):")
tab = pd.DataFrame({k: v["shares"] for k, v in res4.items()}, index=["proximity", "drift", "noise"]).T.round(0)
log(tab.to_string())
diag = [tab.loc["proximity-only", "proximity"], tab.loc["drift-only", "drift"], tab.loc["noise-only", "noise"]]
col_best = [tab["proximity"].idxmax() == "proximity-only", tab["drift"].idxmax() == "drift-only",
            tab["noise"].idxmax() == "noise-only"]
log(f"each component's share is highest under its own planted model: {sum(col_best)}/3  -> "
    f"{'PASS' if all(col_best) else 'PARTIAL' if any(col_best) else 'FAIL'}")

log("\n" + "=" * 78); log("SUMMARY"); log("=" * 78)
log(f"{'':44s}{'AUROC':>7s}{'C3 R^2':>8s}{'C3 noise/drift':>16s}{'baseline':>10s}")
for nm, r in [("E1 cheap, any flip", r1), ("E2 exam-informed, any flip", r2), ("E3 exam-informed, sustained", r3)]:
    log(f"{nm:44s}{r['auc']:7.3f}{r['r2']:8.3f}{r['au_fcx']:16.3f}{r['au_b']:10.3f}")
OUT = MET / ("c3_validation_xgb.txt" if BACKEND == "xgb" else "c3_validation.txt")
OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8"); print(f"\nwrote {OUT}")
