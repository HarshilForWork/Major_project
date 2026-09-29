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
from trace_pd.explain import trajectory as TJ
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND = backend_from_argv(sys.argv)
PROC, MET = _ROOT / "data/processed", _ROOT / "reports/metrics"
SEED, N_BG, N_PERM = 42, 100, 16
LOG = []
def log(s=""): print(s, flush=True); LOG.append(s)
def fit_clf(X, y, balanced=True):
    return fit_classifier(make_classifier(BACKEND, n_classes=len(np.unique(y))), X, y, balanced=balanced)

# ------------------------------------------------------------------ data
df = pd.read_csv(PROC / "ppmi_tdpigd_long.csv", low_memory=False)
dd = pd.read_csv(PROC / "ppmi_tdpigd_dictionary.csv")
FEATS = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
df[FEATS] = df[FEATS].apply(pd.to_numeric, errors="coerce")
d = TJ.build(df)                        # all shifts over the full scheduled sequence
DC = TJ.DELTA_COLS
# leak-free upstream subtype probabilities
Xa = d[FEATS].to_numpy(float); ya = d.LABEL.map({"TD": 0, "PIGD": 1, "INDETERMINATE": 2}).to_numpy()
oof = np.zeros((len(d), 3))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xa, ya, d.PATNO):
    oof[te] = fit_clf(Xa[tr], ya[tr]).predict_proba(Xa[te])
d["P_TD"], d["P_PIGD"], d["P_IND"] = oof.T
sp = np.sort(oof, 1); d["P_MARGIN"] = sp[:, -1] - sp[:, -2]

UP = ["P_TD", "P_PIGD", "P_IND", "P_MARGIN"]
NOISE_F = ["LEDD_TOTAL_MG", "N_CONMEDS", "PDTRTMNT", "D_LEDD_TOTAL_MG", "STATE_ON"]   # medication / exam state
DRIFT_F = [c for c in ["D_" + c for c in DC] + ["PRIOR_VISITS"] if c not in NOISE_F]
EXAM_F = ["ell", "logT", "logP", "DIST_CUT", "STATE_ON"]
CHEAP_SET = FEATS + DRIFT_F + ["D_LEDD_TOTAL_MG"] + UP
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
    flat = sum(groups_def.values(), [])
    assert len(flat) == len(set(flat)), "baseline groups overlap"
    grp = [[feats.index(c) for c in groups_def[k]] for k in "PDN"]
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, G)):
        m = fit_clf(X[tr], y[tr], balanced=False); gp[te] = m.predict_proba(X[te])[:, 1]
        pdn = PDNExplainer().fit(XH[tr], t.eta.to_numpy()[tr], t.mu.to_numpy()[tr], t.noise_abs.to_numpy()[tr],
                                 noise_factor=t.NOISE_FACTOR.to_numpy()[tr],
                                 eta_noise_factor=t.ETA_NOISE_FACTOR.to_numpy()[tr], groups=G[tr])
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
    both = noise_flip.any() and (~noise_flip).any()
    au_fcx = roc_auc_score(noise_flip, phi[fl, 0] + phi[fl, 2] - phi[fl, 1]) if both else np.nan
    au_b = (roc_auc_score(noise_flip, base_phi[fl, 0] + base_phi[fl, 2] - base_phi[fl, 1])
            if both and np.isfinite(base_phi[fl]).all() else np.nan)
    wmean = {c: np.mean([w[c] for w in W]) for c in "PDN"}
    log(f"\n[{name}]  n={len(t):,}  positives {y.mean()*100:.1f}%")
    log(f"  black-box AUROC            {auc:.3f}")
    log(f"  C3 fidelity                R^2 {r2:.3f} | corr {np.corrcoef(z, fid)[0,1]:.3f}")
    log(f"  component weights b_P/b_D/b_N  {wmean['P']:+.2f} / {wmean['D']:+.2f} / {wmean['N']:+.2f}")
    log(f"  risk made of               proximity {sh[0]:.0f}% | drift {sh[1]:.0f}% | noise {sh[2]:.0f}%")
    log(f"  noise-vs-drift flips (AUROC, {fl.sum()} flips, {noise_flip.mean()*100:.0f}% noise):  "
        f"C3 {au_fcx:.3f} | grouped Shapley {au_b:.3f}")
    return dict(auc=auc, r2=r2, shares=sh, w=wmean, au_fcx=au_fcx, au_b=au_b, t=t, XH=XH, G=G)

log("=" * 78); log(f"C3 VALIDATION  |  black box: {'XGBoost' if BACKEND == 'xgb' else 'sklearn HGB'}"); log("=" * 78)
r1 = run("E1 CHEAP  -- deployment setting, any flip", CHEAP_SET, "LABEL_FLIPPED_NEXT")
r2 = run("E2 EXAM-INFORMED  -- full exam done at visit t, any flip", EXAM_SET, "LABEL_FLIPPED_NEXT")
r3 = run("E3 EXAM-INFORMED  -- SUSTAINED flip", EXAM_SET, "SUSTAINED")

log("\n" + "-" * 78); log("E4 PLANTED MECHANISMS  (heads always see the full exam-informed set)"); log("-" * 78)
PLANT = {"proximity-only": ["ell", "DIST_CUT", "logT", "logP"],
         "drift-only": DRIFT_F,
         "noise-only": NOISE_F}
res4 = {}
for k, fs in PLANT.items():
    res4[k] = run(f"E4 planted {k}", fs, "LABEL_FLIPPED_NEXT", head_feats=EXAM_SET,
                  groups_def=dict(P=fs if k == "proximity-only" else [], D=fs if k == "drift-only" else [],
                                  N=fs if k == "noise-only" else []))
log("\nPLANTED TEST (trained boxes) -- share of explained risk per component:")
tab = pd.DataFrame({k: v["shares"] for k, v in res4.items()}, index=["proximity", "drift", "noise"]).T.round(0)
log(tab.to_string())
COMP = {"proximity-only": ("proximity", "P"), "drift-only": ("drift", "D"), "noise-only": ("noise", "N")}
for k, (col, c) in COMP.items():
    r = res4[k]
    if r["auc"] < 0.55 or r["r2"] < 0.10:
        verdict = f"INCONCLUSIVE (box has no signal: AUROC {r['auc']:.2f}, C3 R^2 {r['r2']:.2f})"
    else:
        ok = tab.loc[k, col] > 50 and r["w"][c] > 0
        verdict = ("PASS" if ok else "FAIL") + f" ({col} share {tab.loc[k, col]:.0f}%, weight {r['w'][c]:+.2f})"
    log(f"  {k:15s} {verdict}")

# ---- synthetic planted mechanisms on the REAL feature distribution -----------------
# WARNING -- THIS TEST IS TAUTOLOGICAL AND IS NOT EVIDENCE.
# The mechanism is planted using the surrogate's OWN component s_c:
#     g = sigmoid(-0.9 + 1.5 * z(s_c))   =>   logit(g) is EXACTLY affine in s_c.
# calibrate_fidelity then solves nnls(S, logit(g)) against that same S on those same
# rows, which has the exact zero-residual solution w = e_c * 1.5 / std(s_c). Checked
# numerically: the recovered weight equals 1.5/std(s_c) to 8 decimal places. The test
# therefore passes for any data, any model and any patient, and its "3/3" says nothing
# about whether C3 can recover a real mechanism.
# A falsifiable version -- planting in the TRUE trajectory quantities, which the heads
# only approximate -- is in evaluation/improve_c3.py. Kept here unchanged so the older
# reports remain reproducible.
log("\nPLANTED TEST (synthetic) -- TAUTOLOGICAL, NOT EVIDENCE (see source comment);")
log("recovered weights are the closed form 1.5/std(s_c) by construction:")
XH, Gs = r2["XH"], r2["G"]
pd_ = PDNExplainer().fit(XH, r2["t"].eta.to_numpy(), r2["t"].mu.to_numpy(), r2["t"].noise_abs.to_numpy(),
                         noise_factor=r2["t"].NOISE_FACTOR.to_numpy(),
                         eta_noise_factor=r2["t"].ETA_NOISE_FACTOR.to_numpy(), groups=Gs)
H = pd_.heads(XH); pd_._set_reference(H); scomp, _ = pd_.surrogate_components(H=H)
syn_ok = 0
for c, name in (("P", "proximity"), ("D", "drift"), ("N", "noise")):
    zc = (scomp[c] - scomp[c].mean()) / (scomp[c].std() + 1e-9)
    gsyn = 1 / (1 + np.exp(-(-0.9 + 1.5 * zc)))
    pd_.calibrate_fidelity(XH, gsyn); w = pd_.b
    top = max(w, key=w.get); syn_ok += top == c
    log(f"  planted {name:9s}: b_P {w['P']:.2f} | b_D {w['D']:.2f} | b_N {w['N']:.2f}  -> recovered: "
        f"{dict(P='proximity', D='drift', N='noise')[top]}  {'PASS' if top == c else 'FAIL'}")
log(f"  synthetic planted recovery: {syn_ok}/3")

log("\n" + "=" * 78); log("SUMMARY"); log("=" * 78)
log(f"{'':44s}{'AUROC':>7s}{'C3 R^2':>8s}{'C3 noise/drift':>16s}{'baseline':>10s}")
for nm, r in [("E1 cheap, any flip", r1), ("E2 exam-informed, any flip", r2), ("E3 exam-informed, sustained", r3)]:
    log(f"{nm:44s}{r['auc']:7.3f}{r['r2']:8.3f}{r['au_fcx']:16.3f}{r['au_b']:10.3f}")
OUT = MET / ("c3_validation_xgb.txt" if BACKEND == "xgb" else "c3_validation.txt")
OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8"); print(f"\nwrote {OUT}")
