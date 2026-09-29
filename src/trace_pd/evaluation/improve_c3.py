"""STEP 4 -- a transition model with real signal, and FCX C3 on top of it.

Extends validate_c3.py with two experiments that give the black box the patient's
HISTORY as well as today's exam:

  E5  exam-informed + history, ANY flip at the next scheduled visit
  E6  the same features, FLIP WITHIN 12 MONTHS

History features (all shifted over the full scheduled sequence, so "previous" is the
adjacent scheduled visit): the previous visit's log-ratio `ell_prev`, its distance to
the nearest cutoff, the change `D_ell`, the previous exam medication state, and the
gap in months.

The black box is XGBoost selected over the SAME 6-config grid as Step 2
(max_depth 2/3/4 x learning_rate 0.03/0.08, 300 trees) by grouped CV on the
development patients, and trained UNWEIGHTED so the predicted probabilities keep
their scale and the Brier score means something. Brier is reported against the
base-rate Brier -- the score you get by predicting the cohort flip rate for everyone.

The planted-mechanism tests from validate_c3.py are kept, run against E5's feature
set, because a correctness claim about C3 needs a case where the right answer is known.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.improve_c3 [--backend xgb|hgb]
Out:  reports/metrics/c3_improved_xgb.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from trace_pd import config
from trace_pd.explain import formula as FM
from trace_pd.explain import trajectory as TJ
from trace_pd.explain.shapley import shapley_mc
from trace_pd.explain.pdn import PDNExplainer, logit
from trace_pd.models.backend import make_classifier, fit_classifier
from trace_pd.evaluation.splits import (
    load_locked_test_patients, log_run, patient_bootstrap_ci, CV_SEED,
)

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEED, N_BG, N_PERM = 42, 100, 16
GRID = [dict(max_depth=md, learning_rate=lr, n_estimators=300)
        for md in (2, 3, 4) for lr in (0.03, 0.08)]

LOG = []
def log(s=""):
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------------- data
df = pd.read_csv(config.LONG_TABLE, low_memory=False)
dd = pd.read_csv(config.DICTIONARY)
FEATS = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
df[FEATS] = df[FEATS].apply(pd.to_numeric, errors="coerce")
d = TJ.build(df)
DC = TJ.DELTA_COLS

# leak-free upstream subtype probabilities (grouped out-of-fold)
Xa = d[FEATS].to_numpy(float)
ya = d.LABEL.map({"TD": 0, "PIGD": 1, "INDETERMINATE": 2}).to_numpy()
oof = np.zeros((len(d), 3))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xa, ya, d.PATNO):
    m = fit_classifier(make_classifier("xgb", n_classes=3), Xa[tr], ya[tr])
    oof[te] = m.predict_proba(Xa[te])
d["P_TD"], d["P_PIGD"], d["P_IND"] = oof.T
sp = np.sort(oof, 1)
d["P_MARGIN"] = sp[:, -1] - sp[:, -2]

# ---- FLIP_WITHIN_12M on the labelled sequence --------------------------------
# any labelled visit in (t, t+12] months carrying a different label
flip12 = np.full(len(d), np.nan)
for _, idx in d.groupby("PATNO").indices.items():
    mo = d["VISIT_MONTH"].to_numpy()[idx]
    lab = d["LABEL"].to_numpy()[idx]
    for a in range(len(idx)):
        w = (mo > mo[a]) & (mo <= mo[a] + 12)
        if w.any():
            flip12[idx[a]] = float((lab[w] != lab[a]).any())
d["FLIP_WITHIN_12M"] = flip12

UP = ["P_TD", "P_PIGD", "P_IND", "P_MARGIN"]
NOISE_F = ["LEDD_TOTAL_MG", "N_CONMEDS", "PDTRTMNT", "D_LEDD_TOTAL_MG", "STATE_ON"]
DRIFT_F = [c for c in ["D_" + c for c in DC] + ["PRIOR_VISITS"] if c not in NOISE_F]
EXAM_F = ["ell", "logT", "logP", "DIST_CUT", "STATE_ON"]
HIST_P = ["ell_prev", "DIST_CUT_prev"]
HIST_D = ["D_ell", "MONTHS_SINCE_PREV"]
HIST_N = ["STATE_ON_prev"]

CHEAP_SET = FEATS + DRIFT_F + ["D_LEDD_TOTAL_MG"] + UP
EXAM_SET = CHEAP_SET + EXAM_F
HIST_SET = EXAM_SET + HIST_P + HIST_D + HIST_N

TEST_PATS = set(load_locked_test_patients().tolist())


def groups_for(feats):
    """Assign every feature to proximity / drift / noise exactly once."""
    D = [c for c in feats if c in DRIFT_F or c in HIST_D]
    N = [c for c in feats if c in NOISE_F or c in HIST_N]
    P = [c for c in feats if c not in D and c not in N]
    return dict(P=P, D=D, N=N)


# ------------------------------------------------------------------ model choice

def pick_config(X, y, G):
    """6-config grid, UNWEIGHTED, chosen by grouped CV AUROC on development rows."""
    best, rows = None, []
    for p in GRID:
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=CV_SEED)
        aucs = []
        for tr, te in sgk.split(X, y, G):
            if len(np.unique(y[tr])) < 2:
                continue
            m = XGBClassifier(subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              random_state=42, n_jobs=-1, tree_method="hist",
                              objective="binary:logistic", eval_metric="logloss", **p)
            m.fit(X[tr], y[tr])                       # NO sample weights
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        mean, sd = float(np.mean(aucs)), float(np.std(aucs))
        rows.append((p, mean, sd))
        if best is None or mean > best[1]:
            best = (p, mean, sd)
    return best, rows


def make_box(p):
    return XGBClassifier(subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         random_state=42, n_jobs=-1, tree_method="hist",
                         objective="binary:logistic", eval_metric="logloss", **p)


# -------------------------------------------------------------- one experiment

def run(name, feats, target, head_feats=None, groups_def=None, select=True, params=None):
    head_feats = feats if head_feats is None else head_feats
    t = d[d[target].notna()].reset_index(drop=True)
    X, XH = t[feats].to_numpy(float), t[head_feats].to_numpy(float)
    y = t[target].astype(int).to_numpy()
    G = t.PATNO.to_numpy()
    is_dev = np.array([p not in TEST_PATS for p in G])

    if select:
        (best_p, best_auc, best_sd), grid_rows = pick_config(X[is_dev], y[is_dev], G[is_dev])
        for p, mean, sd in grid_rows:
            log_run("4", name, "xgboost-unweighted", p, "dev grouped CV", "roc_auc",
                    mean, sd=sd, n=int(is_dev.sum()))
    else:
        best_p, best_auc, best_sd, grid_rows = params, float("nan"), float("nan"), []

    gp = np.zeros(len(t)); fid = np.zeros(len(t))
    phi = np.zeros((len(t), 3)); base_phi = np.full((len(t), 3), np.nan)
    W = []

    if groups_def is None:
        groups_def = groups_for(feats)
    flat = sum(groups_def.values(), [])
    assert len(flat) == len(set(flat)), "baseline groups overlap"
    grp = [[feats.index(c) for c in groups_def[k]] for k in "PDN"]

    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, G)):
        m = make_box(best_p).fit(X[tr], y[tr])          # unweighted
        gp[te] = m.predict_proba(X[te])[:, 1]

        pdn = PDNExplainer().fit(XH[tr], t.eta.to_numpy()[tr], t.mu.to_numpy()[tr],
                                 t.noise_abs.to_numpy()[tr],
                                 noise_factor=t.NOISE_FACTOR.to_numpy()[tr],
                                 eta_noise_factor=t.ETA_NOISE_FACTOR.to_numpy()[tr], groups=G[tr])
        pdn.calibrate_fidelity(XH[tr], m.predict_proba(X[tr])[:, 1])
        W.append(pdn.b)
        e = pdn.explain(XH[te])
        fid[te] = e["full"]
        phi[te] = np.c_[e["phi_P"], e["phi_D"], e["phi_N"]]

        if all(len(q) for q in grp):
            bg = X[np.random.default_rng(fold).choice(tr, min(N_BG, len(tr)), replace=False)]
            f = lambda Z, mm=m: logit(mm.predict_proba(Z)[:, 1])
            base_phi[te], _ = shapley_mc(f, X[te], bg, n_perm=N_PERM, groups=grp,
                                         rng=np.random.default_rng(fold))

    z = logit(gp)
    auc = roc_auc_score(y, gp)
    r2 = 1 - np.var(z - fid) / np.var(z)
    brier = brier_score_loss(y, gp)
    base_rate = float(y.mean())
    brier_base = brier_score_loss(y, np.full(len(y), base_rate))
    skill = 1 - brier / brier_base

    sh = np.abs(phi).mean(0); sh = sh / sh.sum() * 100
    fl = (y == 1) & t.eta.notna().to_numpy() & t.eta_next.notna().to_numpy()
    noise_flip = (FM.zone(t.eta.to_numpy()[fl]) == FM.zone(t.eta_next.to_numpy()[fl]))
    both = noise_flip.any() and (~noise_flip).any()
    au_fcx = roc_auc_score(noise_flip, phi[fl, 0] + phi[fl, 2] - phi[fl, 1]) if both else np.nan
    au_b = (roc_auc_score(noise_flip, base_phi[fl, 0] + base_phi[fl, 2] - base_phi[fl, 1])
            if both and np.isfinite(base_phi[fl]).all() else np.nan)
    wmean = {c: float(np.mean([w[c] for w in W])) for c in "PDN"}

    # locked test, once: refit on development rows only
    lo = hi = float("nan")
    if select:
        mfull = make_box(best_p).fit(X[is_dev], y[is_dev])
        p_te = mfull.predict_proba(X[~is_dev])[:, 1]
        y_te, g_te = y[~is_dev], G[~is_dev]
        auc_te = roc_auc_score(y_te, p_te)
        brier_te = brier_score_loss(y_te, p_te)
        lo, hi, _ = patient_bootstrap_ci(g_te, lambda r: roc_auc_score(y_te[r], p_te[r]))
        log_run("4", name, "xgboost-unweighted", best_p, "LOCKED TEST", "roc_auc",
                auc_te, ci=(lo, hi), n=len(y_te), n_patients=len(set(g_te.tolist())))
    else:
        auc_te = brier_te = float("nan")

    log(f"\n[{name}]  n={len(t):,}  positives {base_rate*100:.1f}%")
    if select:
        log(f"  chosen config              {best_p}  (dev CV AUROC {best_auc:.3f} +/- {best_sd:.3f})")
    log(f"  black-box AUROC            {auc:.3f}")
    log(f"  Brier                      {brier:.4f}  vs base-rate {brier_base:.4f}  "
        f"(skill {skill:+.1%}, {'better' if skill > 0 else 'NO BETTER'})")
    log(f"  C3 fidelity                R^2 {r2:.3f} | corr {np.corrcoef(z, fid)[0,1]:.3f}")
    log(f"  component weights b_P/b_D/b_N  {wmean['P']:+.2f} / {wmean['D']:+.2f} / {wmean['N']:+.2f}")
    log(f"  risk made of               proximity {sh[0]:.0f}% | drift {sh[1]:.0f}% | noise {sh[2]:.0f}%")
    log(f"  noise-vs-drift flips (AUROC, {fl.sum()} flips, {noise_flip.mean()*100:.0f}% noise):  "
        f"C3 {au_fcx:.3f} | grouped Shapley {au_b:.3f}")
    if select:
        log(f"  LOCKED TEST AUROC          {auc_te:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  Brier {brier_te:.4f}")

    for k, v in [("roc_auc", auc), ("c3_fidelity_r2", r2), ("brier", brier),
                 ("brier_base_rate", brier_base), ("share_proximity", sh[0]),
                 ("share_drift", sh[1]), ("share_noise", sh[2]),
                 ("noise_vs_drift_auroc_c3", au_fcx), ("noise_vs_drift_auroc_shapley", au_b)]:
        if np.isfinite(v):
            log_run("4", name, "xgboost-unweighted", best_p, "full grouped CV", k, v, n=len(t))

    return dict(auc=auc, r2=r2, brier=brier, brier_base=brier_base, skill=skill,
                shares=sh, w=wmean, au_fcx=au_fcx, au_b=au_b, t=t, XH=XH, G=G,
                params=best_p, auc_te=auc_te, ci=(lo, hi), n=len(t), base_rate=base_rate)


# ------------------------------------------------------------------------- main
log("=" * 86)
log("STEP 4 -- IMPROVED TRANSITION MODEL + FCX C3   (black box: XGBoost, UNWEIGHTED)")
log("=" * 86)
log(f"Config grid: max_depth 2/3/4 x learning_rate 0.03/0.08, 300 trees, chosen by")
log(f"grouped-CV AUROC on the development patients only. Locked test scored once.")
log(f"History features added to the exam-informed set: {HIST_P + HIST_D + HIST_N}")

r5 = run("E5 exam-informed + history, ANY flip", HIST_SET, "LABEL_FLIPPED_NEXT")
r6 = run("E6 exam-informed + history, FLIP WITHIN 12 MONTHS", HIST_SET, "FLIP_WITHIN_12M")

# reference points, same features, previous targets -- not re-selected
r2ref = run("E2ref exam-informed (no history), ANY flip", EXAM_SET, "LABEL_FLIPPED_NEXT",
            select=False, params=r5["params"])

# ---- planted mechanisms, kept from validate_c3.py ----------------------------
log("\n" + "-" * 86)
log("PLANTED MECHANISMS (trained boxes; C3 heads always see the full E5 set)")
log("-" * 86)
PLANT = {"proximity-only": ["ell", "DIST_CUT", "logT", "logP", "ell_prev", "DIST_CUT_prev"],
         "drift-only": DRIFT_F + HIST_D,
         "noise-only": NOISE_F + HIST_N}
res4 = {}
for k, fs in PLANT.items():
    res4[k] = run(f"planted {k}", fs, "LABEL_FLIPPED_NEXT", head_feats=HIST_SET,
                  groups_def=dict(P=fs if k == "proximity-only" else [],
                                  D=fs if k == "drift-only" else [],
                                  N=fs if k == "noise-only" else []),
                  select=False, params=r5["params"])

tab = pd.DataFrame({k: v["shares"] for k, v in res4.items()},
                   index=["proximity", "drift", "noise"]).T.round(0)
log("\nshare of explained risk per component:")
log(tab.to_string())
COMP = {"proximity-only": ("proximity", "P"), "drift-only": ("drift", "D"), "noise-only": ("noise", "N")}
planted_pass = 0
for k, (col, c) in COMP.items():
    r = res4[k]
    if r["auc"] < 0.55 or r["r2"] < 0.10:
        verdict = f"INCONCLUSIVE (box has no signal: AUROC {r['auc']:.2f}, C3 R^2 {r['r2']:.2f})"
    else:
        ok = tab.loc[k, col] > 50 and r["w"][c] > 0
        planted_pass += bool(ok)
        verdict = ("PASS" if ok else "FAIL") + f" ({col} share {tab.loc[k, col]:.0f}%, weight {r['w'][c]:+.2f})"
    log(f"  {k:15s} {verdict}")

# ---- synthetic planted mechanisms -------------------------------------------
# The version in validate_c3.py plants the mechanism using the surrogate's OWN fitted
# component s_c:  g = sigmoid(-0.9 + 1.5 * z(s_c)). Then logit(g) is EXACTLY affine in
# s_c, and calibrate_fidelity solves nnls(S, logit(g)) against the same S on the same
# rows -- an exactly-solvable system whose closed-form answer is w = e_c * 1.5/std(s_c).
# Verified: the recovered weight equals 1.5/std(s_c) to 8 decimal places. That test
# cannot fail for any data, any model and any patient, so its "3/3" is not evidence.
#
# Here the mechanism is planted in the TRUE trajectory quantities instead (distance of
# the smoothed eta to a cutoff, the true drift mu, the true noise magnitude). C3 must
# recover it through its own imperfect heads, so the test can genuinely fail.
log("\nSYNTHETIC PLANTED MECHANISMS")
log("  (planted in the TRUE trajectory quantities, so recovery is not automatic --")
log("   see the note on the tautological variant in validate_c3.py)")
XH, Gs = r5["XH"], r5["G"]
tt = r5["t"]
pdn = PDNExplainer().fit(XH, tt.eta.to_numpy(), tt.mu.to_numpy(), tt.noise_abs.to_numpy(),
                         noise_factor=tt.NOISE_FACTOR.to_numpy(),
                         eta_noise_factor=tt.ETA_NOISE_FACTOR.to_numpy(), groups=Gs)
H = pdn.heads(XH); pdn._set_reference(H)
scomp, _ = pdn.surrogate_components(H=H)

eta_t = tt.eta.to_numpy(float)
truth = {
    "P": -FM.distance_to_cutoff(eta_t),        # nearer a cutoff -> higher risk
    "D": np.abs(tt.mu.to_numpy(float)),        # larger true drift -> higher risk
    "N": tt.noise_abs.to_numpy(float),         # larger true scatter -> higher risk
}
syn_ok = 0
for c, nm in (("P", "proximity"), ("D", "drift"), ("N", "noise")):
    v = truth[c].copy()
    good = np.isfinite(v)
    v[~good] = np.nanmedian(v[good])
    z_true = (v - v.mean()) / (v.std() + 1e-9)
    gsyn = 1 / (1 + np.exp(-(-0.9 + 1.5 * z_true)))
    pdn.calibrate_fidelity(XH, gsyn)
    w = pdn.b
    top = max(w, key=w.get)
    syn_ok += top == c
    # how much of the planted signal the surrogate can express at all
    r_c = np.corrcoef(scomp[c][good], z_true[good])[0, 1]
    log(f"  planted {nm:9s} (true quantity): b_P {w['P']:.2f} | b_D {w['D']:.2f} | b_N {w['N']:.2f}"
        f"  -> recovered {dict(P='proximity', D='drift', N='noise')[top]}  "
        f"{'PASS' if top == c else 'FAIL'}   corr(s_{c}, planted) {r_c:+.2f}")
log(f"  synthetic planted recovery: {syn_ok}/3  (falsifiable version)")

# ------------------------------------------------------------------- summary
log("\n" + "=" * 86)
log("SUMMARY")
log("=" * 86)
log(f"{'':52s}{'AUROC':>7s}{'Brier':>9s}{'base':>9s}{'C3 R^2':>8s}")
for nm, r in [("E2ref exam-informed, any flip (no history)", r2ref),
              ("E5 exam-informed + history, any flip", r5),
              ("E6 exam-informed + history, flip within 12m", r6)]:
    log(f"{nm:52s}{r['auc']:7.3f}{r['brier']:9.4f}{r['brier_base']:9.4f}{r['r2']:8.3f}")

log("")
log(f"{'':52s}{'prox%':>7s}{'drift%':>8s}{'noise%':>8s}{'C3 n/d':>9s}{'Shapley':>9s}")
for nm, r in [("E2ref", r2ref), ("E5", r5), ("E6", r6)]:
    log(f"{nm:52s}{r['shares'][0]:7.0f}{r['shares'][1]:8.0f}{r['shares'][2]:8.0f}"
        f"{r['au_fcx']:9.3f}{r['au_b']:9.3f}")

log("")
log("WHAT DID NOT HOLD")
log("-" * 86)
for nm, r in [("E5", r5), ("E6", r6)]:
    if abs(r["w"]["N"]) < 0.05:
        log(f"  - {nm}: the NOISE weight collapses to b_N {r['w']['N']:+.2f} once history features are")
        log(f"    added (E2ref without history has b_N {r2ref['w']['N']:+.2f}). C3 then explains the risk as")
        log(f"    {r['shares'][0]:.0f}% proximity + {r['shares'][1]:.0f}% drift and has no noise channel left to attribute with,")
        log(f"    which is why its noise-vs-drift AUROC ({r['au_fcx']:.3f}) falls BELOW chance. A better")
        log(f"    black box therefore produced a WORSE C3 decomposition, not a better one.")
    if r["r2"] < r2ref["r2"]:
        log(f"  - {nm} C3 fidelity R^2 {r['r2']:.3f} is LOWER than the no-history E2ref ({r2ref['r2']:.3f}): "
            f"the surrogate tracks the richer black box less well.")
    if r["skill"] <= 0:
        log(f"  - {nm} Brier {r['brier']:.4f} is NO BETTER than predicting the base rate "
            f"({r['brier_base']:.4f}). Ranking improved, calibration did not.")
    if np.isfinite(r["au_fcx"]) and np.isfinite(r["au_b"]) and r["au_fcx"] <= r["au_b"] + 0.02:
        log(f"  - {nm} noise-vs-drift attribution ({r['au_fcx']:.3f}) does not beat grouped "
            f"Shapley on the black box ({r['au_b']:.3f}); C3 still buys structure, not accuracy.")
log(f"  - Every E5/E6 number uses the true exam at visit t, so it describes the question "
    f"'will today's subtype hold?', NOT the deployment setting where the exam is missing.")
log(f"  - Planted-mechanism verdicts: {planted_pass}/3 trained, {syn_ok}/3 synthetic.")
log(f"  - The history features are NaN at a patient's first visit, so early visits carry less "
    f"information than the row count suggests.")
log("=" * 86)

OUT = config.METRICS / "c3_improved_xgb.txt"
OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")
