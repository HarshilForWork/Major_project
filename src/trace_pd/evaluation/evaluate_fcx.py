"""End-to-end evaluation of FCX (Formula-Coordinate Explanations) on PPMI.

Explains three black boxes, all trained here with patient-grouped CV:
  g1  subtype classifier (TD vs PIGD)                    -> C1 implied exam
  g2  split-conformal LAC sets on g1                     -> C2 cutoff straddle
  g3  transition-risk classifier (label flips next visit) -> C3 proximity/drift/noise

Every explanation is scored three ways: FIDELITY to the black box, ALIGNMENT with the
true hidden scores, and CORRECTNESS of what it blames. Baselines use the same own
Shapley implementation on the black box directly, with features grouped by channel.

Run:  python src/trace_pd/evaluation/evaluate_fcx.py [--backend xgb|hgb]   (or: make fcx)
      default backend: xgb if installed, else hgb. Output: reports/metrics/fcx_results[_xgb].txt
"""
from pathlib import Path as _Path
import sys
_ROOT = _Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight
from trace_pd.explain import formula as FM
from trace_pd.explain.shapley import shapley_mc
from trace_pd.explain.implied_exam import ImpliedExamExplainer
from trace_pd.explain.straddle import CutoffStraddleExplainer
from trace_pd.explain.pdn import PDNExplainer, logit
from trace_pd.models.backend import make_classifier, fit_classifier, backend_from_argv
from trace_pd.explain import trajectory as TJ
BACKEND = backend_from_argv(sys.argv)
SFX = "_xgb" if BACKEND == "xgb" else ""
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles / redirected output
except Exception:
    pass

PROC, MET, FIG, INT = (_ROOT / "data/processed", _ROOT / "reports/metrics",
                       _ROOT / "reports/figures", _ROOT / "data/interim")
for p in (MET, FIG, INT): p.mkdir(parents=True, exist_ok=True)
CLF = dict(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)
N_EXPLAIN, N_BG, N_PERM, SEED = 120, 100, 16, 42
LOG = []
def log(s=""): print(s); LOG.append(s)

df = pd.read_csv(PROC / "ppmi_tdpigd_long.csv", low_memory=False).sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True)
dd = pd.read_csv(PROC / "ppmi_tdpigd_dictionary.csv")
FEATS = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
df[FEATS] = df[FEATS].apply(pd.to_numeric, errors="coerce")
df["logT"], df["logP"] = FM.log_scores(df.TREMOR_SCORE, df.PIGD_SCORE)
df["ell"] = df.logT - df.logP
lab = df.LABEL.notna()
zone_ok = (FM.ZONE_NAMES[FM.zone(df.loc[lab, "ell"])] == df.loc[lab, "LABEL"]).mean()

log("=" * 78); log("FCX -- FORMULA-COORDINATE EXPLANATIONS  |  evaluation on PPMI"); log("=" * 78)
log(f"features: {len(FEATS)} cheap | labelled visits: {lab.sum():,} | "
    f"formula zones reproduce stored labels: {zone_ok*100:.2f}%")
log(f"black boxes: {'XGBoost' if BACKEND == 'xgb' else 'sklearn HistGradientBoosting'}  (explainer heads: sklearn HGB)")

def fit_clf(X, y, balanced=True):
    return fit_classifier(make_classifier(BACKEND, n_classes=len(np.unique(y))), X, y, balanced=balanced)

def channel_groups(Xtr, lT, lP):
    """Baseline grouping: a feature is 'tremor-ish' if it correlates more with true log T than log P."""
    t, g = [], []
    for j, c in enumerate(FEATS):
        x = Xtr[:, j]; m = np.isfinite(x)
        ct = abs(np.corrcoef(x[m], lT[m])[0, 1]) if m.sum() > 10 else 0
        cp = abs(np.corrcoef(x[m], lP[m])[0, 1]) if m.sum() > 10 else 0
        (t if ct > cp else g).append(j)
    return t, g

# =========================================================================== C1 + C2
b = df[df.LABEL.isin(["TD", "PIGD"])].reset_index(drop=True)
X = b[FEATS].to_numpy(float); y = (b.LABEL == "PIGD").astype(int).to_numpy(); G = b.PATNO.to_numpy()
lT, lP, ell = b.logT.to_numpy(), b.logP.to_numpy(), b.ell.to_numpy()
n = len(b)
R = dict(g_dec=np.zeros(n, int), g_prob=np.zeros(n), iee_dec=np.zeros(n, int), hT=np.zeros(n), hP=np.zeros(n))
c1_rows, c2_rows = [], []
outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
rng = np.random.default_rng(SEED)
for fold, (tr, te) in enumerate(outer.split(X, y, G)):
    g1 = fit_clf(X[tr], y[tr])
    iee = ImpliedExamExplainer(FEATS).fit(X[tr], lT[tr], lP[tr])
    iee.calibrate_fidelity(X[tr], g1.predict(X[tr]))
    R["g_prob"][te] = g1.predict_proba(X[te])[:, 1]; R["g_dec"][te] = g1.predict(X[te])
    R["iee_dec"][te] = iee.decide(X[te]); R["hT"][te], R["hP"][te], _ = iee.implied(X[te])

    # --- explanations on a sample of test rows (C1) and the black-box baseline
    ex = rng.choice(te, size=min(N_EXPLAIN, len(te)), replace=False)
    bg = X[rng.choice(tr, size=N_BG, replace=False)]
    e = iee.explain(X[ex], bg, n_perm=N_PERM, seed=fold)
    g_logit = lambda Z, m=g1: logit(m.predict_proba(Z)[:, 1])
    tg, gg = channel_groups(X[tr], lT[tr], lP[tr])
    phi_b, _ = shapley_mc(g_logit, X[ex], bg, n_perm=N_PERM, groups=[tg, gg], rng=np.random.default_rng(fold))
    # stability: re-explain with a different background + seed
    bg2 = X[rng.choice(tr, size=N_BG, replace=False)]
    e2 = iee.explain(X[ex], bg2, n_perm=N_PERM, seed=100 + fold)
    phi_full, _ = shapley_mc(g_logit, X[ex], bg, n_perm=N_PERM, rng=np.random.default_rng(fold))
    phi_full2, _ = shapley_mc(g_logit, X[ex], bg2, n_perm=N_PERM, rng=np.random.default_rng(100 + fold))
    for k, i in enumerate(ex):
        c1_rows.append(dict(i=i, fold=fold, y=y[i], g=R["g_dec"][i],
                            trem_contrib=e["phi_tremor"][k].sum(), gait_contrib=-e["phi_gait"][k].sum(),
                            base_trem=phi_b[k, 0], base_gait=phi_b[k, 1],
                            **{f"phiR_{c}": e["phi_ratio"][k, j] for j, c in enumerate(FEATS)},
                            **{f"phiT_{c}": e["phi_tremor"][k, j] for j, c in enumerate(FEATS)},
                            **{f"phiP_{c}": e["phi_gait"][k, j] for j, c in enumerate(FEATS)},
                            stab_fcx=spearmanr(e["phi_ratio"][k], e2["phi_ratio"][k]).statistic,
                            stab_shap=spearmanr(phi_full[k], phi_full2[k]).statistic,
                            narrative=iee.narrative(e, k)))

    # --- C2: conformal sets on g1 (black box g2) and their straddle explanation
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=7)
    a, c = next(inner.split(X[tr], y[tr], G[tr])); itr, cal = tr[a], tr[c]
    g1i = fit_clf(X[itr], y[itr])
    pc = g1i.predict_proba(X[cal]); s = 1 - pc[np.arange(len(cal)), y[cal]]
    qhat = np.sort(s)[min(len(s), int(np.ceil((len(s) + 1) * 0.9))) - 1]
    pt = g1i.predict_proba(X[te]); inset = pt >= 1 - qhat
    inset[inset.sum(1) == 0, pt[inset.sum(1) == 0].argmax(1)] = True
    ieei = ImpliedExamExplainer(FEATS).fit(X[itr], lT[itr], lP[itr]).calibrate_fidelity(X[itr], g1i.predict(X[itr]))
    st = CutoffStraddleExplainer(alpha=0.10).fit(X[itr], lT[itr], lP[itr])
    pc_set = pc >= 1 - qhat; amb_cal = pc_set.sum(1) == 2                       # black-box ambiguity on cal rows
    st.calibrate_fidelity(X[cal], ieei.implied(X[cal])[2], ieei.cut, amb_cal)
    pT, pP, pl = ieei.implied(X[te])
    o = st.explain(X[te], pl, ieei.cut, pT, pP, true_T=lT[te], true_P=lP[te])
    # REAL test of the actionable claim: give the SAME kind of conformal predictor the true
    # score of one channel (as if that part of the exam were done) and see whether ITS set
    # collapses to a singleton. Independent of the explainer's own intervals.
    def lac_sets(Xtr_, ytr_, Xcal_, ycal_, Xte_):
        m_ = fit_clf(Xtr_, ytr_); pc_ = m_.predict_proba(Xcal_)
        s_ = 1 - pc_[np.arange(len(ycal_)), ycal_]
        q_ = np.sort(s_)[min(len(s_), int(np.ceil((len(s_) + 1) * 0.9))) - 1]
        pt_ = m_.predict_proba(Xte_); ins_ = pt_ >= 1 - q_
        ins_[ins_.sum(1) == 0, pt_[ins_.sum(1) == 0].argmax(1)] = True
        return ins_.sum(1)
    XT, XP = np.c_[X, lT], np.c_[X, lP]
    size_T = lac_sets(XT[itr], y[itr], XT[cal], y[cal], XT[te])     # tremor exam done
    size_P = lac_sets(XP[itr], y[itr], XP[cal], y[cal], XP[te])     # gait exam done
    # null: the same test with the TRUE scores shuffled across patients -- how much of
    # "revealing the blamed channel resolves it" happens whatever the true value is?
    prm = np.random.default_rng(1000 + fold).permutation(len(te))
    on = st.explain(X[te], pl, ieei.cut, pT, pP, true_T=lT[te][prm], true_P=lP[te][prm])
    # baseline blame on the SAME model that produced the sets (g1i), for the explained rows
    pos = {i: k for k, i in enumerate(te)}
    tgi, ggi = channel_groups(X[itr], lT[itr], lP[itr])
    g1i_logit = lambda Z, m=g1i: logit(m.predict_proba(Z)[:, 1])
    phib_i, _ = shapley_mc(g1i_logit, X[ex], X[rng.choice(itr, size=N_BG, replace=False)], n_perm=N_PERM,
                           groups=[tgi, ggi], rng=np.random.default_rng(200 + fold))
    bi = {i: phib_i[k] for k, i in enumerate(ex)}
    for k, i in enumerate(te):
        row = dict(i=i, fold=fold, set_size=int(inset[k].sum()), covered=bool(inset[k, y[i]]),
                   kappa=st.kappa, set_size_if_tremor=int(size_T[k]), set_size_if_gait=int(size_P[k]), null_res_T=bool(on["really_resolves_tremor"][k]),
                   null_res_P=bool(on["really_resolves_gait"][k]), **{kk: vv[k] for kk, vv in o.items()})
        if i in bi: row.update(base_trem_i=bi[i][0], base_gait_i=bi[i][1])
        c2_rows.append(row)
    print(f"  fold {fold+1}/5 done", flush=True)

c1 = pd.DataFrame(c1_rows); c2 = pd.DataFrame(c2_rows)

# ------------------------------------------------------------------ C1 metrics
log("\n" + "-" * 78); log("C1  IMPLIED-EXAM EXPLANATION of the subtype classifier (TD vs PIGD)"); log("-" * 78)
agree = np.mean(R["iee_dec"] == R["g_dec"])
rho = spearmanr(R["g_prob"], -(R["hT"] - R["hP"])).statistic
log(f"FIDELITY   implied-exam decision agrees with the classifier: {agree*100:.1f}%   "
    f"| Spearman(P(PIGD), -implied ratio): {rho:.3f}")
log(f"           balanced acc vs truth -- classifier {balanced_accuracy_score(y, R['g_dec']):.3f} | implied exam {balanced_accuracy_score(y, R['iee_dec']):.3f}")
log(f"ALIGNMENT  corr(implied, true):  log T {np.corrcoef(R['hT'], lT)[0,1]:.3f} | log P {np.corrcoef(R['hP'], lP)[0,1]:.3f} "
    f"| ratio {np.corrcoef(R['hT']-R['hP'], ell)[0,1]:.3f}")
# channel carrying the decision (global)
aT = np.abs(c1[[f"phiT_{c}" for c in FEATS]].to_numpy()).sum(1); aP = np.abs(c1[[f"phiP_{c}" for c in FEATS]].to_numpy()).sum(1)
log(f"CHANNELS   share of attribution mass through the gait channel: {np.mean(aP/(aT+aP))*100:.1f}% "
    f"(tremor {np.mean(aT/(aT+aP))*100:.1f}%)")
# error attribution: when the classifier is wrong, which channel is truly to blame?
err = c1[(c1.y != c1.g) & (R["iee_dec"][c1.i] == c1.g)].copy()   # classifier wrong AND explainer faithful there
eT = R["hT"][err.i] - lT[err.i]; eP = R["hP"][err.i] - lP[err.i]
dirn = np.sign((R["hT"][err.i] - R["hP"][err.i]) - ell[err.i])          # direction the implied ratio is off
true_blame = np.where(eT * dirn >= -eP * dirn, "tremor", "gait")          # channel contributing more to the error
toward = np.where(err.g == 1, -1, 1)                                       # PIGD decision <-> ratio pushed down
fcx_blame = np.where(err.trem_contrib * toward >= err.gait_contrib * toward, "tremor", "gait")
base_blame = np.where(err.base_trem * (-toward) >= err.base_gait * (-toward), "tremor", "gait")  # logit(PIGD) axis
acc = lambda a, t: balanced_accuracy_score(t, a)
_c2w = pd.DataFrame(c2_rows).set_index("i")
wt, wg = _c2w.loc[err.i, "hw_tremor"].to_numpy(), _c2w.loc[err.i, "hw_gait"].to_numpy()
fcx_unc_blame = np.where(wt >= wg, "tremor", "gait")
log(f"ERROR ATTRIBUTION on {len(err)} misclassifications the explainer reproduces -- which of the EXPLAINER's channels is wrong?")
log(f"           true blame: tremor {np.mean(true_blame=='tremor')*100:.0f}% / gait {np.mean(true_blame=='gait')*100:.0f}%")
log(f"           FCX (uncertainty channel, C1+C2) balanced acc {acc(fcx_unc_blame, true_blame):.3f} | "
    f"FCX (attribution direction) {acc(fcx_blame, true_blame):.3f} | grouped Shapley on classifier {acc(base_blame, true_blame):.3f} | chance 0.500")
log(f"STABILITY  rank corr of attributions across background/seed:  FCX {c1.stab_fcx.median():.3f}  vs  Shapley-on-classifier {c1.stab_shap.median():.3f}")
imp = pd.DataFrame({"tremor": np.abs(c1[[f'phiT_{c}' for c in FEATS]].to_numpy()).mean(0),
                    "gait": np.abs(c1[[f'phiP_{c}' for c in FEATS]].to_numpy()).mean(0)}, index=FEATS)
imp["total"] = imp.sum(1); imp = imp.sort_values("total", ascending=False)
log("top features by channel (mean |attribution|, log-units):")
log(imp.head(8).round(4).to_string())

# ------------------------------------------------------------------ C2 metrics
log("\n" + "-" * 78); log("C2  CUTOFF-STRADDLE EXPLANATION of the conformal sets"); log("-" * 78)
log(f"black-box conformal (LAC): coverage {c2.covered.mean()*100:.1f}% | ambiguous sets {np.mean(c2.set_size==2)*100:.1f}%")
log(f"fidelity-calibrated kappa (mean over folds): {c2.kappa.mean():.2f} | channel intervals then cover the TRUE score: "
    f"tremor {c2.covered_T.mean()*100:.1f}% | gait {c2.covered_P.mean()*100:.1f}%")
amb = (c2.set_size == 2).to_numpy(); strd = c2.straddle.to_numpy().astype(bool)
log(f"FIDELITY   straddle vs ambiguous set: agreement {np.mean(amb==strd)*100:.1f}% | kappa {cohen_kappa_score(amb, strd):.3f} | "
    f"ambiguous sets explained by a straddle: {np.mean(strd[amb])*100:.1f}%")
s = c2[strd]
log(f"BLAME      among straddles: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in s.blame.value_counts(normalize=True).items()))
for ch, col in (("tremor", "really_resolves_tremor"), ("gait", "really_resolves_gait")):
    other = "really_resolves_gait" if ch == "tremor" else "really_resolves_tremor"
    sub = s[s.blame == ch]
    if len(sub):
        log(f"VERIFIED   blamed {ch:6s} (n={len(sub)}): revealing the TRUE {ch} score resolves it {sub[col].mean()*100:.0f}% "
            f"| revealing the other channel instead: {sub[other].mean()*100:.0f}%")
log(f"           overall: revealing true tremor resolves {s.really_resolves_tremor.mean()*100:.0f}% of straddles, true gait {s.really_resolves_gait.mean()*100:.0f}%")
sb = s[s.blame.isin(["tremor", "gait"])]
hit = np.where(sb.blame == "tremor", sb.really_resolves_tremor, sb.really_resolves_gait)
nul = np.where(sb.blame == "tremor", sb.null_res_T, sb.null_res_P)
log(f"NULL CHECK   blamed channel resolves it with the TRUE score {np.mean(hit)*100:.0f}% | with a SHUFFLED "
    f"(wrong patient's) score {np.mean(nul)*100:.0f}%  -> real information beyond the mechanics: {(np.mean(hit)-np.mean(nul))*100:+.0f} pts")
# ---- the real test: does measuring the blamed channel collapse the CONFORMAL SET itself?
A2 = c2[(c2.set_size == 2) & c2.blame.isin(["tremor", "gait"])].copy()
A2["res_T"] = A2.set_size_if_tremor == 1; A2["res_P"] = A2.set_size_if_gait == 1
hitR = np.where(A2.blame == "tremor", A2.res_T, A2.res_P); othR = np.where(A2.blame == "tremor", A2.res_P, A2.res_T)
log(f"REAL TEST    on {len(A2)} ambiguous sets with a single-channel blame: giving the conformal model the TRUE score of the")
log(f"             blamed channel collapses the set {np.mean(hitR)*100:.0f}% | the OTHER channel {np.mean(othR)*100:.0f}% | "
    f"random channel {np.mean((A2.res_T.astype(float)+A2.res_P.astype(float))/2)*100:.0f}%")
log(f"             overall: tremor exam collapses {np.mean(c2.loc[c2.set_size==2,'set_size_if_tremor']==1)*100:.0f}% of ambiguous sets, "
    f"gait exam {np.mean(c2.loc[c2.set_size==2,'set_size_if_gait']==1)*100:.0f}%")
log(f"TRIVIAL RULE 'always examine tremor' on the same sets: {A2.res_T.mean()*100:.0f}%  "
    f"(FCX per-patient blame {np.mean(hitR)*100:.0f}%)")
gb = A2[A2.blame == "gait"]
log(f"             when FCX blames GAIT (n={len(gb)}): gait exam collapses {gb.res_P.mean()*100:.0f}% | tremor exam {gb.res_T.mean()*100:.0f}%")
jr = A2[A2.base_trem_i.notna()] if "base_trem_i" in A2 else A2.iloc[:0]
if len(jr):
    bbr = np.where(jr.base_trem_i.abs() >= jr.base_gait_i.abs(), "tremor", "gait")
    f_r = np.where(jr.blame == "tremor", jr.res_T, jr.res_P); b_r = np.where(bbr == "tremor", jr.res_T, jr.res_P)
    log(f"REAL vs BASELINE on {len(jr)} of those (same rows): FCX blame collapses the set {np.mean(f_r)*100:.0f}% | "
        f"grouped Shapley blame {np.mean(b_r)*100:.0f}% | random {np.mean((jr.res_T.astype(float)+jr.res_P.astype(float))/2)*100:.0f}%")
j = sb[sb.base_trem_i.notna()] if "base_trem_i" in sb else sb.iloc[:0]
fcx_hit = np.where(j.blame == "tremor", j.really_resolves_tremor, j.really_resolves_gait)
fcx_nul = np.where(j.blame == "tremor", j.null_res_T, j.null_res_P)
bb = np.where(j.base_trem_i.abs() >= j.base_gait_i.abs(), "tremor", "gait")
base_hit = np.where(bb == "tremor", j.really_resolves_tremor, j.really_resolves_gait)
base_nul = np.where(bb == "tremor", j.null_res_T, j.null_res_P)
rnd_hit = (j.really_resolves_tremor.astype(float) + j.really_resolves_gait.astype(float)) / 2
log(f"BLAME vs BASELINE on {len(j)} straddles (same rows, same model): blamed channel really resolves it -- "
    f"FCX {np.mean(fcx_hit)*100:.0f}% (shuffled {np.mean(fcx_nul)*100:.0f}%) | grouped Shapley on the set's model "
    f"{np.mean(base_hit)*100:.0f}% (shuffled {np.mean(base_nul)*100:.0f}%) | random channel {rnd_hit.mean()*100:.0f}%")

# =========================================================================== C3
log("\n" + "-" * 78); log("C3  PROXIMITY / DRIFT / NOISE explanation of the transition-risk model"); log("-" * 78)
d3 = TJ.build(df.drop(columns=["logT", "logP", "ell"]))      # shifts over the FULL scheduled sequence
DCOLS = TJ.DELTA_COLS
# out-of-fold 3-class subtype probabilities (leak-free version of the upstream features)
X_all = d3[FEATS].to_numpy(float); y_all = d3.LABEL.map({"TD": 0, "PIGD": 1, "INDETERMINATE": 2}).to_numpy()
oofp = np.zeros((len(d3), 3))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X_all, y_all, d3.PATNO):
    oofp[te] = fit_clf(X_all[tr], y_all[tr]).predict_proba(X_all[te])
d3["P_TD"], d3["P_PIGD"], d3["P_IND"] = oofp.T
sp = np.sort(oofp, 1); d3["P_MARGIN"] = sp[:, -1] - sp[:, -2]
F3 = FEATS + ["D_" + c for c in DCOLS] + ["PRIOR_VISITS", "P_TD", "P_PIGD", "P_IND", "P_MARGIN"]
t3 = d3[d3.LABEL_FLIPPED_NEXT.notna()].reset_index(drop=True)
X3 = t3[F3].to_numpy(float); y3 = t3.LABEL_FLIPPED_NEXT.astype(int).to_numpy(); G3 = t3.PATNO.to_numpy()
g3p = np.zeros(len(t3)); fid = np.zeros(len(t3)); rows3 = []; W3 = []
LEVEL = list(range(len(FEATS))) + [F3.index(c) for c in ("P_TD", "P_PIGD", "P_IND", "P_MARGIN")]
DRIFT = [F3.index("D_" + c) for c in DCOLS] + [F3.index("PRIOR_VISITS")]
NOISEF = [F3.index(c) for c in ("LEDD_TOTAL_MG", "N_CONMEDS", "PDTRTMNT", "D_LEDD_TOTAL_MG")]
LEVEL = [j for j in LEVEL if j not in NOISEF]; DRIFT = [j for j in DRIFT if j not in NOISEF]
for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X3, y3, G3)):
    g3 = fit_clf(X3[tr], y3[tr], balanced=False)       # probabilities are reported -> no reweighting
    g3p[te] = g3.predict_proba(X3[te])[:, 1]
    pdn = PDNExplainer().fit(X3[tr], t3.eta.to_numpy()[tr], t3.mu.to_numpy()[tr], t3.noise_abs.to_numpy()[tr],
                             noise_factor=t3.NOISE_FACTOR.to_numpy()[tr],
                             eta_noise_factor=t3.ETA_NOISE_FACTOR.to_numpy()[tr], groups=G3[tr])
    pdn.calibrate_fidelity(X3[tr], g3.predict_proba(X3[tr])[:, 1])
    fid[te] = pdn.predict_logit(X3[te])
    e = pdn.explain(X3[te]); W3.append(e["weights"])
    g3l = lambda Z, m=g3: logit(m.predict_proba(Z)[:, 1])
    bg = X3[np.random.default_rng(fold).choice(tr, size=N_BG, replace=False)]
    phb, _ = shapley_mc(g3l, X3[te], bg, n_perm=N_PERM, groups=[LEVEL, DRIFT, NOISEF], rng=np.random.default_rng(fold))
    for k, i in enumerate(te):
        rows3.append(dict(i=i, phi_P=e["phi_P"][k], phi_D=e["phi_D"][k], phi_N=e["phi_N"][k],
                          eta_hat=e["eta"][k], mu_hat=e["mu"][k], sigma_hat=e["sigma"][k],
                          b_level=phb[k, 0], b_drift=phb[k, 1], b_noise=phb[k, 2]))
    print(f"  C3 fold {fold+1}/5 done", flush=True)
c3 = pd.DataFrame(rows3).set_index("i").sort_index()
t3 = t3.join(c3)
zg = logit(g3p)
log(f"black-box transition model: AUROC {roc_auc_score(y3, g3p):.3f} (base flip rate {y3.mean()*100:.1f}%)")
log(f"FIDELITY   corr(logit g3, FCX surrogate) {np.corrcoef(zg, fid)[0,1]:.3f} | R^2 {1 - np.var(zg-fid)/np.var(zg):.3f} | "
    f"surrogate AUROC {roc_auc_score(y3, fid):.3f}")
log("component weights b_P / b_D / b_N (mean over folds): " + " / ".join(f"{np.mean([w[c] for w in W3]):+.2f}" for c in "PDN"))
aP_, aD_, aN_ = (t3.phi_P.abs().mean(), t3.phi_D.abs().mean(), t3.phi_N.abs().mean())
tot = aP_ + aD_ + aN_
log(f"WHAT DRIVES PREDICTED RISK (mean |phi| share, surrogate R^2 {1 - np.var(zg-fid)/np.var(zg):.2f}):  "
    f"proximity {aP_/tot*100:.0f}% | drift {aD_/tot*100:.0f}% | noise {aN_/tot*100:.0f}%")
# ground-truth flip type from the smoothed TRUE trajectory
fl = t3[(t3.LABEL_FLIPPED_NEXT == 1) & t3.eta.notna() & t3.eta_next.notna()].copy()
fl["noise_flip"] = FM.zone(fl.eta) == FM.zone(fl.eta_next)
log(f"ground-truth flip types ({len(fl)} flips with smoothed trajectory): noise flips {fl.noise_flip.mean()*100:.0f}% | drift flips {(~fl.noise_flip).mean()*100:.0f}%")
au_f = roc_auc_score(fl.noise_flip, fl.phi_N + fl.phi_P - fl.phi_D)
au_b = roc_auc_score(fl.noise_flip, fl.b_noise + fl.b_level - fl.b_drift)
log(f"CORRECTNESS  AUROC for telling noise flips from drift flips:  FCX {au_f:.3f}  vs  grouped Shapley on g3 {au_b:.3f}  (chance 0.5)")
t3["reverted"] = t3.REVERTED                              # scheduled t+2, from trajectory.build
f2 = t3[(t3.LABEL_FLIPPED_NEXT == 1) & t3.reverted.notna()]
# Dominance must be judged on MAGNITUDE, the same way the shares three lines above are.
# The old signed rule (phi_N + phi_P > phi_D) labelled a row "drift-dominated" whenever
# proximity sat BELOW the cohort reference (phi_P < 0), which is most rows -- phi_P
# reaches -2.22 while phi_D is pinned near zero (std 0.012). It reported 863 of 1203
# flips as drift-dominated in a model where drift explains 3% of the risk.
nd = (f2.phi_N.abs() + f2.phi_P.abs()) > f2.phi_D.abs()
n_nd, n_d = int(nd.sum()), int((~nd).sum())
if n_d < 30 or n_nd < 30:
    log(f"OUTCOME    not computable: the magnitude split puts {n_nd} flips on the "
        f"noise/proximity side and {n_d} on the drift side.")
    log(f"           With drift at {aD_/tot*100:.0f}% of attributed risk there is no drift-dominated "
        f"group to compare, so no revert-rate contrast can be claimed.")
else:
    log(f"OUTCOME    flips FCX calls noise/proximity-dominated revert at t+2: {f2[nd].reverted.mean()*100:.0f}% (n={n_nd}) "
        f"| drift-dominated: {f2[~nd].reverted.mean()*100:.0f}% (n={n_d})")

# =========================================================================== examples
OUT = MET / f"fcx_results{SFX}.txt"
OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")          # metrics are safe even if printing fails below
log("\n" + "-" * 78); log("EXAMPLE EXPLANATIONS"); log("-" * 78)
for _, r in c1.sample(3, random_state=1).iterrows():
    log("C1 · " + r.narrative)
for _, r in c2[c2.straddle & (c2.set_size == 2)].sample(3, random_state=2).iterrows():
    ch = r.blame
    log(f"C2 · Set {{TD, PIGD}}: the implied-ratio interval [{np.exp(r.ratio_lo):.2f}, {np.exp(r.ratio_hi):.2f}] straddles the cutoff. "
        f"Responsible channel: {ch} (tremor carries {r.tremor_share*100:.0f}% of the width). "
        + (f"Examining {ch} would settle it." if ch in ('tremor', 'gait') else "Neither channel alone settles it."))
pe = PDNExplainer.narrative
for i in t3[t3.phi_P.notna()].sample(3, random_state=3).index:
    r = t3.loc[i]; e1 = {k: np.array([r[k]]) for k in ("phi_P", "phi_D", "phi_N")}; e1["eta"] = np.array([r["eta_hat"]])
    log("C3 · " + pe(e1, 0, 1 / (1 + np.exp(-zg[i]))))

OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")
c1.drop(columns=[c for c in c1 if c.startswith(("phiT_", "phiP_"))]).to_csv(INT / f"fcx_c1{SFX}.csv", index=False)
c2.to_csv(INT / f"fcx_c2{SFX}.csv", index=False); t3.to_csv(INT / f"fcx_c3{SFX}.csv", index=False)
imp.to_csv(INT / f"fcx_c1_importance{SFX}.csv")
print(f"\nwrote {OUT}")
