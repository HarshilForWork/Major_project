"""AR(1)-SHAP on ADNI -- the second-cohort run that supports the "framework" claim.

Mirrors evaluate_ar1_shap.py's structure and tests exactly (T1-T5), with ADNI's
columns (RID not PATNO), task (binary CN-vs-Dementia, paralleling PPMI's decision to
drop the unstable middle class and report TD-vs-PIGD), and reference clock feature
(AGE_AT_VISIT -- ADNI has no "years since diagnosis" equivalent). The point of this
file existing separately, rather than parameterising evaluate_ar1_shap.py in place, is
the same reason preprocess_adni.py is separate from preprocess.py: lower risk of
breaking the working PPMI pipeline while both are actively used, same as the project's
existing convention (train_baseline.py / train_xgboost.py, splits.py / splits_adni.py).
Folding the two into one parameterised script is future work (HANDOFF_AR1_SHAP.md
section 9.A.2, "wrap in a simple API").

Model explained: the binary CN-vs-Dementia XGBoost (0.98 balanced accuracy, locked
test -- see reports/metrics/adni_baseline_results.txt), trained on development
patients only. Output explained: log-odds of Dementia.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.evaluate_ar1_shap_adni [--quick]
Out:  reports/metrics/ar1_shap_results_adni.txt ; models/ar1_shap_artifacts_adni.pkl
"""
import itertools
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, DMatrix

from trace_pd import config
from trace_pd.evaluation.splits import log_run
from trace_pd.evaluation.splits_adni import split_dev_test
from trace_pd.explain import ar1
from trace_pd.explain.conditional_shapley import (
    InterventionalFiller, PopulationFiller, AR1Filler, explain_many, shapley_instance,
)
from trace_pd.models.backend import make_classifier, fit_classifier

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

QUICK = "--quick" in sys.argv
N_PERM, N_MC = (16, 12) if QUICK else (48, 24)
SYN_PERM, SYN_MC = (16, 12) if QUICK else (64, 48)
N_STAB = 30 if QUICK else 80
STEP = "ar1-adni"
OUT = config.METRICS / "ar1_shap_results_adni.txt"
ID_COL = "RID"
POS, NEG = "Dementia", "CN"                 # positive/negative class
CLOCK_OF_INTEREST = "AGE_AT_VISIT"          # ADNI has no "years since diagnosis" analogue

LOG = []
def log(s=""):
    print(s, flush=True)
    LOG.append(s)

def record(experiment, metric, value, n=None, n_patients=None, note=""):
    if QUICK:
        return
    log_run(STEP, experiment, "xgboost-binary", {"n_perm": N_PERM, "n_mc": N_MC},
            "locked test (explained)", metric, float(value), n=n, n_patients=n_patients, note=note)


# =============================================================================== data
def load():
    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.ADNI_DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
    for c in F:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c], sort=True)[0].astype(float)
            df.loc[df[c] < 0, c] = np.nan
    df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    dev, test = split_dev_test(df)
    return df, dev, test, F


# ============================================================ T2 synthetic ground truth
def synthetic_check(n_pat=300, n_vis=12, n_test=60, seed=7):
    """Identical to evaluate_ar1_shap.py's T2 -- pure simulation, no ADNI data touched.
    Kept here (not imported) so this file is fully self-contained and auditable on
    its own, matching the project's existing duplication convention."""
    rng = np.random.default_rng(seed)
    p = 5
    names = [f"D{j}" for j in range(p)]
    A = 0.75 * np.eye(p) + 0.08 * rng.normal(size=(p, p))
    L = rng.normal(size=(p, p)) * 0.5 + np.eye(p)
    Sig = L @ L.T / p
    w = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    f = lambda Z: np.asarray(Z, float) @ w

    def simulate(n, start_id):
        rows = []
        for i in range(n):
            x = rng.multivariate_normal(np.zeros(p), Sig * 3)
            for t in range(n_vis):
                x = A @ x + rng.multivariate_normal(np.zeros(p), Sig)
                rows.append([start_id + i, 6 * t] + list(x))
        return pd.DataFrame(rows, columns=["PATNO", "VISIT_MONTH"] + names)

    tr, te = simulate(n_pat, 0), simulate(n_test, 10_000)
    P_tr, P_te = ar1.build_pairs(tr, names), ar1.build_pairs(te, names)
    tm = ar1.TransitionModel(names, "full_dt", roles=False).fit(P_tr, tr)
    P_te = P_te.groupby("PATNO").tail(1).reset_index(drop=True)
    X = P_te[names].to_numpy(float)
    Xp = P_te[["PREV_" + c for c in names]].to_numpy(float)

    def exact_ar(x, xp):
        mu = A @ xp
        def v(S):
            S = list(S); H = [j for j in range(p) if j not in S]
            z = x.copy()
            if H:
                if S:
                    K = np.linalg.solve(Sig[np.ix_(S, S)], Sig[np.ix_(H, S)].T).T
                    z[H] = mu[H] + K @ (x[S] - mu[S])
                else:
                    z[H] = mu[H]
            return float(z @ w)
        from math import factorial
        phi = np.zeros(p)
        for j in range(p):
            others = [k for k in range(p) if k != j]
            for r in range(p):
                for S in itertools.combinations(others, r):
                    wt = factorial(r) * factorial(p - r - 1) / factorial(p)
                    phi[j] += wt * (v(S + (j,)) - v(S))
        return phi

    true = np.array([exact_ar(X[i], Xp[i]) for i in range(len(X))])
    est, _ = explain_many(f, X, AR1Filler(tm, names), [P_te.iloc[[i]] for i in range(len(X))],
                          n_perm=SYN_PERM, n_mc=SYN_MC, seed=1)

    class TrueParams:
        name = "true-parameter"
        def for_instance(self, x, xp):
            self.mu = A @ xp; return self
        def chain_state(self, n, r):
            return r.standard_normal((n, p))
        def fill(self, x, shown, n, r, state=None):
            from trace_pd.explain.conditional_shapley import gauss_conditional_draw
            rows = np.tile(x, (n, 1)); h = np.flatnonzero(~shown); k = np.flatnonzero(shown)
            rows[:, h] = gauss_conditional_draw(self.mu, Sig, k, x[k], h, n, r, state)
            return rows
    est_true, _ = explain_many(f, X, TrueParams(), list(Xp), n_perm=SYN_PERM, n_mc=SYN_MC, seed=1)
    stat_mean = tr[names].mean().to_numpy()
    est_int, _ = explain_many(f, X, InterventionalFiller(tr[names].to_numpy(float)),
                              n_perm=SYN_PERM, n_mc=SYN_MC, seed=1)
    true_int = (X - stat_mean) * w
    r = np.corrcoef(est.ravel(), true.ravel())[0, 1]
    rel = np.abs(est - true).mean() / np.abs(true).mean()
    r_int = np.corrcoef(est_int.ravel(), true_int.ravel())[0, 1]
    r_true = np.corrcoef(est_true.ravel(), true.ravel())[0, 1]
    rel_true = np.abs(est_true - true).mean() / np.abs(true).mean()
    return dict(r=r, rel=rel, r_int=r_int, r_true=r_true, rel_true=rel_true,
                dummy_true=np.abs(true[:, 3]).mean(), dummy_est=np.abs(est[:, 3]).mean(),
                dummy_int=np.abs(est_int[:, 3]).mean(), mean_abs_true=np.abs(true).mean(), n=len(X))


# ======================================================================== T3 realism
def realism(fillers, X, ctxs, Xprev, groups, dyn_idx, seed=11):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    shown_masks = []
    for i in range(n):
        s = rng.random(d) < 0.5
        s |= np.isnan(X[i])
        shown_masks.append(s)
    out = {}
    for fl in fillers:
        r2 = np.random.default_rng(seed + 1)
        synth = np.zeros_like(X)
        for i in range(n):
            fl.for_instance(X[i], ctxs[i])
            synth[i] = fl.fill(X[i], shown_masks[i], 1, r2)[0]
        res = {}
        for tag, cols in (("all features", np.arange(d)), ("changing only", dyn_idx)):
            def feats(Z):
                return np.column_stack([Xprev[:, cols], Z[:, cols], Z[:, cols] - Xprev[:, cols]])
            D = np.vstack([feats(X), feats(synth)])
            y = np.r_[np.zeros(n), np.ones(n)]
            g = np.r_[groups, groups]
            oof = np.zeros(2 * n)
            for tr, te in GroupKFold(5).split(D, y, g):
                m = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                                  subsample=0.8, random_state=0, n_jobs=-1, tree_method="hist")
                m.fit(D[tr], y[tr]); oof[te] = m.predict_proba(D[te])[:, 1]
            res[tag] = roc_auc_score(y, oof)
        hid = np.array([~s for s in shown_masks])
        ratios = []
        for j in dyn_idx:
            ok = hid[:, j] & ~np.isnan(Xprev[:, j]) & ~np.isnan(X[:, j])
            real = np.abs(X[ok, j] - Xprev[ok, j]).mean()
            # nan-aware: a fallback (non-complete) background row can leave a NaN in
            # synth at a column not being tested here, but never at THIS j when ok=True
            # requires X/Xprev non-null -- still, be robust rather than let one NaN
            # silently poison the whole median via plain .mean()
            fake = np.nanmean(np.abs(synth[ok, j] - Xprev[ok, j]))
            if ok.sum() > 20 and real > 0 and np.isfinite(fake):
                ratios.append(fake / real)
        res["jump ratio"] = float(np.median(ratios))
        out[fl.name] = res
    return out


# ============================================================================== main
def main():
    t0 = time.time()
    df, dev, test, F = load()
    Pd, Pt = ar1.build_pairs(dev, F, id_col=ID_COL), ar1.build_pairs(test, F, id_col=ID_COL)
    STATIC, CLOCK, DYN, role_report = ar1.detect_feature_roles(dev, F, id_col=ID_COL)
    dyn_idx = np.array([F.index(f) for f in DYN])
    log("=" * 78)
    log("AR(1)-SHAP on ADNI -- second-cohort evaluation" + ("  [QUICK MODE]" if QUICK else ""))
    log("=" * 78)
    log(f"features: {len(F)}  (fixed {len(STATIC)}: {STATIC}; clock {len(CLOCK)}: {CLOCK}; changing {len(DYN)})")
    log("Feature roles were DETECTED from the data (ar1.detect_feature_roles), not hard-coded:")
    for f in STATIC + CLOCK:
        log(f"  {f:<28} {role_report[f]}")
    log(f"Shapley sampling: {N_PERM} permutations x {N_MC} draws per coalition")

    # ---------------------------------------------------------------- the black box
    s = dev[dev.LABEL.isin([NEG, POS])]
    clf = fit_classifier(make_classifier("xgb", 2), s[F].to_numpy(float), (s.LABEL == POS).astype(int))
    f = lambda Z: clf.predict(np.asarray(Z, float), output_margin=True)
    tt = test[test.LABEL.isin([NEG, POS])]
    ba = balanced_accuracy_score((tt.LABEL == POS).astype(int), clf.predict(tt[F].to_numpy(float)))
    log(f"\nModel explained: binary XGBoost ({NEG} vs {POS}), dev patients only. "
        f"Locked-test balanced accuracy {ba:.3f} ({len(tt)} visits) -- matches "
        f"reports/metrics/adni_baseline_results.txt (0.980).")

    # ------------------------------------------------------------------- T1 transition
    log("\n" + "-" * 78)
    log("T1  Does the patient's own past predict the next visit?")
    log("-" * 78)
    log(f"visit pairs: development {len(Pd)}, locked test {len(Pt)}")
    tms = {}
    log(f"{'model':<44}{'held-out log-lik':>17}{'mean R2':>9}")
    labels = {"population": "population only (no history)",
              "scalar": "own previous value only (per feature)",
              "full": "all previous values (fixed gap)",
              "full_dt": "all previous values + gap length  [USED]"}
    for v in ["population", "scalar", "full", "full_dt"]:
        tm = ar1.TransitionModel(F, v, id_col=ID_COL).fit(Pd, dev)
        assert not (tm.patients_seen_ & set(test[ID_COL])), "locked-test patient used in fitting"
        tms[v] = tm
        ll = tm.loglik(Pt)
        r2 = np.nanmean(tm.r2(Pt)) if v != "population" else 0.0
        log(f"{labels[v]:<44}{ll:>17.4f}{r2:>9.3f}")
        record("T1 transition fit", f"loglik_{v}", ll, n=len(Pt), n_patients=Pt[ID_COL].nunique())
    tm = tms["full_dt"]
    b = ar1.gap_bucket(Pt.DT)
    log("\nby gap (held-out log-lik): " + ", ".join(
        f"{name} n={int((b == k).sum())}: fixed-gap {tms['full'].loglik(Pt[b == k]):.3f} vs gap-aware {tm.loglik(Pt[b == k]):.3f}"
        for k, name in enumerate(["~3 mo", "~6 mo", ">=9 mo"]) if (b == k).sum() > 20))
    sds = {k: np.sqrt(np.diag(tm.cov_[k])).mean() for k in range(3)}
    log(f"average leftover spread by gap: ~3 mo {sds[0]:.3f}, ~6 mo {sds[1]:.3f}, >=9 mo {sds[2]:.3f}"
        f"  (ridge alpha {tm.alpha_}, shrinkage {tm.lam_})")
    tm_learn = ar1.TransitionModel(F, "full_dt", roles=False, id_col=ID_COL).fit(Pd, dev)

    # ------------------------------------------------------------------ T2 synthetic
    log("\n" + "-" * 78)
    log("T2  Is the estimator correct? (simulated patients, true answer computed exactly)")
    log("-" * 78)
    sy = synthetic_check()
    log(f"AR(1)-SHAP estimate vs exact truth: correlation {sy['r']:.3f}, average error "
        f"{100 * sy['rel']:.1f}% of the average true value ({sy['n']} visits x 5 features)")
    log(f"  same estimator given the TRUE parameters: correlation {sy['r_true']:.3f}, average error "
        f"{100 * sy['rel_true']:.1f}%")
    log(f"standard SHAP estimate vs its own exact truth: correlation {sy['r_int']:.3f}")
    log(f"unused feature D3 (model weight 0, but correlated with used ones): true AR(1)-SHAP credit "
        f"{sy['dummy_true']:.3f}, estimated {sy['dummy_est']:.3f}, standard SHAP {sy['dummy_int']:.3f}"
        f"  (average true credit of a feature: {sy['mean_abs_true']:.3f})")
    record("T2 synthetic", "corr_est_vs_exact", sy["r"], n=sy["n"])
    record("T2 synthetic", "rel_abs_error", sy["rel"], n=sy["n"])

    # --------------------------------------------------------------- explain test visits
    tl = Pt[Pt.LABEL.isin([NEG, POS])].reset_index(drop=True)
    if QUICK:
        tl = tl.head(60)
    X = tl[F].to_numpy(float)
    Xprev = tl[["PREV_" + c for c in F]].to_numpy(float)
    ctxs = [tl.iloc[[i]] for i in range(len(tl))]
    groups = tl[ID_COL].to_numpy()
    # See evaluate_ar1_shap.py for why this falls back instead of requiring dropna():
    # ADNI's instruments rotate across visit types so heavily that not one visit has
    # every one of the 23 cheap features present (max observed: 21/23).
    B_complete = s[F].dropna()
    USE_COMPLETE_BG = len(B_complete) >= 50
    B = B_complete.to_numpy(float) if USE_COMPLETE_BG else s[F].to_numpy(float)
    fillers = {
        "standard SHAP": InterventionalFiller(B),
        "population-conditional": PopulationFiller(dev[F].to_numpy(float)),
        "AR(1)-SHAP": AR1Filler(tm, F),
        "AR(1)-SHAP, roles learned": AR1Filler(tm_learn, F),
    }
    for k, fl in fillers.items():
        fl.name = k
    log(f"\nExplaining {len(tl)} locked-test visits ({tl[ID_COL].nunique()} patients); background "
        f"for standard SHAP = {len(B)} {'complete' if USE_COMPLETE_BG else 'development (NaN included)'} visits")

    # ------------------------------------------------------------------- T3 realism
    log("\n" + "-" * 78)
    log("T3  Are the filled-in visits realistic? ('Frankenstein patient' test)")
    log("-" * 78)
    rz = realism(list(fillers.values()), X, ctxs, Xprev, groups, dyn_idx)
    log(f"{'filler':<30}{'detectable (all)':>18}{'detectable (changing)':>24}{'jump ratio':>12}")
    for k, r in rz.items():
        log(f"{k:<30}{r['all features']:>18.3f}{r['changing only']:>24.3f}{r['jump ratio']:>12.2f}")
        record("T3 realism", f"disc_auc_all[{k}]", r["all features"], n=len(tl))
        record("T3 realism", f"disc_auc_dyn[{k}]", r["changing only"], n=len(tl))
        record("T3 realism", f"jump_ratio[{k}]", r["jump ratio"], n=len(tl))
    log("detectable = AUROC of a classifier telling real from filled visits (0.5 = looks real).")
    log("jump ratio = typical visit-to-visit jump in filled values / in real values (1.0 = realistic).")

    # --------------------------------------------------------------- T4 explanations
    log("\n" + "-" * 78)
    log("T4  How do the explanations change?")
    log("-" * 78)
    PHI, BASE = {}, {}
    for k, fl in fillers.items():
        t1 = time.time()
        PHI[k], BASE[k] = explain_many(f, X, fl, ctxs, N_PERM, N_MC, seed=0, progress=250)
        eff = np.nanmax(np.abs(PHI[k].sum(1) - (f(X) - BASE[k])))
        log(f"  {k:<28} {time.time() - t1:6.0f}s   max add-up error {eff:.1e}")
    contribs = clf.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
    PHI["TreeSHAP (default shap)"] = contribs

    mean_abs = pd.DataFrame({k: np.abs(v).mean(0) for k, v in PHI.items()}, index=F)
    share = mean_abs / mean_abs.sum() * 100
    order = share["standard SHAP"].sort_values(ascending=False).index
    cols = list(PHI)
    short = {"standard SHAP": "standard", "population-conditional": "pop-cond",
             "AR(1)-SHAP": "AR(1)", "AR(1)-SHAP, roles learned": "AR(1)-learn",
             "TreeSHAP (default shap)": "TreeSHAP"}
    log("\nShare of total credit, % (mean |contribution| over visits):")
    log(f"{'feature':<26}" + "".join(f"{short[c]:>12}" for c in cols))
    for feat in order:
        log(f"{feat:<26}" + "".join(f"{share.loc[feat, c]:>12.1f}" for c in cols))
    log(f"{'total |credit| (log-odds)':<26}" + "".join(f"{mean_abs[c].sum():>12.3f}" for c in cols))
    gap = {k: np.abs(PHI[k].sum(1)).mean() for k in fillers}
    log("average |prediction - starting point|: " + ", ".join(f"{short[k]} {v:.3f}" for k, v in gap.items()))

    log(f"\n{CLOCK_OF_INTEREST} (ADNI's clock feature, no 'years since diagnosis' equivalent exists), "
        "change against standard SHAP:")
    for k in ["population-conditional", "AR(1)-SHAP", "AR(1)-SHAP, roles learned"]:
        raw = 100 * (mean_abs.loc[CLOCK_OF_INTEREST, k] / mean_abs.loc[CLOCK_OF_INTEREST, "standard SHAP"] - 1)
        sh = 100 * (share.loc[CLOCK_OF_INTEREST, k] / share.loc[CLOCK_OF_INTEREST, "standard SHAP"] - 1)
        rank = list(share[k].sort_values(ascending=False).index).index(CLOCK_OF_INTEREST) + 1
        log(f"  {k:<28} raw credit {raw:+6.1f}%   share of credit {sh:+6.1f}%   rank {rank}")
        record("T4 clock feature", f"raw_change_pct[{k}]", raw, n=len(tl))
        record("T4 clock feature", f"share_change_pct[{k}]", sh, n=len(tl))
    r_std = list(share["standard SHAP"].sort_values(ascending=False).index).index(CLOCK_OF_INTEREST) + 1
    log(f"  (rank under standard SHAP: {r_std})")

    grp = {"fixed (sex, handedness, education)": STATIC, "clock (age)": CLOCK, f"changing ({len(DYN)})": DYN}
    log("\nCredit by feature type, % of total:")
    log(f"{'type':<36}" + "".join(f"{short[c]:>12}" for c in cols))
    for g, fs in grp.items():
        log(f"{g:<36}" + "".join(f"{share.loc[fs, c].sum():>12.1f}" for c in cols))

    log("\nMost common top driver per visit (% of visits):")
    for k in cols:
        top = pd.Series(np.array(F)[np.abs(PHI[k]).argmax(1)]).value_counts(normalize=True).head(3) * 100
        log(f"  {short[k]:<12} " + ", ".join(f"{i} {v:.0f}%" for i, v in top.items()))

    def per_visit_rho(a, b, cols=None):
        out = []
        for i in range(len(a)):
            ok = ~np.isnan(X[i])
            if cols is not None:
                ok &= np.isin(np.arange(len(F)), cols)
            if ok.sum() > 3:
                r = spearmanr(np.abs(a[i, ok]), np.abs(b[i, ok])).correlation
                if np.isfinite(r):
                    out.append(r)
        return np.mean(out)
    log("\nPer-visit agreement of rankings (Spearman; 1 = same order):")
    for a, b_ in [("standard SHAP", "AR(1)-SHAP"), ("population-conditional", "AR(1)-SHAP"),
                  ("standard SHAP", "population-conditional"), ("AR(1)-SHAP", "AR(1)-SHAP, roles learned")]:
        log(f"  {short[a]} vs {short[b_]}: all features {per_visit_rho(PHI[a], PHI[b_]):.3f}, "
            f"changing features only {per_visit_rho(PHI[a], PHI[b_], dyn_idx):.3f}")
    top_std = np.abs(PHI["standard SHAP"]).argmax(1); top_ar = np.abs(PHI["AR(1)-SHAP"]).argmax(1)
    flips = (top_std != top_ar).mean()
    a_s, a_a = np.abs(PHI["standard SHAP"][:, dyn_idx]), np.abs(PHI["AR(1)-SHAP"][:, dyn_idx])
    flips_dyn = (a_s.argmax(1) != a_a.argmax(1)).mean()
    log(f"top driver differs between standard SHAP and AR(1)-SHAP on {100 * flips:.1f}% of visits; "
        f"top CHANGING-feature driver differs on {100 * flips_dyn:.1f}%")
    record("T4 divergence", "top1_flip_rate_changing_only", flips_dyn, n=len(tl))
    record("T4 divergence", "top1_flip_rate", flips, n=len(tl))

    ac = []
    dd_ = dev.sort_values([ID_COL, "VISIT_MONTH"])
    for f_ in DYN:
        x = dd_[f_]; lag = x.groupby(dd_[ID_COL]).shift(1); ok = x.notna() & lag.notna()
        ac.append(np.corrcoef(x[ok], lag[ok])[0, 1])
    chg = np.log(share.loc[DYN, "AR(1)-SHAP"] / share.loc[DYN, "population-conditional"])
    rho = spearmanr(ac, chg).correlation
    log(f"\nDoes history matter more for features that carry over more? Spearman between a feature's "
        f"visit-to-visit carry-over and its change in share (AR(1) vs population-conditional): {rho:+.3f}")
    record("T4 carry-over link", "spearman_autocorr_vs_share_change", rho, n=len(DYN))

    surprise = np.abs(tm.cop_dyn.to_normal(ar1.numeric_matrix(tl, DYN)) - np.vstack([tm.predict_mean(c) for c in ctxs]))
    def surprise_rho(P):
        a = np.abs(P[:, dyn_idx]); out = []
        for i in range(len(X)):
            ok = ~np.isnan(surprise[i])
            if ok.sum() > 3:
                r = spearmanr(a[i, ok], surprise[i, ok]).correlation
                if np.isfinite(r):
                    out.append(r)
        return np.mean(out)
    log("Does credit go to the features that moved unexpectedly since the last visit? Per-visit "
        "Spearman between |credit| and |surprise|: " + ", ".join(
            f"{short[k]} {surprise_rho(PHI[k]):+.3f}" for k in fillers))

    # ---------------------------------------------------------------- T5 stability
    log("\n" + "-" * 78)
    log(f"T5  Are the explanations stable? (same {N_STAB} visits, different random seed)")
    log("-" * 78)
    for k, fl in fillers.items():
        P2, _ = explain_many(f, X[:N_STAB], fl, ctxs[:N_STAB], N_PERM, N_MC, seed=99)
        rho = per_visit_rho(PHI[k][:N_STAB], P2)
        log(f"  {k:<28} seed-to-seed ranking agreement {rho:.3f}")
        record("T5 stability", f"seed_rho[{k}]", rho, n=N_STAB)

    with open(config.MODELS / "ar1_shap_artifacts_adni.pkl", "wb") as fh:
        pickle.dump({"classifier": clf, "features": F, "transition_model": tm,
                     "transition_model_learned_roles": tm_learn, "attributions": PHI,
                     "base_values": BASE, "visits": tl[[ID_COL, "VISIT_MONTH", "DT"]]}, fh)
    log(f"\nRuntime {(time.time() - t0) / 60:.1f} min. Attributions saved to models/ar1_shap_artifacts_adni.pkl")
    if not QUICK:
        OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")
        print(f"written {OUT}")


if __name__ == "__main__":
    main()
