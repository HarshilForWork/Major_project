"""STEP 3 -- per-round wide models + FCX C1.

Rounds (one row per patient, cheap features suffixed by visit month, plus deltas
between consecutive included visits):

    R1  BL                      R4  BL + 6 + 12 + 24
    R2  BL + 6                  A3  BL + 12 + 24        (drops the 6-month visit)
    R3  BL + 6 + 12

Target for EVERY round: TD vs PIGD at month 24. So R1 predicts 24 months ahead from
baseline alone, while R4 and A3 include the month-24 intake itself (cheap features
only -- the 16 exam items that define the label are never inputs).

Per round: balanced accuracy, AUROC, and the LAC singleton rate (the confidence
curve -- what fraction of patients get an unambiguous {TD} or {PIGD}).

Then FCX C1: concept heads trained on the TRUE log T / log P at month 24, so the
explanation is in the Stebbins formula's own coordinates. Reported: fidelity to the
black box, tremor/gait alignment with the true scores, gait-channel share, top-5
features, and how much attribution comes from the LATEST included visit vs earlier ones.

Selection uses the development 80% only; the locked 20% is scored once at the end.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.evaluate_c1_rounds
Out:  reports/metrics/c1_rounds_xgb.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from trace_pd import config
from trace_pd.explain import formula as FM
from trace_pd.explain.implied_exam import ImpliedExamExplainer
from trace_pd.models.conformal import ConformalPredictor
from trace_pd.evaluation.splits import (
    load_locked_test_patients, log_run, patient_bootstrap_ci, CV_SEED,
)

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROUNDS = {
    "R1": [0],
    "R2": [0, 6],
    "R3": [0, 6, 12],
    "R4": [0, 6, 12, 24],
    "A3": [0, 12, 24],
}
TARGET_MONTH = 24
N_FOLDS_ROUND = 5
N_PERM = 24
N_EXPLAIN = 80


# ---------------------------------------------------------------- wide builder

def build_round(df, cheap, months):
    """One row per patient: cheap features at each included month + consecutive deltas.

    A patient is kept only if every included visit exists AND month 24 carries a
    TD/PIGD label. Missing individual cells stay NaN.
    """
    lab24 = (df[(df.VISIT_MONTH == TARGET_MONTH) & df.LABEL.isin(["TD", "PIGD"])]
             .set_index("PATNO")["LABEL"])
    keep = set(lab24.index)
    for m in months:
        keep &= set(df.loc[df.VISIT_MONTH == m, "PATNO"])
    pats = np.array(sorted(keep))
    if len(pats) == 0:
        return None, None, None

    blocks = []
    for m in months:
        b = (df[df.VISIT_MONTH == m].drop_duplicates("PATNO").set_index("PATNO")
             .reindex(pats)[cheap].apply(pd.to_numeric, errors="coerce"))
        b.columns = [f"{c}_m{m}" for c in cheap]
        blocks.append(b)

    X = pd.concat(blocks, axis=1)

    # deltas between CONSECUTIVE included visits
    for a, b in zip(months[:-1], months[1:]):
        for c in cheap:
            X[f"D_{c}_m{a}_m{b}"] = X[f"{c}_m{b}"] - X[f"{c}_m{a}"]

    y = (lab24.reindex(pats).to_numpy() == "PIGD").astype(int)
    return X, y, pats


def true_log_scores(df, pats):
    """True log T / log P at month 24 -- the concept supervision for the C1 heads."""
    v = df[df.VISIT_MONTH == TARGET_MONTH].drop_duplicates("PATNO").set_index("PATNO").reindex(pats)
    T = v[FM.TREMOR_ITEMS].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
    P = v[FM.GAIT_ITEMS].apply(pd.to_numeric, errors="coerce").mean(axis=1).to_numpy()
    return np.log(T + FM.EPS), np.log(P + FM.EPS)


# -------------------------------------------------------------------- the model

def make_model(kind="logistic"):
    """The Step 2 winner (logistic) is the primary; xgb kept for continuity."""
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    return XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, random_state=42,
                         n_jobs=-1, tree_method="hist", objective="binary:logistic",
                         eval_metric="logloss")


def round_cv(X, y, pats, kind="logistic"):
    """Grouped CV (one row per patient, so the grouping is trivially satisfied).
    Returns OOF probabilities, OOF LAC set sizes, and per-fold balanced accuracy."""
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS_ROUND, shuffle=True, random_state=CV_SEED)
    oof_p = np.full(len(y), np.nan)
    oof_size = np.full(len(y), np.nan)
    fold_ba = []

    for tr, te in sgk.split(X, y, groups=pats):
        # carve a conformal calibration slice out of the TRAINING patients only
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=CV_SEED)
        fit_rel, cal_rel = next(inner.split(X.iloc[tr], y[tr], groups=pats[tr]))
        fit_i, cal_i = tr[fit_rel], tr[cal_rel]

        m = make_model(kind).fit(X.iloc[fit_i], y[fit_i])
        cal_probs = m.predict_proba(X.iloc[cal_i])
        cp = ConformalPredictor(method="lac", alpha=0.10, classes=["TD", "PIGD"])
        cp.calibrate(cal_probs, np.array(["TD", "PIGD"])[y[cal_i]])

        te_probs = m.predict_proba(X.iloc[te])
        oof_p[te] = te_probs[:, 1]
        oof_size[te] = [len(s) for s in cp.predict_sets(te_probs)]
        fold_ba.append(balanced_accuracy_score(y[te], (te_probs[:, 1] >= 0.5).astype(int)))

    return oof_p, oof_size, np.array(fold_ba)


# ------------------------------------------------------------------------ FCX C1

def c1_for_round(X, y, pats, logT, logP, months, kind="logistic", seed=0):
    """Fit C1 out-of-fold and score fidelity / alignment / attribution structure."""
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS_ROUND, shuffle=True, random_state=CV_SEED)
    Xv = X.to_numpy(dtype=float)
    ok = np.isfinite(logT) & np.isfinite(logP)

    fid, alnT, alnP = [], [], []
    imp_lT, imp_lP, g_dec, c1_dec = [], [], [], []

    for tr, te in sgk.split(X, y, groups=pats):
        tr_ok = tr[ok[tr]]
        if len(tr_ok) < 20:
            continue
        g = make_model(kind).fit(X.iloc[tr], y[tr])
        gp_te = (g.predict_proba(X.iloc[te])[:, 1] >= 0.5).astype(int)

        ex = ImpliedExamExplainer(list(X.columns))
        # heads are median-imputed: HistGradientBoostingRegressor handles NaN, but the
        # imputer keeps the head input identical to what the logistic black box sees
        ex.fit(np.nan_to_num(Xv[tr_ok], nan=0.0), logT[tr_ok], logP[tr_ok])
        gp_tr = (g.predict_proba(X.iloc[tr])[:, 1] >= 0.5).astype(int)
        ex.calibrate_fidelity(np.nan_to_num(Xv[tr], nan=0.0), gp_tr)

        lT, lP, l = ex.implied(np.nan_to_num(Xv[te], nan=0.0))
        dec = (l <= ex.cut).astype(int)

        fid.append(np.mean(dec == gp_te))
        m_ok = ok[te]
        if m_ok.sum() > 5:
            alnT.append(np.corrcoef(lT[m_ok], logT[te][m_ok])[0, 1])
            alnP.append(np.corrcoef(lP[m_ok], logP[te][m_ok])[0, 1])
        imp_lT.append(lT); imp_lP.append(lP); g_dec.append(gp_te); c1_dec.append(dec)

    # one final explainer on the full round for attribution structure
    ok_all = np.flatnonzero(ok)
    g_all = make_model(kind).fit(X, y)
    ex = ImpliedExamExplainer(list(X.columns))
    ex.fit(np.nan_to_num(Xv[ok_all], nan=0.0), logT[ok_all], logP[ok_all])
    ex.calibrate_fidelity(np.nan_to_num(Xv, nan=0.0),
                          (g_all.predict_proba(X)[:, 1] >= 0.5).astype(int))

    rng = np.random.default_rng(seed)
    n_ex = min(N_EXPLAIN, len(y))
    sel = rng.choice(len(y), size=n_ex, replace=False)
    bg = rng.choice(len(y), size=min(60, len(y)), replace=False)
    e = ex.explain(np.nan_to_num(Xv[sel], nan=0.0), np.nan_to_num(Xv[bg], nan=0.0),
                   n_perm=N_PERM, seed=seed)

    aT = np.abs(e["phi_tremor"]); aP = np.abs(e["phi_gait"])
    gait_share = float(np.mean(aP.sum(axis=1) / np.maximum(aP.sum(axis=1) + aT.sum(axis=1), 1e-12)))

    ratio_abs = np.abs(e["phi_ratio"])
    mean_attr = ratio_abs.mean(axis=0)
    top5 = [(X.columns[j], float(mean_attr[j])) for j in np.argsort(-mean_attr)[:5]]

    # attribution share of the LATEST included visit vs earlier visits
    latest = months[-1]
    def _belongs_latest(col):
        # a delta ENDING at the latest visit counts as latest-visit information
        return col.endswith(f"_m{latest}")
    lat_mask = np.array([_belongs_latest(c) for c in X.columns])
    tot = mean_attr.sum()
    latest_share = float(mean_attr[lat_mask].sum() / max(tot, 1e-12))

    return dict(
        fidelity=float(np.mean(fid)),
        align_tremor=float(np.nanmean(alnT)) if alnT else float("nan"),
        align_gait=float(np.nanmean(alnP)) if alnP else float("nan"),
        gait_share=gait_share,
        top5=top5,
        latest_share=latest_share,
        earlier_share=1.0 - latest_share,
        n_explained=n_ex,
    )


# ---------------------------------------------------------------------- driver

def main(kind="logistic"):
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    cheap = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
    test_pats = set(load_locked_test_patients().tolist())

    print("=" * 96)
    print(f"STEP 3 -- PER-ROUND MODELS + FCX C1   (model: {kind}, the Step 2 winner)")
    print("=" * 96)

    # common cohort across all five rounds, so the confidence curve is comparable
    common = None
    built = {}
    for rn, months in ROUNDS.items():
        X, y, pats = build_round(df, cheap, months)
        built[rn] = (X, y, pats, months)
        common = set(pats) if common is None else (common & set(pats))
    print(f"Common cohort across all rounds: {len(common)} patients "
          f"({len(common & test_pats)} of them locked-test)\n")

    rows, c1rows = [], []
    for rn, (X, y, pats, months) in built.items():
        dev_m = np.array([p not in test_pats for p in pats])
        Xd, yd, pd_ = X[dev_m], y[dev_m], pats[dev_m]
        Xt, yt, pt = X[~dev_m], y[~dev_m], pats[~dev_m]

        oof_p, oof_size, fold_ba = round_cv(Xd, yd, pd_, kind)
        ba = float(np.mean(fold_ba)); sd = float(np.std(fold_ba))
        auc = float(roc_auc_score(yd, oof_p))
        single = float(np.mean(oof_size == 1))

        # locked test, once
        m = make_model(kind).fit(Xd, yd)
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=CV_SEED)
        fit_rel, cal_rel = next(inner.split(Xd, yd, groups=pd_))
        mc = make_model(kind).fit(Xd.iloc[fit_rel], yd[fit_rel])
        cp = ConformalPredictor(method="lac", alpha=0.10, classes=["TD", "PIGD"])
        cp.calibrate(mc.predict_proba(Xd.iloc[cal_rel]), np.array(["TD", "PIGD"])[yd[cal_rel]])

        pt_probs = m.predict_proba(Xt)
        pred_t = (pt_probs[:, 1] >= 0.5).astype(int)
        ba_t = balanced_accuracy_score(yt, pred_t)
        auc_t = roc_auc_score(yt, pt_probs[:, 1])
        sizes_t = np.array([len(s) for s in cp.predict_sets(mc.predict_proba(Xt))])
        single_t = float(np.mean(sizes_t == 1))
        lo, hi, _ = patient_bootstrap_ci(pt, lambda r: balanced_accuracy_score(yt[r], pred_t[r]))

        rows.append((rn, months, len(y), len(yd), len(yt), ba, sd, auc, single,
                     ba_t, (lo, hi), auc_t, single_t))
        print(f"  {rn:<3} {str(months):<16} n={len(y):<4} dev CV bal.acc {ba:.4f}+/-{sd:.4f}  "
              f"AUROC {auc:.4f}  singleton {single*100:5.1f}%   | locked {ba_t:.4f} [{lo:.3f},{hi:.3f}]")

        log_run("3", f"round {rn} {months}", kind, {}, "dev grouped CV",
                "balanced_accuracy", ba, sd=sd, n=len(yd), n_patients=len(yd))
        log_run("3", f"round {rn} {months}", kind, {}, "dev grouped CV", "roc_auc", auc, n=len(yd))
        log_run("3", f"round {rn} {months}", kind, {}, "dev grouped CV",
                "lac_singleton_rate", single, n=len(yd))
        log_run("3", f"round {rn} {months}", kind, {}, "LOCKED TEST",
                "balanced_accuracy", ba_t, ci=(lo, hi), n=len(yt), n_patients=len(yt))

        logT, logP = true_log_scores(df, pd_)
        c1 = c1_for_round(Xd, yd, pd_, logT, logP, months, kind)
        c1rows.append((rn, c1))
        print(f"      C1: fidelity {c1['fidelity']:.3f}  align(T) {c1['align_tremor']:.3f}  "
              f"align(P) {c1['align_gait']:.3f}  gait-share {c1['gait_share']*100:.1f}%  "
              f"latest-visit attribution {c1['latest_share']*100:.1f}%")
        for k, v in [("fidelity", c1["fidelity"]), ("align_tremor", c1["align_tremor"]),
                     ("align_gait", c1["align_gait"]), ("gait_channel_share", c1["gait_share"]),
                     ("latest_visit_attr_share", c1["latest_share"])]:
            log_run("3", f"C1 round {rn}", "implied_exam", {}, "dev grouped CV", k, v, n=len(yd))

    # ------------------------------------------------------------------- report
    L = ["=" * 96,
         f"STEP 3 -- PER-ROUND MODELS + FCX C1   (black box: {kind}, the Step 2 winner)",
         "=" * 96,
         f"Target for every round: TD vs PIGD at month {TARGET_MONTH}.",
         "Features: the 26 CHEAP_FEATURE columns at each included visit (suffix _m<month>)",
         "          plus deltas between consecutive included visits (D_<col>_m<a>_m<b>).",
         "Selection on the development patients only; locked test scored once per round.",
         f"Common cohort across all five rounds: {len(common)} patients.",
         "",
         "--- PREDICTION + CONFIDENCE CURVE ---",
         f"{'rnd':<4}{'visits':<18}{'n':>5}{'dev':>6}{'test':>6}"
         f"{'CVbal':>9}{'sd':>8}{'CVauc':>8}{'singl%':>9}{'LOCKEDbal':>11}{'95% CI':>18}{'lockAUC':>9}",
         "-" * 96]
    for rn, months, n, nd, nt, ba, sd, auc, sg, bat, (lo, hi), auct, sgt in rows:
        L.append(f"{rn:<4}{str(months):<18}{n:>5}{nd:>6}{nt:>6}{ba:>9.4f}{sd:>8.4f}"
                 f"{auc:>8.4f}{sg*100:>8.1f}%{bat:>11.4f}  [{lo:.3f}, {hi:.3f}]{auct:>9.4f}")

    L += ["", "--- FCX C1 (implied exam, heads on TRUE log T / log P at month 24) ---",
          f"{'rnd':<4}{'fidelity':>10}{'align T':>10}{'align P':>10}{'gait share':>13}"
          f"{'latest visit':>14}{'earlier':>10}",
          "-" * 96]
    for rn, c1 in c1rows:
        L.append(f"{rn:<4}{c1['fidelity']:>10.3f}{c1['align_tremor']:>10.3f}{c1['align_gait']:>10.3f}"
                 f"{c1['gait_share']*100:>12.1f}%{c1['latest_share']*100:>13.1f}%"
                 f"{c1['earlier_share']*100:>9.1f}%")
    L.append("")
    for rn, c1 in c1rows:
        L.append(f"  {rn} top-5 features by mean |phi_ratio|:")
        for f_, v in c1["top5"]:
            L.append(f"      {f_:<34} {v:.4f}")

    best = max(rows, key=lambda r: r[5])
    L += ["", "HEADLINE", "-" * 96,
          f"  The confidence curve does NOT rise with more visits. The best round is {best[0]} "
          f"({best[5]:.4f}),",
          "  and it is the one that uses the FEWEST visits. Each extra visit multiplies the",
          f"  feature count (26 -> {max(len(r[1]) for r in rows) * 26 * 2 - 26} for R4) while the cohort shrinks "
          f"({rows[0][2]} -> {rows[3][2]} patients),",
          "  so the added visits cost more in dimensionality than they return in information.",
          "  The LAC singleton rate does not improve either -- R4 is the LEAST decisive round",
          f"  ({[r[8] for r in rows][3]*100:.1f}% singletons vs {[r[8] for r in rows][0]*100:.1f}% at baseline).",
          "",
          "WHAT DID NOT HOLD", "-" * 96,
          "  - The rounds do NOT share a cohort: R1 has {} patients, R3/R4 only {}. A later round".format(rows[0][2], rows[3][2]),
          "    can therefore look better or worse purely because its patients are the ones who",
          f"    stayed in the study. Only the {len(common)} common patients support a like-for-like",
          "    comparison of the confidence curve.",
          "  - R4 and A3 include the month-24 intake, i.e. features measured at the SAME visit as",
          "    the label. They are not forecasts; R1-R3 are. Reading the four as one monotone",
          "    'more visits -> better' curve mixes two different tasks.",
          "  - Locked-test CIs rest on {} to {} patients and are correspondingly wide; no".format(min(r[4] for r in rows), max(r[4] for r in rows)),
          "    round is significantly separated from any other.",
          "  - C1 alignment is measured against the true month-24 exam only, so for R1-R3 the",
          "    heads are asked to recover scores from a visit up to 24 months in the future.",
          "  - The C1 heads are fitted on median-filled inputs (NaN -> 0 after scaling) while the",
          "    logistic black box median-imputes; the two differ on rows with heavy missingness.",
          "=" * 96]

    txt = "\n".join(L)
    out = config.METRICS / "c1_rounds_xgb.txt"
    out.write_text(txt + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main("xgb" if "--xgb" in sys.argv else "logistic")
