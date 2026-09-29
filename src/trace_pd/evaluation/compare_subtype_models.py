"""STEP 2 -- binary TD vs PIGD per visit: pick a model honestly.

Every candidate is scored by 5-fold StratifiedGroupKFold (seed 42) grouped by PATNO on
the DEVELOPMENT 80% of patients only (R1, R3). The winner by CV balanced accuracy is
then evaluated ONCE on the locked 20% test patients, with a 200x patient-bootstrap 95%
CI (R2, R6). Every config, including the losers, is appended to experiment_log.csv (R5).

Candidates:
  logistic            L2, median-impute + standardise, balanced class weight
  xgb_default         the project's standing config
  xgb_grid_*          max_depth 2/3/4 x learning_rate 0.03/0.08, 300 trees
  lightgbm_default    if installed
  xgb_missind         xgb + per-column missingness indicators
  xgb_thresh          xgb with the decision threshold tuned on an inner grouped CV
  formula_aware       regress log T and log P from cheap features, then apply the
                      published Stebbins cutoffs to the predicted ratio

Run:  PYTHONPATH=src python -m trace_pd.evaluation.compare_subtype_models
Out:  reports/metrics/subtype_model_comparison_xgb.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from trace_pd import config
from trace_pd.explain import formula as FM
from trace_pd.evaluation.splits import (
    split_dev_test, log_run, patient_bootstrap_ci, CV_SEED, N_FOLDS,
)

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

POS = "PIGD"   # positive class for AUROC


# --------------------------------------------------------------------------- data

def load_binary():
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    feats = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())

    sub = df[df.LABEL.isin(["TD", "PIGD"])].copy()
    # true hidden exam scores, needed only by the formula-aware candidate
    sub["_T"] = sub[FM.TREMOR_ITEMS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    sub["_P"] = sub[FM.GAIT_ITEMS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return sub, feats


def xy(sub, feats):
    X = sub[feats].apply(pd.to_numeric, errors="coerce")
    y = (sub["LABEL"].to_numpy() == POS).astype(int)
    g = sub["PATNO"].to_numpy()
    return X, y, g


# --------------------------------------------------------------------- candidates

def _xgb(max_depth=3, learning_rate=0.05, n_estimators=300):
    return XGBClassifier(
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
    )


def _add_missing_indicators(X):
    ind = X.isna().astype(int)
    ind.columns = [f"{c}__isna" for c in X.columns]
    keep = [c for c in ind.columns if ind[c].nunique() > 1]
    return pd.concat([X, ind[keep]], axis=1)


def fit_predict_generic(kind, Xtr, ytr, Xte, sub_tr=None, sub_te=None, **kw):
    """Return (prob_of_POS, hard_prediction) for one train/test split."""
    from sklearn.utils.class_weight import compute_sample_weight

    if kind == "logistic":
        pipe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        return p, (p >= 0.5).astype(int)

    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight="balanced")
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        return p, (p >= 0.5).astype(int)

    if kind == "missind":
        Xtr2, Xte2 = _add_missing_indicators(Xtr), _add_missing_indicators(Xte)
        Xte2 = Xte2.reindex(columns=Xtr2.columns, fill_value=0)
        m = _xgb(**kw)
        m.fit(Xtr2, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        p = m.predict_proba(Xte2)[:, 1]
        return p, (p >= 0.5).astype(int)

    if kind == "formula_aware":
        # Two regression heads on the TRUE hidden exam scores, then the published rule.
        # The heads are trained on log(T+eps) / log(P+eps); the cutoffs are NOT fitted.
        lt = np.log(sub_tr["_T"].to_numpy(dtype=float) + FM.EPS)
        lp = np.log(sub_tr["_P"].to_numpy(dtype=float) + FM.EPS)
        ok = np.isfinite(lt) & np.isfinite(lp)
        hT = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, random_state=42,
                          n_jobs=-1, tree_method="hist").fit(Xtr[ok], lt[ok])
        hP = XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, random_state=42,
                          n_jobs=-1, tree_method="hist").fit(Xtr[ok], lp[ok])
        l_hat = hT.predict(Xte) - hP.predict(Xte)
        # map the implied log-ratio to P(PIGD) monotonically for AUROC
        p = 1.0 / (1.0 + np.exp((l_hat - FM.CUT_PIGD) * 4))
        pred = (l_hat <= (FM.CUT_PIGD + FM.CUT_TD) / 2).astype(int)
        return p, pred

    # plain xgboost variants
    m = _xgb(**kw)
    m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    p = m.predict_proba(Xte)[:, 1]
    return p, (p >= 0.5).astype(int)


def tuned_threshold(Xtr, ytr, gtr, **kw):
    """Pick the decision threshold maximising balanced accuracy on an INNER grouped CV
    of the training fold only -- never on the data it will be scored against."""
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=CV_SEED)
    oof = np.full(len(ytr), np.nan)
    for tr, te in inner.split(Xtr, ytr, groups=gtr):
        p, _ = fit_predict_generic("xgb", Xtr.iloc[tr], ytr[tr], Xtr.iloc[te], **kw)
        oof[te] = p
    grid = np.linspace(0.2, 0.8, 61)
    scores = [balanced_accuracy_score(ytr, (oof >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------------- CV loop

def cv_evaluate(name, kind, sub, feats, kw=None):
    """5-fold grouped CV -> (mean, sd) of per-fold balanced accuracy, plus mean AUROC."""
    kw = kw or {}
    X, y, g = xy(sub, feats)
    sgk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    bas, aucs = [], []
    for tr, te in sgk.split(X, y, groups=g):
        if kind == "xgb_thresh":
            thr = tuned_threshold(X.iloc[tr], y[tr], g[tr], **kw)
            p, _ = fit_predict_generic("xgb", X.iloc[tr], y[tr], X.iloc[te], **kw)
            pred = (p >= thr).astype(int)
        else:
            p, pred = fit_predict_generic(
                kind, X.iloc[tr], y[tr], X.iloc[te],
                sub_tr=sub.iloc[tr], sub_te=sub.iloc[te], **kw)
        bas.append(balanced_accuracy_score(y[te], pred))
        aucs.append(roc_auc_score(y[te], p))

    return float(np.mean(bas)), float(np.std(bas)), float(np.mean(aucs))


def main():
    sub_all, feats = load_binary()
    dev, test = split_dev_test(sub_all)

    print("=" * 78)
    print("STEP 2 -- BINARY TD vs PIGD, MODEL SELECTION ON THE DEVELOPMENT SET")
    print("=" * 78)
    print(f"development : {len(dev):,} visits / {dev.PATNO.nunique()} patients")
    print(f"locked test : {len(test):,} visits / {test.PATNO.nunique()} patients (untouched until the winner is chosen)")
    print(f"features    : {len(feats)} CHEAP_FEATURE columns")
    print()

    candidates = [("logistic", "logistic", {}), ("xgb_default", "xgb", {})]
    for md in (2, 3, 4):
        for lr in (0.03, 0.08):
            candidates.append((f"xgb_d{md}_lr{lr}", "xgb", dict(max_depth=md, learning_rate=lr, n_estimators=300)))
    try:
        import lightgbm  # noqa: F401
        candidates.append(("lightgbm_default", "lightgbm", {}))
    except ImportError:
        print("[lightgbm not installed -- candidate skipped]\n")
    candidates += [
        ("xgb_missingness_ind", "missind", {}),
        ("xgb_tuned_threshold", "xgb_thresh", {}),
        ("formula_aware", "formula_aware", {}),
    ]

    rows = []
    for name, kind, kw in candidates:
        mean, sd, auc = cv_evaluate(name, kind, dev, feats, kw)
        rows.append((name, kind, kw, mean, sd, auc))
        print(f"  {name:<22} bal.acc {mean:.4f} +/- {sd:.4f}   AUROC {auc:.4f}")
        log_run("2", "binary TD vs PIGD", name, kw, "dev 80% grouped CV",
                "balanced_accuracy", mean, sd=sd, n=len(dev), n_patients=dev.PATNO.nunique())
        log_run("2", "binary TD vs PIGD", name, kw, "dev 80% grouped CV",
                "roc_auc", auc, n=len(dev), n_patients=dev.PATNO.nunique())

    rows.sort(key=lambda r: -r[3])
    win_name, win_kind, win_kw, win_mean, win_sd, win_auc = rows[0]
    print(f"\nWINNER BY CV: {win_name}  ({win_mean:.4f} +/- {win_sd:.4f})")

    # ------------------------------------------------ locked test, used exactly once
    Xd, yd, gd = xy(dev, feats)
    Xt, yt, gt = xy(test, feats)

    if win_kind == "xgb_thresh":
        thr = tuned_threshold(Xd, yd, gd, **win_kw)
        p_te, _ = fit_predict_generic("xgb", Xd, yd, Xt, **win_kw)
        pred_te = (p_te >= thr).astype(int)
        print(f"(threshold tuned on development inner CV: {thr:.3f})")
    else:
        p_te, pred_te = fit_predict_generic(
            win_kind, Xd, yd, Xt, sub_tr=dev, sub_te=test, **win_kw)

    ba_te = balanced_accuracy_score(yt, pred_te)
    auc_te = roc_auc_score(yt, p_te)
    lo_ba, hi_ba, nb = patient_bootstrap_ci(gt, lambda r: balanced_accuracy_score(yt[r], pred_te[r]))
    lo_au, hi_au, _ = patient_bootstrap_ci(gt, lambda r: roc_auc_score(yt[r], p_te[r]))

    # majority-class reference on the same locked patients
    maj = np.full(len(yt), int(np.mean(yt) >= 0.5))
    ba_maj = balanced_accuracy_score(yt, maj)

    print(f"\nLOCKED TEST ({len(yt):,} visits / {test.PATNO.nunique()} patients), 200x patient bootstrap:")
    print(f"  balanced accuracy {ba_te:.4f}   95% CI [{lo_ba:.4f}, {hi_ba:.4f}]")
    print(f"  AUROC             {auc_te:.4f}   95% CI [{lo_au:.4f}, {hi_au:.4f}]")
    print(f"  majority baseline {ba_maj:.4f}  chance 0.5000")
    sig = "YES" if lo_ba > 0.5 else "NO -- CI includes chance"
    print(f"  better than chance? {sig}")

    log_run("2", "binary TD vs PIGD", win_name, win_kw, "LOCKED TEST 20%",
            "balanced_accuracy", ba_te, ci=(lo_ba, hi_ba), n=len(yt),
            n_patients=test.PATNO.nunique(), note=f"winner by dev CV; {nb}/200 bootstrap draws usable")
    log_run("2", "binary TD vs PIGD", win_name, win_kw, "LOCKED TEST 20%",
            "roc_auc", auc_te, ci=(lo_au, hi_au), n=len(yt), n_patients=test.PATNO.nunique())

    # ------------------------------------------------------------------- report
    L = ["=" * 78,
         "STEP 2 -- SUBTYPE CLASSIFIER (binary TD vs PIGD, per visit)",
         "=" * 78,
         f"Development : {len(dev):,} visits / {dev.PATNO.nunique()} patients (model selection ONLY)",
         f"Locked test : {len(test):,} visits / {test.PATNO.nunique()} patients (used once, for the winner)",
         "Selection   : 5-fold StratifiedGroupKFold by PATNO, seed 42",
         "",
         f"{'config':<24}{'CV bal.acc':>14}{'sd':>9}{'CV AUROC':>10}",
         "-" * 78]
    for name, _k, _kw, mean, sd, auc in rows:
        L.append(f"{name:<24}{mean:>14.4f}{sd:>9.4f}{auc:>10.4f}")
    L += ["-" * 78,
          f"Winner by CV balanced accuracy: {win_name}",
          "",
          "LOCKED TEST (evaluated once; 200x patient-level bootstrap 95% CI)",
          f"  balanced accuracy : {ba_te:.4f}  [{lo_ba:.4f}, {hi_ba:.4f}]",
          f"  AUROC             : {auc_te:.4f}  [{lo_au:.4f}, {hi_au:.4f}]",
          f"  majority baseline : {ba_maj:.4f}   (chance = 0.5000)",
          f"  above chance      : {sig}",
          "",
          "WHAT DID NOT HOLD",
          f"  - Spread across all {len(rows)} configs is {rows[0][3]-rows[-1][3]:.4f}, comparable to the",
          "    per-fold sd, so the ranking between the top configs is not meaningful.",
          "  - Configs whose CV mean sits inside the winner's +/- 1 sd band are ties, not losses:",
          "    " + ", ".join(n for n, _k, _kw, m, _s, _a in rows[1:] if m >= win_mean - win_sd) or "    (none)",
          "  - The locked-test CI is wide because it rests on only "
          f"{test.PATNO.nunique()} patients.",
          "=" * 78]
    txt = "\n".join(L)
    out = config.METRICS / "subtype_model_comparison_xgb.txt"
    out.write_text(txt + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")
    return win_name, win_kind, win_kw


if __name__ == "__main__":
    main()
