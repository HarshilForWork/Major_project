"""ADNI subtype classifier -- the model AR(1)-SHAP explains on the second cohort.

Mirrors compare_subtype_models.py's design exactly, so the two cohorts' numbers are
genuinely comparable: candidates scored by 5-fold StratifiedGroupKFold (seed 42)
grouped by RID on the DEVELOPMENT 80% of patients, winner evaluated ONCE on the locked
20% test patients with a 200x patient-bootstrap 95% CI, every config logged to
experiment_log.csv.

Two tasks, exactly paralleling PPMI's design decision to drop the unstable
Indeterminate middle class and report binary TD-vs-PIGD as the headline result:

  3-class   CN vs MCI vs Dementia (all labelled visits)
  binary    CN vs Dementia, MCI dropped -- MCI is ADNI's unstable middle diagnosis
            (13.4% flip to something else at the next visit, vs CN 5.2% and
            Dementia 2.4% -- see HANDOFF_AR1_SHAP.md section 13.4), the same role
            Indeterminate plays in PPMI.

SEX and HANDED are text ('Male'/'Female', 'Left'/'Right') -- encoded to integer codes
once, on the full population, before fitting. Encoding a static demographic category
this way carries no label information, so it doesn't leak between folds the way
target encoding would.

Run:  PYTHONPATH=src python -m trace_pd.models.train_adni
Out:  reports/metrics/adni_baseline_results.txt
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from trace_pd import config
from trace_pd.evaluation.splits import log_run, patient_bootstrap_ci
from trace_pd.evaluation.splits_adni import split_dev_test, CV_SEED, N_FOLDS

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STEP = "adni-1"
OUT = config.METRICS / "adni_baseline_results.txt"
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def _xgb(n_classes):
    p = dict(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1,
              tree_method="hist")
    if n_classes > 2:
        p.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    else:
        p.update(objective="binary:logistic", eval_metric="logloss")
    return XGBClassifier(**p)


def _logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))


def _hgb():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)


# ============================================================================== data
def load(features_numeric_cache={}):
    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.ADNI_DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])

    # encode text categories once, on the full population -- no label information
    # involved, so this is not a leak between folds (same reasoning as PPMI's
    # SEX/HANDED already being numeric codes from the source data).
    for c in F:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c], sort=True)[0].astype(float)
            df.loc[df[c] < 0, c] = np.nan

    df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    dev, test = split_dev_test(df)
    return df, dev, test, F


def _fit_balanced(m, X, y):
    """Fit with balanced sample weights, whether m is a plain estimator or a
    Pipeline (sklearn pipelines need the weight routed to the final step by name)."""
    from sklearn.pipeline import Pipeline
    sw = compute_sample_weight("balanced", y)
    if isinstance(m, Pipeline):
        m.fit(X, y, **{f"{m.steps[-1][0]}__sample_weight": sw})
    else:
        m.fit(X, y, sample_weight=sw)
    return m


def cv_score(model_fn, X, y, groups, n_splits=N_FOLDS, seed=CV_SEED):
    oof_pred = np.zeros(len(y), int)
    oof_prob = np.zeros((len(y), len(np.unique(y))))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        m = _fit_balanced(model_fn(), X.iloc[tr], y[tr])
        oof_pred[te] = m.predict(X.iloc[te])
        oof_prob[te] = m.predict_proba(X.iloc[te])
    ba = balanced_accuracy_score(y, oof_pred)
    f1 = f1_score(y, oof_pred, average="macro")
    return ba, f1, oof_pred, oof_prob


def run_task(name, sub, F, classes, pos_label=None):
    log("\n" + "=" * 74)
    log(f"TASK: {name}  (classes: {classes})")
    log("=" * 74)
    c2i = {c: i for i, c in enumerate(classes)}
    y = sub["LABEL"].map(c2i).to_numpy()
    X = sub[F]
    groups = sub["RID"].to_numpy()
    log(f"n = {len(sub)} visits, {sub.RID.nunique()} patients, "
        f"class counts: {sub['LABEL'].value_counts().to_dict()}")

    majority_ba = balanced_accuracy_score(y, np.full(len(y), np.bincount(y).argmax()))
    log(f"\nmajority baseline balanced accuracy: {majority_ba:.4f}")
    log_run(STEP, name, "majority", {}, "dev grouped CV", "balanced_accuracy", majority_ba,
            n=len(y), n_patients=sub.RID.nunique())

    candidates = {
        "logistic": lambda: _logistic(),
        "xgboost": lambda: _xgb(len(classes)),
        "hgb": lambda: _hgb(),
    }
    results = {}
    for cname, fn in candidates.items():
        ba, f1, pred, prob = cv_score(fn, X, y, groups)
        log(f"{cname:<12} CV balanced_acc {ba:.4f}  macro_F1 {f1:.4f}")
        log_run(STEP, name, cname, {}, "dev grouped CV", "balanced_accuracy", ba,
                n=len(y), n_patients=sub.RID.nunique())
        log_run(STEP, name, cname, {}, "dev grouped CV", "macro_f1", f1,
                n=len(y), n_patients=sub.RID.nunique())
        results[cname] = (ba, fn)

    winner_name = max(results, key=lambda k: results[k][0])
    winner_fn = results[winner_name][1]
    log(f"\nwinner by CV balanced accuracy: {winner_name}")

    # ---- locked test, scored once ----
    test_sub = TEST_BY_TASK[name]
    Xte, yte = test_sub[F], test_sub["LABEL"].map(c2i).to_numpy()
    final = _fit_balanced(winner_fn(), X, y)
    pred_te = final.predict(Xte)
    prob_te = final.predict_proba(Xte)
    ba_te = balanced_accuracy_score(yte, pred_te)
    f1_te = f1_score(yte, pred_te, average="macro")
    auc_te = np.nan
    if len(classes) == 2 and pos_label is not None:
        auc_te = roc_auc_score(yte, prob_te[:, c2i[pos_label]])

    def stat(rows):
        yr, pr = yte[rows], pred_te[rows]
        return balanced_accuracy_score(yr, pr) if len(set(yr)) > 1 else None

    ci = patient_bootstrap_ci(test_sub.RID.to_numpy(), stat)
    log(f"\nLOCKED TEST ({winner_name}): balanced_acc {ba_te:.4f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  "
        f"macro_F1 {f1_te:.4f}" + (f"  AUROC {auc_te:.4f}" if not np.isnan(auc_te) else ""))
    log_run(STEP, name, winner_name, {}, "LOCKED TEST", "balanced_accuracy", ba_te,
            ci=(ci[0], ci[1]), n=len(yte), n_patients=test_sub.RID.nunique(), note=f"{ci[2]} bootstrap draws usable")
    if not np.isnan(auc_te):
        log_run(STEP, name, winner_name, {}, "LOCKED TEST", "roc_auc", auc_te,
                n=len(yte), n_patients=test_sub.RID.nunique())
    return winner_name, ba_te, ci


def main():
    df, dev, test, F = load()
    log(f"ADNI classifier bake-off -- {len(F)} cheap features")
    log(f"development: {len(dev)} visits, {dev.RID.nunique()} patients; "
        f"locked test: {len(test)} visits, {test.RID.nunique()} patients")

    global TEST_BY_TASK
    TEST_BY_TASK = {}

    dev3 = dev[dev.LABEL.isin(["CN", "MCI", "Dementia"])]
    TEST_BY_TASK["3-class CN/MCI/Dementia"] = test[test.LABEL.isin(["CN", "MCI", "Dementia"])]
    run_task("3-class CN/MCI/Dementia", dev3, F, ["CN", "MCI", "Dementia"])

    dev2 = dev[dev.LABEL.isin(["CN", "Dementia"])]
    TEST_BY_TASK["binary CN vs Dementia"] = test[test.LABEL.isin(["CN", "Dementia"])]
    run_task("binary CN vs Dementia", dev2, F, ["CN", "Dementia"], pos_label="Dementia")

    OUT.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
