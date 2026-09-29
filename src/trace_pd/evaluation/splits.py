"""Locked patient-level test split and the experiment log.

Two rules this module exists to enforce:

  R2  A fixed 20% of PATNOs is held out ONCE (stratified on the patient's baseline
      label, seed 2026) and written to data/interim/locked_test_patients.csv. Model
      selection never sees these patients; each step evaluates its single chosen
      model on them exactly once.
  R5  Every run -- including the ones that lose -- is appended to
      reports/metrics/experiment_log.csv.

Also holds the 200x patient-level bootstrap used for locked-test confidence
intervals (R6): patients, not rows, are resampled, because ~17 visits of one
patient are not independent observations.
"""
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trace_pd import config

LOCKED_TEST_FILE = config.INTERIM / "locked_test_patients.csv"
EXPERIMENT_LOG = config.METRICS / "experiment_log.csv"

TEST_FRACTION = 0.20
TEST_SEED = 2026
CV_SEED = 42
N_FOLDS = 5
N_BOOT = 200


def _patient_strata(df):
    """One stratum per patient: the label at that patient's earliest labelled visit.

    Stratifying on a per-visit label would be ill-defined (a patient has ~17 of them);
    the baseline label is the patient's single stable descriptor.
    """
    lab = df[df["LABEL"].notna()].sort_values(["PATNO", "VISIT_MONTH"])
    first = lab.groupby("PATNO")["LABEL"].first()
    all_pats = pd.Index(sorted(df["PATNO"].unique()), name="PATNO")
    return first.reindex(all_pats).fillna("UNLABELLED")


def make_locked_test_split(force=False):
    """Create the locked split if absent. Returns the sorted array of test PATNOs."""
    if LOCKED_TEST_FILE.exists() and not force:
        return load_locked_test_patients()

    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    strata = _patient_strata(df)
    patients = strata.index.to_numpy()

    _, test_pats = train_test_split(
        patients,
        test_size=TEST_FRACTION,
        random_state=TEST_SEED,
        stratify=strata.to_numpy(),
    )
    test_pats = np.sort(test_pats)

    out = pd.DataFrame({
        "PATNO": test_pats,
        "BASELINE_LABEL": strata.loc[test_pats].to_numpy(),
    })
    out.to_csv(LOCKED_TEST_FILE, index=False)
    return test_pats


def load_locked_test_patients():
    if not LOCKED_TEST_FILE.exists():
        return make_locked_test_split()
    return np.sort(pd.read_csv(LOCKED_TEST_FILE)["PATNO"].to_numpy())


def split_dev_test(df, patno_col="PATNO"):
    """Split any per-visit or per-patient frame into (development 80%, locked test 20%)."""
    test_pats = set(load_locked_test_patients().tolist())
    is_test = df[patno_col].isin(test_pats)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


# ------------------------------------------------------------------ shared folds

def assign_patient_folds(df, n_splits=N_FOLDS, seed=CV_SEED, label_col="LABEL"):
    """Map every PATNO -> fold id, once, for the whole project.

    Returned as a dict so that two different row subsets (e.g. the labelled visits used
    to fit the upstream subtype model, and the visit pairs used to fit the transition
    model) are guaranteed to share the SAME patient partition. Calling
    StratifiedGroupKFold separately on each subset does not guarantee that: the folds it
    produces depend on the rows handed to it, so a patient can land in the training half
    of one and the test half of the other -- which is exactly how in-sample upstream
    features leak into transition CV.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    strata = _patient_strata(df if label_col == "LABEL" else df)
    pats = strata.index.to_numpy()
    y = strata.to_numpy()

    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, te) in enumerate(sgk.split(pats.reshape(-1, 1), y, groups=pats)):
        for p in pats[te]:
            fold_of[p] = k
    return fold_of


# --------------------------------------------------------------------------- log

LOG_FIELDS = [
    "timestamp", "step", "experiment", "model", "params", "split",
    "metric", "value", "sd", "ci_low", "ci_high", "n", "n_patients", "note",
]


def log_run(step, experiment, model, params, split, metric, value,
            sd=None, ci=None, n=None, n_patients=None, note=""):
    """Append one row to reports/metrics/experiment_log.csv (R5)."""
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not EXPERIMENT_LOG.exists()
    ci_low, ci_high = (ci if ci is not None else (None, None))

    def _f(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)

    with open(EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step": step, "experiment": experiment, "model": model,
            "params": str(params), "split": split, "metric": metric,
            "value": _f(float(value)), "sd": _f(sd),
            "ci_low": _f(ci_low), "ci_high": _f(ci_high),
            "n": _f(n), "n_patients": _f(n_patients), "note": note,
        })


# ---------------------------------------------------------------- bootstrap (R6)

def patient_bootstrap_ci(groups, statistic, n_boot=N_BOOT, seed=2026, alpha=0.05):
    """200x patient-level bootstrap percentile CI.

    `groups` is the PATNO of every row. Patients are resampled with replacement and
    ALL of a sampled patient's rows come along, so within-patient correlation is kept.
    `statistic(row_index_array) -> float`; draws that are degenerate (e.g. one class
    only, so AUROC is undefined) are skipped and reported via the returned count.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_pat = {p: np.flatnonzero(groups == p) for p in uniq}
    rng = np.random.RandomState(seed)

    vals = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_pat[p] for p in drawn])
        try:
            v = statistic(rows)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(float(v))

    if len(vals) < 20:
        return (float("nan"), float("nan"), len(vals))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi), len(vals))


if __name__ == "__main__":
    pats = make_locked_test_split()
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dev, test = split_dev_test(df)
    print(f"locked test patients : {len(pats)} -> {LOCKED_TEST_FILE}")
    print(f"development rows     : {len(dev)} ({dev.PATNO.nunique()} patients)")
    print(f"locked test rows     : {len(test)} ({test.PATNO.nunique()} patients)")
    print(pd.read_csv(LOCKED_TEST_FILE)["BASELINE_LABEL"].value_counts().to_string())
