"""Locked patient-level test split for ADNI -- the ADNI analogue of splits.py.

Same methodology and same constants as PPMI's splits.py (20% held out once, stratified
on baseline label, patient-grouped CV folds), with ADNI's column names (RID, not PATNO;
VISIT_MONTH is shared). `log_run` and `patient_bootstrap_ci` are dataset-agnostic and
are reused directly from splits.py rather than duplicated.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

from trace_pd import config
from trace_pd.evaluation.splits import TEST_FRACTION, TEST_SEED, CV_SEED, N_FOLDS  # noqa: F401 (re-exported)

LOCKED_TEST_FILE = config.INTERIM / "locked_test_patients_adni.csv"


def _patient_strata(df):
    """One stratum per patient: the label at that patient's earliest labelled visit."""
    lab = df[df["LABEL"].notna()].sort_values(["RID", "VISIT_MONTH"])
    first = lab.groupby("RID")["LABEL"].first()
    all_pats = pd.Index(sorted(df["RID"].unique()), name="RID")
    return first.reindex(all_pats).fillna("UNLABELLED")


def make_locked_test_split(force=False):
    if LOCKED_TEST_FILE.exists() and not force:
        return load_locked_test_patients()

    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    strata = _patient_strata(df)
    patients = strata.index.to_numpy()

    _, test_pats = train_test_split(
        patients, test_size=TEST_FRACTION, random_state=TEST_SEED, stratify=strata.to_numpy(),
    )
    test_pats = np.sort(test_pats)
    out = pd.DataFrame({"RID": test_pats, "BASELINE_LABEL": strata.loc[test_pats].to_numpy()})
    out.to_csv(LOCKED_TEST_FILE, index=False)
    return test_pats


def load_locked_test_patients():
    if not LOCKED_TEST_FILE.exists():
        return make_locked_test_split()
    return np.sort(pd.read_csv(LOCKED_TEST_FILE)["RID"].to_numpy())


def split_dev_test(df, id_col="RID"):
    test_pats = set(load_locked_test_patients().tolist())
    is_test = df[id_col].isin(test_pats)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def assign_patient_folds(df, n_splits=N_FOLDS, seed=CV_SEED):
    strata = _patient_strata(df)
    pats = strata.index.to_numpy()
    y = strata.to_numpy()
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, te) in enumerate(sgk.split(pats.reshape(-1, 1), y, groups=pats)):
        for p in pats[te]:
            fold_of[p] = k
    return fold_of


if __name__ == "__main__":
    pats = make_locked_test_split()
    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    dev, test = split_dev_test(df)
    print(f"locked test patients : {len(pats)} -> {LOCKED_TEST_FILE}")
    print(f"development rows     : {len(dev)} ({dev.RID.nunique()} patients)")
    print(f"locked test rows     : {len(test)} ({test.RID.nunique()} patients)")
    print(pd.read_csv(LOCKED_TEST_FILE)["BASELINE_LABEL"].value_counts().to_string())
