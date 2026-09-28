"""Contract tests on the processed dataset.

These are the invariants the rest of the project depends on. If any of them
break, the preprocessing stage has regressed and every downstream number
(baselines, ablations, the paper) is suspect.
"""
import pandas as pd
import pytest

from trace_pd import config

LABEL_DEFINING = [
    "NP2TRMR",
    "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
    "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON",
    "NP2WALK", "NP2FREZ",
    "NP3GAIT", "NP3FRZGT", "NP3PSTBL",
]


@pytest.fixture(scope="module")
def df():
    if not config.LONG_TABLE.exists():
        pytest.skip("processed dataset absent - run `make preprocess` first")
    return pd.read_csv(config.LONG_TABLE, low_memory=False)


@pytest.fixture(scope="module")
def dictionary():
    if not config.DICTIONARY.exists():
        pytest.skip("dictionary absent - run `make preprocess` first")
    return pd.read_csv(config.DICTIONARY)


def test_shape(df):
    assert df.shape == (6922, 75)


def test_patient_count(df):
    assert df["PATNO"].nunique() == 439


def test_no_duplicate_visits(df):
    assert not df.duplicated(subset=["PATNO", "EVENT_ID"]).any()


def test_label_values(df):
    assert set(df["LABEL"].dropna().unique()) <= {"TD", "PIGD", "INDETERMINATE"}


def test_labelled_row_count(df):
    assert df["LABEL"].notna().sum() == 5742


def test_transition_pair_count(df):
    assert df["LABEL_FLIPPED_NEXT"].notna().sum() == 4918


def test_no_label_defining_item_is_a_feature(dictionary):
    """The strict leakage policy: nothing used to compute Y may train the model."""
    features = set(dictionary.loc[dictionary["bucket"].isin(["CHEAP_FEATURE", "RESOURCE_DEPENDENT_FEATURE"]), "column"])
    assert features.isdisjoint(LABEL_DEFINING)


def test_label_is_reproducible_from_component_scores(df):
    """Recompute the Stebbins classification and check it matches, as the
    preprocessing stage claims (0 mismatches across all labelled rows)."""
    sub = df[df["LABEL"].notna() & df["TD_PIGD_RATIO"].notna()]
    recomputed = pd.Series("INDETERMINATE", index=sub.index)
    recomputed[sub["TD_PIGD_RATIO"] >= 1.15] = "TD"
    recomputed[sub["TD_PIGD_RATIO"] <= 0.90] = "PIGD"
    assert (recomputed == sub["LABEL"]).all()
