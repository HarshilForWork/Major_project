"""Contract tests on the processed ADNI dataset -- the ADNI analogue of
test_dataset_contract.py. If any of these break, preprocess_adni.py has
regressed and the second-cohort ("framework") claim is unsupported.
"""
import pandas as pd
import pytest

from trace_pd import config

CDR_COLS = ["CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE", "CDGLOBAL", "CDRSB"]


@pytest.fixture(scope="module")
def df():
    if not config.ADNI_LONG_TABLE.exists():
        pytest.skip("ADNI processed dataset absent - run `make preprocess-adni` first")
    return pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)


@pytest.fixture(scope="module")
def dictionary():
    if not config.ADNI_DICTIONARY.exists():
        pytest.skip("ADNI dictionary absent - run `make preprocess-adni` first")
    return pd.read_csv(config.ADNI_DICTIONARY)


def test_shape(df):
    assert df.shape == (24302, 43)


def test_patient_count(df):
    assert df["RID"].nunique() == 4470


def test_no_duplicate_visits(df):
    assert not df.duplicated(subset=["RID", "VISCODE2"]).any()


def test_label_values(df):
    assert set(df["LABEL"].dropna().unique()) <= {"CN", "MCI", "Dementia"}


def test_labelled_row_count(df):
    assert df["LABEL"].notna().sum() == 14336


def test_no_cdr_item_is_a_feature(dictionary):
    """Strict leakage policy, same as PPMI: nothing the diagnosis is informed by
    or gated on may train the model."""
    features = set(dictionary.loc[dictionary["bucket"].isin(
        ["CHEAP_FEATURE", "RESOURCE_DEPENDENT_FEATURE"]), "column"])
    assert features.isdisjoint(CDR_COLS)


def test_apoe_genotype_is_held_out_not_a_cheap_feature(dictionary):
    cheap = set(dictionary.loc[dictionary["bucket"] == "CHEAP_FEATURE", "column"])
    assert "APOE_GENOTYPE" not in cheap
    assert (dictionary.loc[dictionary.column == "APOE_GENOTYPE", "bucket"] == "RESOURCE_DEPENDENT_FEATURE").all()


def test_cdr_global_is_strongly_associated_with_label(df):
    """Empirical justification for banning CDR (preprocess_adni.py decision 2):
    CDGLOBAL=0 should mostly be CN, and CDGLOBAL>=1 should mostly be Dementia.
    A regression here means the extract changed and the ban needs re-checking,
    not that the test is wrong."""
    sub = df.dropna(subset=["LABEL", "CDGLOBAL"])
    assert (sub.loc[sub.CDGLOBAL == 0, "LABEL"] == "CN").mean() > 0.85
    assert (sub.loc[sub.CDGLOBAL >= 1, "LABEL"] == "Dementia").mean() > 0.85


def test_no_sentinel_codes_in_vitals(df):
    """ADNI's -1 ('not recorded') sentinel in VSWEIGHT/VSHEIGHT must never reach
    the converted WEIGHT_KG/HEIGHT_CM columns -- the ADNI analogue of PPMI's
    101/9 missing-code bug."""
    assert df["WEIGHT_KG"].min() >= 30
    assert df["HEIGHT_CM"].min() >= 130


def test_age_at_visit_is_plausible(df):
    ages = df["AGE_AT_VISIT"].dropna()
    assert ages.min() >= 40 and ages.max() <= 110


def test_transition_pair_count(df):
    assert df["LABEL_FLIPPED_NEXT"].notna().sum() == 5615


def test_visit_month_parses_the_full_cohort_timeline(df):
    """VISCODE2 was used as the timeline (preprocess_adni.py decision 5), not
    EXAMDATE, specifically because EXAMDATE has real gaps. VISIT_MONTH itself
    must never be missing or negative."""
    assert df["VISIT_MONTH"].notna().all()
    assert (df["VISIT_MONTH"] >= 0).all()
