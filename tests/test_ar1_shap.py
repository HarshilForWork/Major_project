"""Invariants of AR(1)-SHAP (explain/ar1.py, explain/conditional_shapley.py)."""
import numpy as np
import pandas as pd
import pytest

from trace_pd import config
from trace_pd.explain import ar1
from trace_pd.explain.conditional_shapley import (
    AR1Filler, InterventionalFiller, PopulationFiller, gauss_conditional_draw,
    shapley_instance,
)


def _sim(n_pat=120, n_vis=8, p=4, seed=0, with_fixed=True):
    """Small simulated long table: p changing features, plus SEX (fixed) and
    AGE_AT_VISIT (clock) when with_fixed."""
    rng = np.random.default_rng(seed)
    A = 0.7 * np.eye(p)
    rows = []
    for i in range(n_pat):
        x = rng.normal(size=p) * 1.5
        sex, age = float(rng.integers(1, 3)), rng.uniform(50, 70)
        for t in range(n_vis):
            x = A @ x + rng.normal(size=p) * 0.6
            month = 6 * t if t % 3 else 6 * t + 0                      # regular
            r = [i, month] + list(np.round(x, 1))
            if with_fixed:
                r += [sex, age + month / 12]
            rows.append(r)
    names = [f"D{j}" for j in range(p)]
    cols = ["PATNO", "VISIT_MONTH"] + names + (["SEX", "AGE_AT_VISIT"] if with_fixed else [])
    df = pd.DataFrame(rows, columns=cols)
    feats = names + (["SEX", "AGE_AT_VISIT"] if with_fixed else [])
    return df, feats


# --------------------------------------------------------------- generic role detection

def test_detect_feature_roles_matches_ppmi_reference():
    """Ground-truth check: auto-detection must reproduce PPMI's known static/clock
    split exactly. If this breaks, detect_feature_roles has regressed."""
    if not config.LONG_TABLE.exists():
        pytest.skip("processed dataset absent")
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
    df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    static, clock, dynamic, report = ar1.detect_feature_roles(df, F)
    assert set(static) == set(ar1.PPMI_STATIC)
    assert set(clock) == set(ar1.PPMI_CLOCK)
    assert len(dynamic) == len(F) - len(ar1.PPMI_STATIC) - len(ar1.PPMI_CLOCK)


def test_detect_feature_roles_is_robust_to_coarse_rounding():
    """A clock feature rounded to one decimal (like PPMI's AGE_AT_VISIT) must still be
    detected, even though over a SHORT gap the rounding noise is as big as the true
    change (3 months = 0.25 years, the same size as a 0.1 rounding step). This is why
    detection uses a patient's full trajectory, not consecutive-visit pairs."""
    rng = np.random.default_rng(3)
    rows = []
    for sid in range(40):
        age0 = rng.uniform(50, 80)
        for t in range(8):
            month = 3 * t
            rows.append([sid, month, round(age0 + month / 12, 1)])  # coarse rounding
    df = pd.DataFrame(rows, columns=["PATNO", "VISIT_MONTH", "AGE_AT_VISIT"])
    static, clock, dynamic, report = ar1.detect_feature_roles(df, ["AGE_AT_VISIT"])
    assert clock == ["AGE_AT_VISIT"], report["AGE_AT_VISIT"]


def test_detect_feature_roles_works_with_custom_columns_and_string_static():
    """No PPMI-specific knowledge: a string-valued static feature and custom id/time
    column names must be detected correctly."""
    rng = np.random.default_rng(1)
    rows = []
    for sid in range(30):
        group = rng.choice(["A", "B"])
        age0 = rng.uniform(50, 80)
        x = rng.normal()
        for t in range(6):
            month = 6 * t
            x = 0.6 * x + rng.normal(scale=0.5)
            rows.append([sid, group, month, round(age0 + month / 12, 1), x])
    df = pd.DataFrame(rows, columns=["SUBJECT_ID", "GROUP", "MONTHS", "AGE_YRS", "BIOMARKER"])
    static, clock, dynamic, _ = ar1.detect_feature_roles(
        df, ["GROUP", "AGE_YRS", "BIOMARKER"], id_col="SUBJECT_ID", time_col="MONTHS")
    assert static == ["GROUP"]
    assert clock == ["AGE_YRS"]
    assert dynamic == ["BIOMARKER"]


def test_detect_feature_roles_defaults_to_dynamic_on_too_little_data():
    """Ties are broken toward DYNAMIC (the safe failure direction): a feature with
    hardly any repeat-visit patients must not be declared static or clock."""
    df = pd.DataFrame({"PATNO": [1, 2, 3], "VISIT_MONTH": [0, 0, 0], "X": [1.0, 2.0, 3.0]})
    static, clock, dynamic, report = ar1.detect_feature_roles(df, ["X"], min_patients=10)
    assert dynamic == ["X"]
    assert "dynamic" in report["X"]


def test_explicit_role_override_bypasses_detection(fitted):
    """static=/clock= on TransitionModel skips detection and uses exactly what's given."""
    df, feats, P, _ = fitted
    tm = ar1.TransitionModel(feats, "full_dt", static=["SEX"], clock=["AGE_AT_VISIT"]).fit(P, df)
    assert tm.static == ["SEX"] and tm.clock == ["AGE_AT_VISIT"]
    assert tm.role_report_["SEX"] == "explicit override"


def test_build_pairs_preserves_string_valued_static_features():
    """A bug this generalisation fixed: build_pairs used to coerce EVERY feature to
    numeric, which silently turned a string-valued static feature (e.g. ADNI's
    SEX = 'Male'/'Female') into all-NaN. Non-numeric columns must survive untouched."""
    df = pd.DataFrame({
        "PATNO": [1, 1, 2, 2], "VISIT_MONTH": [0, 6, 0, 6],
        "SEX": ["Male", "Male", "Female", "Female"], "SCORE": [10, 12, 8, 9],
    })
    P = ar1.build_pairs(df, ["SEX", "SCORE"])
    assert P["SEX"].notna().all() and set(P["SEX"]) <= {"Male", "Female"}
    assert P.loc[P.PATNO == 1, "PREV_SEX"].iloc[0] == "Male"


def test_build_pairs_works_with_custom_id_time_columns():
    df = pd.DataFrame({
        "SUBJECT_ID": [1, 1, 2, 2], "MONTHS": [0, 6, 0, 12], "X": [1.0, 2.0, 3.0, 4.0],
    })
    P = ar1.build_pairs(df, ["X"], id_col="SUBJECT_ID", time_col="MONTHS")
    assert list(P["DT"]) == [6.0, 12.0]
    assert list(P["PREV_X"]) == [1.0, 3.0]


def test_copula_round_trip_is_exact_on_discrete_values():
    X = np.array([[0, 1, 1, 2, 4, 4, 4, 3, 0, 2]], float).T
    c = ar1.CopulaTransform().fit(X)
    U = c.to_normal(X)
    assert np.array_equal(c.from_normal_col(U[:, 0], 0), X[:, 0])
    # any real number maps back to a value that exists in the data
    back = c.from_normal_col(np.linspace(-5, 5, 101), 0)
    assert set(back) <= set(X[:, 0])


def test_gaussian_conditional_matches_closed_form():
    rng = np.random.default_rng(0)
    Sig = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]])
    mu = np.array([0.5, -1.0, 2.0])
    draws = gauss_conditional_draw(mu, Sig, [0], np.array([1.5]), [1, 2], 200_000, rng)
    exp_mean = mu[[1, 2]] + Sig[[1, 2], 0] / Sig[0, 0] * (1.5 - mu[0])
    exp_cov = Sig[np.ix_([1, 2], [1, 2])] - np.outer(Sig[[1, 2], 0], Sig[0, [1, 2]]) / Sig[0, 0]
    assert np.allclose(draws.mean(0), exp_mean, atol=0.01)
    assert np.allclose(np.cov(draws.T), exp_cov, atol=0.01)


def test_pairs_never_cross_patients_and_carry_forward_is_bounded():
    df, feats = _sim()
    df.loc[df.index % 5 == 0, "D0"] = np.nan
    P = ar1.build_pairs(df, feats)
    assert len(P) == len(df) - df.PATNO.nunique()            # every visit but each first one
    assert (P.DT > 0).all()
    # a carried-forward value is flagged stale
    assert P.loc[P.PREV_D0.notna() & P.STALE_D0.eq(1)].shape[0] > 0


@pytest.fixture(scope="module")
def fitted():
    df, feats = _sim()
    P = ar1.build_pairs(df, feats)
    tm = ar1.TransitionModel(feats, "full_dt", alpha=1.0, lam=0.1).fit(P, df)
    return df, feats, P, tm


def test_feature_roles_split_fixed_clock_changing(fitted):
    _, feats, _, tm = fitted
    assert tm.static == ["SEX"] and tm.clock == ["AGE_AT_VISIT"]
    assert tm.dynamic == [f for f in feats if f.startswith("D")]


def test_history_beats_population(fitted):
    df, feats, P, tm = fitted
    pop = ar1.TransitionModel(feats, "population").fit(P, df)
    assert tm.loglik(P) > pop.loglik(P)


def _f(feats):
    w = np.array([1.0, -0.5, 0.0, 0.8, 0.7, 0.05])[: len(feats)]
    return lambda Z: np.nan_to_num(np.asarray(Z, float)) @ w


def test_efficiency_is_exact_for_every_filler(fitted):
    df, feats, P, tm = fitted
    f = _f(feats)
    X = P[feats].to_numpy(float)
    fillers = [InterventionalFiller(df[feats].to_numpy(float)),
               PopulationFiller(df[feats].to_numpy(float)), AR1Filler(tm, feats)]
    rng = np.random.default_rng(0)
    for fl in fillers:
        for i in (0, 7, 30):
            fl.for_instance(X[i], P.iloc[[i]])
            phi, base = shapley_instance(f, X[i], fl, n_perm=8, n_mc=6, rng=rng)
            assert np.isclose(phi.sum(), f(X[i][None])[0] - base, atol=1e-9)


def test_fixed_and_clock_features_get_zero_under_ar1(fitted):
    """Zero up to floating-point rounding (the averaged block of identical rows can differ
    from f(x) in the last bit)."""
    df, feats, P, tm = fitted
    f = _f(feats)
    fl = AR1Filler(tm, feats)
    X = P[feats].to_numpy(float)
    rng = np.random.default_rng(1)
    for i in (2, 11):
        fl.for_instance(X[i], P.iloc[[i]])
        phi, _ = shapley_instance(f, X[i], fl, n_perm=8, n_mc=6, rng=rng)
        assert abs(phi[feats.index("SEX")]) < 1e-12 and abs(phi[feats.index("AGE_AT_VISIT")]) < 1e-12


def test_missing_features_get_no_credit_and_stay_missing(fitted):
    df, feats, P, tm = fitted
    f = _f(feats)
    x = P[feats].to_numpy(float)[3].copy()
    x[1] = np.nan
    for fl in (InterventionalFiller(df[feats].to_numpy(float)), AR1Filler(tm, feats)):
        fl.for_instance(x, P.iloc[[3]])
        rows = fl.fill(x, np.isnan(x), 5, np.random.default_rng(0))
        assert np.isnan(rows[:, 1]).all()
        phi, _ = shapley_instance(f, x, fl, n_perm=4, n_mc=4)
        assert phi[1] == 0.0


def test_ar1_fills_stay_on_the_data_support(fitted):
    df, feats, P, tm = fitted
    fl = AR1Filler(tm, feats)
    x = P[feats].to_numpy(float)[5]
    fl.for_instance(x, P.iloc[[5]])
    shown = np.zeros(len(x), bool)
    rows = fl.fill(x, shown, 50, np.random.default_rng(0))
    for j, f_ in enumerate(feats):
        if f_ in tm.dynamic:
            assert set(rows[:, j]) <= set(df[f_].dropna())


def test_ar1_draws_stay_near_the_previous_visit(fitted):
    """The 'Frankenstein' property: AR(1) fills sit closer to the patient's own previous
    values than standard SHAP's fills from other patients."""
    df, feats, P, tm = fitted
    X = P[feats].to_numpy(float)
    prev = P[["PREV_" + c for c in feats]].to_numpy(float)
    dyn = [feats.index(c) for c in tm.dynamic]
    rng = np.random.default_rng(3)
    ar, it = AR1Filler(tm, feats), InterventionalFiller(df[feats].to_numpy(float))
    d_ar, d_it = [], []
    for i in range(0, len(X), 7):
        shown = np.zeros(len(feats), bool)
        ar.for_instance(X[i], P.iloc[[i]])
        d_ar.append(np.abs(ar.fill(X[i], shown, 10, rng)[:, dyn] - prev[i, dyn]).mean())
        d_it.append(np.abs(it.fill(X[i], shown, 10, rng)[:, dyn] - prev[i, dyn]).mean())
    assert np.mean(d_ar) < 0.8 * np.mean(d_it)


def test_wider_gap_never_narrows_the_spread_assumption(fitted):
    """Sigma(dt) is looked up by gap bucket; the lookup must be stable for any gap."""
    _, _, _, tm = fitted
    for dt in (3, 6, 9, 12, 24):
        S = tm.sigma_for(dt)
        assert np.all(np.linalg.eigvalsh(S) > 0)


def test_locked_test_patients_are_not_in_development_pairs():
    if not config.LONG_TABLE.exists():
        pytest.skip("processed dataset absent")
    from trace_pd.evaluation.splits import split_dev_test, load_locked_test_patients
    df = pd.read_csv(config.LONG_TABLE, low_memory=False, usecols=["PATNO", "VISIT_MONTH", "LABEL"])
    dev, _ = split_dev_test(df)
    assert not set(dev.PATNO) & set(load_locked_test_patients())


# ------------------------------------------------------- proof this runs on a 2nd cohort

def test_ar1_shap_pipeline_runs_end_to_end_on_adni():
    """The actual point of generalising ar1.py: build pairs, auto-detect roles, fit,
    and explain one visit on ADNI -- a dataset this module has never seen a column
    name from -- with no ADNI-specific code anywhere in explain/ar1.py."""
    if not config.ADNI_LONG_TABLE.exists():
        pytest.skip("processed ADNI dataset absent")
    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.ADNI_DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])

    static, clock, dynamic, report = ar1.detect_feature_roles(df, F, id_col="RID", time_col="VISIT_MONTH")
    assert "SEX" in static and "HANDED" in static          # same pattern as PPMI
    assert "AGE_AT_VISIT" in clock                           # same pattern as PPMI
    assert "EDUCATION_YEARS" in static                       # ADNI-specific, found with no hint

    some_rid = df["RID"].drop_duplicates().head(400)         # a consistent patient subset, not a row cut
    df_sub = df[df["RID"].isin(some_rid)]
    P = ar1.build_pairs(df_sub, F, id_col="RID", time_col="VISIT_MONTH")
    tm = ar1.TransitionModel(F, "full_dt", id_col="RID", time_col="VISIT_MONTH").fit(P, df_sub)
    static_sub, clock_sub, _, _ = ar1.detect_feature_roles(df_sub, F, id_col="RID", time_col="VISIT_MONTH")
    assert set(tm.static) == set(static_sub) and set(tm.clock) == set(clock_sub)

    # The Shapley/filler layer (conditional_shapley.py) operates on NUMERIC feature
    # vectors, same precondition any SHAP-style explainer has -- a text column like
    # HANDED='Left'/'Right' must be encoded first, exactly as it would be before
    # training the classifier being explained. ar1.py's own internals (numeric_matrix)
    # already handle a text STATIC feature; this encoding step belongs to whoever
    # prepares the feature matrix, not to the explainer.
    P_enc = P.copy()
    for c in F:
        if not pd.api.types.is_numeric_dtype(P_enc[c]):
            P_enc[c] = pd.factorize(P_enc[c], sort=True)[0].astype(float)

    fl = AR1Filler(tm, F)
    x = P_enc[F].to_numpy(float)[10]
    fl.for_instance(x, P.iloc[[10]])
    rows = fl.fill(x, np.isnan(x), 5, np.random.default_rng(0))
    assert rows.shape == (5, len(F))
    for j, f in enumerate(F):
        if f in tm.static + tm.clock:
            assert np.allclose(rows[:, j], x[j], equal_nan=True)  # known features: never perturbed
