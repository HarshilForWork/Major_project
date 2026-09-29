"""Invariants of the FCX explanation framework."""
import numpy as np
import pandas as pd
import pytest
from trace_pd import config
from trace_pd.explain import formula as F
from trace_pd.explain.shapley import shapley_mc, shapley_exact
from trace_pd.explain.pdn import PDNExplainer, flip_prob
from trace_pd.explain.straddle import CutoffStraddleExplainer


def test_formula_zones_reproduce_stored_labels():
    if not config.LONG_TABLE.exists():
        pytest.skip("processed dataset absent")
    d = pd.read_csv(config.LONG_TABLE, low_memory=False)
    d = d[d.LABEL.notna()]
    lT, lP = F.log_scores(d.TREMOR_SCORE, d.PIGD_SCORE)
    assert (F.ZONE_NAMES[F.zone(lT - lP)] == d.LABEL.to_numpy()).all()


def test_shapley_mc_efficiency_is_exact():
    rng = np.random.default_rng(0)
    f = lambda Z: Z[:, 0] * Z[:, 1] + np.sin(Z[:, 2]) + Z[:, 3] ** 2
    X, B = rng.normal(size=(8, 4)), rng.normal(size=(50, 4))
    phi, base = shapley_mc(f, X, B, n_perm=20)
    assert np.allclose(phi.sum(1), f(X) - base)


def test_shapley_mc_converges_to_exact_for_additive_model():
    rng = np.random.default_rng(1)
    w = np.array([1.0, -2.0, 0.5])
    f = lambda Z: Z @ w
    X, B = rng.normal(size=(5, 3)), rng.normal(size=(200, 3))
    phi, _ = shapley_mc(f, X, B, n_perm=64)
    exact = (X - B.mean(0)) * w
    assert np.abs(phi - exact).max() < 0.35          # single background draw per permutation -> MC noise


def test_shapley_exact_efficiency():
    v = lambda S: float(len(S) ** 2)
    out = shapley_exact(v, ["a", "b", "c"])
    assert np.isclose(sum(out.values()), v({"a", "b", "c"}) - v(set()))


def test_flip_prob_behaviour():
    near = flip_prob([F.CUT_TD], [0.01], [0.0], [0.3])
    far = flip_prob([3.0], [0.01], [0.0], [0.3])
    noisy = flip_prob([1.0], [0.01], [0.0], [1.0])
    quiet = flip_prob([1.0], [0.01], [0.0], [0.1])
    assert near > 0.4 and far < 1e-3 and noisy > quiet


def test_pdn_shapley_efficiency():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5))
    eta = X[:, 0] * 0.5; mu = X[:, 1] * 0.1; noise = np.abs(rng.normal(size=300)) * 0.3
    p = PDNExplainer().fit(X, eta, mu, noise)
    p.calibrate_fidelity(X, np.clip(0.3 + 0.1 * X[:, 0], 0.01, 0.99))
    e = p.explain(X[:20])
    assert np.allclose(e["phi_P"] + e["phi_D"] + e["phi_N"], e["full"] - e["base"])
    assert np.allclose(e["full"], p.predict_logit(X[:20]))


def test_pdn_component_weights_follow_the_black_box():
    """A black box driven only by the proximity component should load on P, not D or N."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 5))
    p = PDNExplainer().fit(X, X[:, 0] * 0.5, X[:, 1] * 0.2, np.abs(X[:, 2]) * 0.3 + 0.05)
    H = p.heads(X); p._set_reference(H); s, _ = p.surrogate_components(H=H)
    g = 1 / (1 + np.exp(-(-1.0 + 1.5 * s["P"])))            # planted: proximity only
    p.calibrate_fidelity(X, g)
    assert abs(p.b["P"]) > 5 * max(abs(p.b["D"]), abs(p.b["N"]))


def test_straddle_logic():
    st = CutoffStraddleExplainer(); st.kappa = 1.0
    st.half_widths = lambda X: {"T": np.array([0.5, 0.05, 0.3]), "P": np.array([0.05, 0.5, 0.3])}
    l = np.array([0.2, 0.2, 0.1]); cut = 0.0
    o = st.explain(np.zeros((3, 1)), l, cut, point_T=l, point_P=np.zeros(3))
    assert list(o["blame"]) == ["tremor", "gait", "both"]


def test_shapley_groups_must_be_disjoint():
    f = lambda Z: Z.sum(1)
    X, B = np.ones((2, 3)), np.zeros((5, 3))
    with pytest.raises(ValueError):
        shapley_mc(f, X, B, n_perm=2, groups=[[0, 1], [1, 2]])


def test_implied_exam_channels_are_exactly_additive():
    from trace_pd.explain.implied_exam import ImpliedExamExplainer
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 4)); lT = X[:, 0] - 1; lP = 0.5 * X[:, 1] - 1
    iee = ImpliedExamExplainer(["a", "b", "c", "d"]).fit(X, lT, lP)
    e = iee.explain(X[:10], X[100:160], n_perm=8)
    assert np.allclose(e["phi_ratio"], e["phi_tremor"] - e["phi_gait"])
    assert np.allclose(e["phi_ratio"].sum(1), e["log_ratio"] - (e["base_tremor"] - e["base_gait"]))


def test_pdn_weights_are_non_negative():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 5))
    p = PDNExplainer().fit(X, X[:, 0] * 0.5, X[:, 1] * 0.2, np.abs(X[:, 2]) * 0.3 + 0.05)
    p.calibrate_fidelity(X, 1 / (1 + np.exp(-(0.2 - 0.8 * X[:, 1]))))     # arbitrary box
    assert min(p.b.values()) >= 0


def test_trajectory_never_spans_an_unlabelled_visit():
    from trace_pd.explain import trajectory as TJ
    df = pd.DataFrame(dict(PATNO=[1] * 4, VISIT_MONTH=[0, 3, 6, 9],
                           LABEL=["TD", None, "TD", "PIGD"], NEXT_LABEL=[None, "TD", "PIGD", None],
                           LABEL_FLIPPED_NEXT=[np.nan, np.nan, 1.0, np.nan],
                           TREMOR_SCORE=[1.0, np.nan, 1.0, 0.2], PIGD_SCORE=[0.2, np.nan, 0.4, 1.0],
                           PDSTATE_USED=["OFF"] * 4, **{c: [1.0, 2.0, 3.0, 4.0] for c in TJ.DELTA_COLS}))
    d = TJ.build(df)
    row6 = d[d.VISIT_MONTH == 6].iloc[0]
    assert np.isnan(row6.eta)                    # previous scheduled visit (month 3) is unlabelled
    assert row6.D_NP2RISE == 1.0                 # delta vs the adjacent scheduled visit, not month 0
