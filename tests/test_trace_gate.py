"""TRACE Stage 3 gate (docs/TRACE_BRIEF.md): exactness checks on REAL locked-test patients.

No synthetic data or toy fixtures (brief rule 1). Each model is checked once per session;
the first run fills results/cache/ (roughly 25 minutes per ADNI model on one core), later
runs reuse it. Skipped when the processed cohorts are not present.

    PYTHONPATH=src python -m pytest tests/test_trace_gate.py -v
"""
import pytest

from trace_pd import config

pytestmark = pytest.mark.skipif(
    not (config.LONG_TABLE.exists() and config.ADNI_LONG_TABLE.exists()),
    reason="processed PPMI/ADNI cohorts not present",
)

GATE_MODELS = ["ppmi", "adni_minus", "adni_minus_lm", "adni_minus2", "adni_full"]
_RESULTS = {}


def _gate(model):
    from trace_pd.evaluation import trace_gate
    if model not in _RESULTS:
        _RESULTS[model] = trace_gate.run_gate(model)
    return _RESULTS[model], trace_gate.TOL


@pytest.mark.parametrize("model", GATE_MODELS)
def test_gate_G1_add_up(model):
    r, tol = _gate(model)
    assert r["G1_pair_max"] <= tol["add_up"], r["G1_pair_max"]
    assert r["G1_trajectory_max"] <= tol["add_up"], r["G1_trajectory_max"]


@pytest.mark.parametrize("model", GATE_MODELS)
def test_gate_G2_consistency(model):
    r, tol = _gate(model)
    assert r["G2_triple_max"] <= tol["consistency"], r["G2_triple_max"]
    assert r["G2_closed_vs_lstsq_max"] <= tol["consistency"], r["G2_closed_vs_lstsq_max"]
    assert r["G2_closed_vs_background_max"] <= tol["consistency"], r["G2_closed_vs_background_max"]


@pytest.mark.parametrize("model", GATE_MODELS)
def test_gate_G3_zero(model):
    r, tol = _gate(model)
    assert r["G3_constant_feature_cases"] > 0, "no never-changing feature found: G3a would be vacuous"
    assert r["G3_constant_feature_max"] <= tol["zero"], r["G3_constant_feature_max"]
    assert r["G3_unused_feature_max"] <= tol["zero"], r["G3_unused_feature_max"]


@pytest.mark.parametrize("model", GATE_MODELS)
def test_gate_G4_brute_force(model):
    r, tol = _gate(model)
    assert r["G4_pairs"] >= 2000, r["G4_pairs"]
    for cat in ["nan_ref", "nan_expl", "nan_both"]:
        if r["G4_available_by_nan_category"][cat] > 0:
            assert r["G4_pairs_by_nan_category"][cat] > 0, f"sample has no {cat} pair"
    assert r["G4_max"] <= tol["brute_force"], r["G4_max_by_nan_category"]
    assert r["G4_antisymmetry_max"] <= tol["brute_force"], r["G4_antisymmetry_max"]


@pytest.mark.parametrize("model", GATE_MODELS)
def test_gate_pilot_code_match(model):
    """Supplementary (user request, 14 Oct 2026): the engine's F equals the pilot's own exact
    tree-Shapley algorithm on 2,000 random ordered pairs of any size k."""
    r, tol = _gate(model)
    assert r["pilot_pairs"] >= 2000, r["pilot_pairs"]
    assert r["pilot_max"] <= tol["brute_force"], r["pilot_max"]
