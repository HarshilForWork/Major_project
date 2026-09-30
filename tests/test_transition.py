"""Unit tests for Transition Risk Modeling."""
import pickle
import pytest
from trace_pd import config


def test_transition_model_artifacts_exist():
    """Verify production transition model artifact exists and contains required targets.

    Skips (rather than fails) on a fresh clone where ``models/*.pkl`` are
    git-ignored and haven't been trained yet.
    """
    model_path = config.MODELS / "transition_model_production.pkl"
    if not model_path.exists():
        pytest.skip("transition_model_production.pkl not found — run `make transition` first")
    
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
        
    assert "models" in artifacts
    assert "features" in artifacts
    assert "LABEL_FLIPPED_NEXT" in artifacts["models"]
    assert "FLIP_WITHIN_12M" in artifacts["models"]
    assert len(artifacts["features"]) > 26


def test_transition_metrics_file():
    """Verify transition metrics report is present and contains evaluation sections."""
    metrics_path = config.METRICS / "transition_results.txt"
    if not metrics_path.exists():
        pytest.skip("transition_results.txt not found — run `make transition` first")
    content = metrics_path.read_text(encoding="utf-8")
    assert "NEXT_VISIT" in content
    assert "12_MONTH_HORIZON" in content
    assert "ROC-AUC" in content
