"""Unit tests for Transition Risk Modeling."""
import pickle
import pytest
from trace_pd import config


def test_transition_model_artifacts_exist():
    """Verify production transition model artifact exists and contains required targets."""
    model_path = config.MODELS / "transition_model_production.pkl"
    assert model_path.exists(), "transition_model_production.pkl does not exist"
    
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
    assert metrics_path.exists()
    content = metrics_path.read_text(encoding="utf-8")
    assert "NEXT_VISIT" in content
    assert "12_MONTH_HORIZON" in content
    assert "ROC-AUC" in content
