"""Unit tests for Conformal Confidence Predictor."""
import numpy as np
import pytest

from trace_pd.models.conformal import ConformalPredictor


def test_conformal_lac_coverage():
    """Verify that LAC calibration achieves target empirical coverage on synthetic test data."""
    np.random.seed(42)
    n_cal = 500
    n_test = 500
    classes = ["TD", "PIGD"]
    
    # Synthetic well-separated probabilities
    p1_cal = np.random.beta(5, 2, n_cal)
    probs_cal = np.column_stack([p1_cal, 1 - p1_cal])
    y_cal = np.array([classes[0] if p > 0.5 else classes[1] for p in p1_cal])
    
    conformal = ConformalPredictor(method="lac", alpha=0.10, classes=classes)
    q_hat = conformal.calibrate(probs_cal, y_cal)
    assert 0.0 < q_hat < 1.0
    
    # Test data
    p1_test = np.random.beta(5, 2, n_test)
    probs_test = np.column_stack([p1_test, 1 - p1_test])
    y_test = np.array([classes[0] if p > 0.5 else classes[1] for p in p1_test])
    
    eval_res = conformal.evaluate(probs_test, y_test)
    assert eval_res["empirical_coverage"] >= 0.88
    assert 1.0 <= eval_res["mean_set_size"] <= 2.0


def test_conformal_aps_prediction_sets():
    """Verify APS prediction sets are valid subsets of classes."""
    classes = ["TD", "PIGD", "INDETERMINATE"]
    conformal = ConformalPredictor(method="aps", alpha=0.10, classes=classes)
    
    # Calibrate on dummy data
    probs_cal = np.array([
        [0.8, 0.15, 0.05],
        [0.1, 0.7, 0.2],
        [0.35, 0.45, 0.2],
        [0.6, 0.3, 0.1],
        [0.2, 0.2, 0.6],
    ])
    y_cal = np.array(["TD", "PIGD", "PIGD", "TD", "INDETERMINATE"])
    conformal.calibrate(probs_cal, y_cal)
    
    test_probs = np.array([
        [0.95, 0.03, 0.02],
        [0.34, 0.33, 0.33],
    ])
    sets = conformal.predict_sets(test_probs)
    assert len(sets) == 2
    for s in sets:
        assert len(s) >= 1
        assert set(s).issubset(set(classes))
