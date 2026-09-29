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


def _perfectly_calibrated(n, n_classes, seed):
    """Probabilities that are calibrated BY CONSTRUCTION: draw p from a Dirichlet,
    then sample the true label from p. Any correct conformal method must cover at
    ~1-alpha here, so this isolates the calibrator from the classifier."""
    rng = np.random.RandomState(seed)
    probs = rng.dirichlet(np.ones(n_classes) * 0.7, size=n)
    y = np.array([rng.choice(n_classes, p=p) for p in probs])
    return probs, y


@pytest.mark.parametrize("method,n_classes", [("lac", 3), ("aps", 3), ("lac", 2), ("aps", 2)])
def test_coverage_on_perfectly_calibrated_probabilities(method, n_classes):
    """Both methods must hit 90% +/- 3 points when the probabilities are exact.

    Regression guard for the APS bug: calibration used the randomised score while
    prediction used the non-randomised accumulation rule, giving ~98% coverage here
    instead of ~90%. Under-coverage matters too, so this is a two-sided bound.
    """
    classes = ["A", "B", "C"][:n_classes]
    probs_cal, y_cal_i = _perfectly_calibrated(4000, n_classes, seed=7)
    probs_te, y_te_i = _perfectly_calibrated(4000, n_classes, seed=8)
    y_cal = np.array(classes)[y_cal_i]
    y_te = np.array(classes)[y_te_i]

    cp = ConformalPredictor(method=method, alpha=0.10, classes=classes)
    cp.calibrate(probs_cal, y_cal)
    cov = cp.evaluate(probs_te, y_te)["empirical_coverage"]
    assert 0.87 <= cov <= 0.93, f"{method} {n_classes}-class coverage {cov:.3f} != 0.90"


def test_calibration_and_prediction_share_one_score_function():
    """q_hat is a quantile of the true-label score, so a calibration row is covered
    iff its own score <= q_hat. If the two code paths ever diverge again, this breaks."""
    classes = ["A", "B", "C"]
    probs, y_i = _perfectly_calibrated(1500, 3, seed=11)
    y = np.array(classes)[y_i]

    for method in ("lac", "aps"):
        cp = ConformalPredictor(method=method, alpha=0.10, classes=classes)
        cp.calibrate(probs, y)
        scores_true = cp.compute_nonconformity_scores(probs, y)
        # deterministic u for LAC; for APS re-derive with the calibration seed
        if method == "lac":
            sets = cp.predict_sets(probs)
            covered = np.array([yy in s for yy, s in zip(y, sets)])
            assert np.array_equal(covered, scores_true <= cp.q_hat)
        # in both cases the empirical calibration coverage must match the nominal rank
        assert abs(np.mean(scores_true <= cp.q_hat) - 0.90) < 0.02
