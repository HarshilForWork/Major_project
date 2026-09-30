"""Unit tests for PatientHistoryTracker."""
import math
import tempfile
from pathlib import Path
from trace_pd.tracking.tracker import PatientHistoryTracker, VisitRecord


def test_tracker_stability_and_deltas():
    tracker = PatientHistoryTracker()
    
    # Visit 1 (Baseline)
    rec1 = VisitRecord(
        visit_id="BL",
        visit_month=0,
        features={"NP2RISE": 0.0, "LEDD_TOTAL_MG": 0.0},
        probabilities={"TD": 0.8, "PIGD": 0.15, "INDETERMINATE": 0.05},
        predicted_label="TD",
        conformal_set=["TD"],
        transition_risk_12m=0.10,
    )
    tracker.add_visit("PAT_001", rec1)
    
    assert tracker.compute_stability_index("PAT_001") == 1.0
    
    # Deltas at visit 2
    feat2 = {"NP2RISE": 1.0, "LEDD_TOTAL_MG": 150.0}
    deltas = tracker.compute_feature_deltas("PAT_001", feat2)
    assert deltas["NP2RISE"] == 1.0
    assert deltas["LEDD_TOTAL_MG"] == 150.0
    
    # Visit 2 (Month 6) - same subtype
    rec2 = VisitRecord(
        visit_id="V02",
        visit_month=6,
        features=feat2,
        probabilities={"TD": 0.7, "PIGD": 0.25, "INDETERMINATE": 0.05},
        predicted_label="TD",
        conformal_set=["TD"],
        transition_risk_12m=0.15,
    )
    tracker.add_visit("PAT_001", rec2)
    assert tracker.compute_stability_index("PAT_001") == 1.0
    
    # Visit 3 (Month 12) - flipped to PIGD
    rec3 = VisitRecord(
        visit_id="V04",
        visit_month=12,
        features={"NP2RISE": 2.0, "LEDD_TOTAL_MG": 300.0},
        probabilities={"TD": 0.2, "PIGD": 0.75, "INDETERMINATE": 0.05},
        predicted_label="PIGD",
        conformal_set=["PIGD"],
        transition_risk_12m=0.20,
    )
    tracker.add_visit("PAT_001", rec3)
    
    # Total visits = 3, transitions = 2, flips = 1 -> stability = 1 - 0.5 = 0.5
    assert tracker.compute_stability_index("PAT_001") == 0.5
    
    # Test JSON serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "history.json"
        tracker.to_json(json_file)
        
        loaded = PatientHistoryTracker.from_json(json_file)
        assert len(loaded.get_history("PAT_001")) == 3
        assert loaded.compute_stability_index("PAT_001") == 0.5


def test_first_visit_deltas_are_nan():
    """First-visit deltas must be NaN (not 0.0), matching training behavior
    where ``groupby().shift()`` produces NaN. XGBoost treats NaN and 0.0
    differently (routes to a different leaf), so mismatching is a
    train/serve skew bug."""
    tracker = PatientHistoryTracker()
    feats = {"NP2RISE": 1.0, "LEDD_TOTAL_MG": 100.0}
    deltas = tracker.compute_feature_deltas("NEW_PAT", feats)
    for k, v in deltas.items():
        assert math.isnan(v), f"First-visit delta for {k} should be NaN, got {v}"


def test_missing_previous_feature_delta_is_nan():
    """If the previous visit is missing a feature value, the delta should be NaN."""
    tracker = PatientHistoryTracker()
    rec = VisitRecord(
        visit_id="BL", visit_month=0,
        features={"NP2RISE": 1.0},  # LEDD is absent
        probabilities={"TD": 0.5, "PIGD": 0.4, "INDETERMINATE": 0.1},
        predicted_label="TD", conformal_set=["TD"],
    )
    tracker.add_visit("PAT_002", rec)
    deltas = tracker.compute_feature_deltas("PAT_002", {"NP2RISE": 2.0, "LEDD_TOTAL_MG": 100.0})
    assert deltas["NP2RISE"] == 1.0
    assert math.isnan(deltas["LEDD_TOTAL_MG"]), "Missing previous value should give NaN"

