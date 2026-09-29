"""Integration test for TRACEPDInferencePipeline."""
import tempfile
from pathlib import Path
import pandas as pd
from trace_pd import config
from trace_pd.inference import TRACEPDInferencePipeline


def test_end_to_end_inference():
    """Verify that the end-to-end pipeline processes a visit and outputs verified reports."""
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    sample_row = df.iloc[0].to_dict()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "test_tracker.json"
        pipeline = TRACEPDInferencePipeline(history_file=history_path)
        
        result = pipeline.predict_visit(
            patient_id="TEST_PAT",
            visit_id=str(sample_row.get("EVENT_ID", "BL")),
            visit_month=int(sample_row.get("VISIT_MONTH", 0)),
            features=sample_row,
            persist=True
        )
        
        assert "predicted_subtype" in result
        assert result["predicted_subtype"] in ["TD", "PIGD", "INDETERMINATE"]
        assert len(result["conformal_confidence_set"]) >= 1
        assert 0.0 <= result["transition_risk_12m"] <= 1.0
        assert 0.0 <= result["transition_risk_next"] <= 1.0
        assert result["stability_index"] == 1.0
        assert len(result["explanation_text"]) > 20
        assert result["validation_status"]["is_valid"]
