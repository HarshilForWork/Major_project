"""Unit tests for Explanation Orchestrator and Fact Validator."""
from trace_pd.explanations.orchestrator import ExplanationOrchestrator, FactValidator


def test_fact_validator_catches_hallucinations_and_imperatives():
    validator = FactValidator()
    payload = {
        "prediction": {"label": "PIGD", "margin_top2": 0.20},
        "conformal": {"set": ["PIGD"], "size": 1, "coverage": 0.90},
        "trajectory": {"stability": 0.85},
        "transition_risk": {"p_flip_12m": 0.15, "base_rate_12m": 0.42},
    }
    
    # 1. Text with invalid imperative
    bad_text = "Motor subtype is PIGD. You should prescribe medication immediately."
    res = validator.validate(bad_text, payload)
    assert not res["is_valid"]
    assert any("Imperative" in err for err in res["errors"])
    
    # 2. Text with wrong percentage
    bad_pct_text = "Motor subtype is PIGD. There is a 73% probability of transition."
    res_pct = validator.validate(bad_pct_text, payload)
    assert not res_pct["is_valid"]
    assert any("73%" in err for err in res_pct["errors"])


def test_orchestrator_generates_verified_summary():
    orchestrator = ExplanationOrchestrator()
    payload = orchestrator.assemble_payload(
        patient_id="PAT_4082",
        visit_id="V04",
        visit_month=12,
        probabilities={"TD": 0.15, "PIGD": 0.80, "INDETERMINATE": 0.05},
        conformal_set=["PIGD"],
        previous_conformal_set=["TD", "PIGD"],
        transition_risk_12m=0.14,
        transition_risk_next=0.10,
        trajectory_labels=["TD", "PIGD"],
        stability_index=0.50,
        feature_deltas={"NP2RISE": 1.0, "LEDD_TOTAL_MG": 150.0},
    )
    
    out = orchestrator.explain(payload)
    assert out["validation"]["is_valid"]
    assert "PIGD" in out["explanation_text"]
    assert "14%" in out["explanation_text"]
    assert "getting out of a chair" in out["explanation_text"]
    assert "Audit" in out["explanation_text"]
