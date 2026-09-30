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


def test_validator_rejects_and_produces_fallback():
    """When the full template fails validation, the orchestrator must return
    is_valid=False with a safe fallback — NOT silently accept the bad text."""
    orchestrator = ExplanationOrchestrator()
    payload = orchestrator.assemble_payload(
        patient_id="PAT_9999",
        visit_id="BL",
        visit_month=0,
        probabilities={"TD": 0.60, "PIGD": 0.30, "INDETERMINATE": 0.10},
        conformal_set=["TD"],
        previous_conformal_set=None,
        transition_risk_12m=0.30,
        transition_risk_next=0.20,
        trajectory_labels=["TD"],
        stability_index=1.0,
        feature_deltas={},
    )

    # Manually inject an imperative into the payload text path to trigger rejection
    class BadOrchestrator(ExplanationOrchestrator):
        def generate_deterministic_explanation(self, payload):
            return "Motor subtype is TD. You should prescribe medication."

    bad = BadOrchestrator()
    result = bad.explain(payload)
    assert not result["validation"]["is_valid"]
    assert result["validation"].get("fallback_applied") is True
    # The fallback text should be safe (no imperative)
    assert "prescribe" not in result["explanation_text"].lower()
    assert "TD" in result["explanation_text"]


def test_neutral_feature_direction():
    """Features like 'time since onset' should say 'increased', not 'worsened'."""
    orchestrator = ExplanationOrchestrator()
    payload = orchestrator.assemble_payload(
        patient_id="PAT_5555",
        visit_id="V04",
        visit_month=12,
        probabilities={"TD": 0.50, "PIGD": 0.40, "INDETERMINATE": 0.10},
        conformal_set=["TD", "PIGD"],
        previous_conformal_set=None,
        transition_risk_12m=0.42,
        transition_risk_next=0.29,
        trajectory_labels=["TD"],
        stability_index=1.0,
        feature_deltas={"YRS_SINCE_SYMPTOM_ONSET": 0.5, "N_CONMEDS": 1.0},
    )

    # Check the delta_drivers have neutral direction
    for dd in payload["delta_drivers"]:
        assert dd["direction"] in ("increased", "decreased"), (
            f"{dd['feature']} should use neutral wording, got '{dd['direction']}'"
        )


def test_inline_comparison_for_matching_rates():
    """When the flip probability matches the base rate (within ±1 pp), the
    explanation should say 'in line with' instead of 'above' or 'below'."""
    orchestrator = ExplanationOrchestrator()
    payload = orchestrator.assemble_payload(
        patient_id="PAT_7777",
        visit_id="V02",
        visit_month=6,
        probabilities={"TD": 0.50, "PIGD": 0.40, "INDETERMINATE": 0.10},
        conformal_set=["TD", "PIGD"],
        previous_conformal_set=None,
        transition_risk_12m=0.42,  # same as default base rate
        transition_risk_next=0.29,
        trajectory_labels=["TD"],
        stability_index=1.0,
        feature_deltas={},
    )

    text = orchestrator.generate_deterministic_explanation(payload)
    assert "in line with" in text
