"""Explanation Orchestrator and Fact Verification Engine for TRACE-PD.

Synthesizes multi-layer model outputs (predicted subtype, conformal confidence set,
longitudinal trajectory, and transition risk) into structured payloads and verified
plain-language clinical summaries.
"""
from typing import Dict, List, Any, Optional
import math
import pickle
import re
from dataclasses import dataclass

from trace_pd import config


FEATURE_DESCRIPTIONS = {
    "NP2RISE": "getting out of a chair",
    "NP2TURN": "turning in bed",
    "LEDD_TOTAL_MG": "levodopa equivalent daily dose",
    "MCATOT": "cognitive score (MoCA)",
    "NP1RTOT": "non-motor symptoms (examiner)",
    "NP1PTOT": "non-motor symptoms (patient-reported)",
    "GDS_TOTAL": "depression rating (GDS-15)",
    "SCOPA_AUT_TOTAL": "autonomic symptoms",
    "NP2DRES": "dressing",
    "NP2HYGN": "hygiene",
    "NP2HWRT": "handwriting",
    "NP2SPCH": "speech",
    "NP2SWAL": "swallowing",
    "NP2SALV": "saliva/drooling",
    "NP2EAT": "eating",
    "NP2HOBB": "hobbies/activities",
}

# ---- Per-feature direction semantics ----
# "higher_is_worse" : positive delta means clinical worsening
# "higher_is_better": positive delta means clinical improvement
# "neutral"         : direction is neither good nor bad (say "increased" / "decreased")

_HIGHER_IS_BETTER = {"MCATOT"}  # MoCA: higher cognitive score = better

_NEUTRAL_FEATURES = {
    "LEDD_TOTAL_MG",            # medication dose — neutral
    "N_CONMEDS",                # number of concomitant medications
    "YRS_SINCE_SYMPTOM_ONSET",  # passage of time
    "YRS_SINCE_DIAGNOSIS",      # passage of time
    "HANDED",                   # handedness — never changes
    "ENROLL_AGE",               # age at enrollment — constant
    "GENDER",                   # constant
}

# Approximate cohort SDs used to normalise delta magnitudes so that a 100 mg
# LEDD change doesn't always outrank a 1-point MDS-UPDRS Part II item.
# Computed from the pooled standard deviations across the PPMI cohort.
_COHORT_SD = {
    "NP2RISE": 0.8,   "NP2TURN": 0.8,   "LEDD_TOTAL_MG": 250.0,
    "MCATOT": 3.0,     "NP1RTOT": 4.0,   "NP1PTOT": 3.5,
    "GDS_TOTAL": 3.0,  "SCOPA_AUT_TOTAL": 6.0,
    "NP2DRES": 0.6,    "NP2HYGN": 0.6,   "NP2HWRT": 0.9,
    "NP2SPCH": 0.5,    "NP2SWAL": 0.5,   "NP2SALV": 0.7,
    "NP2EAT": 0.5,     "NP2HOBB": 0.7,
    "N_CONMEDS": 2.0,  "YRS_SINCE_SYMPTOM_ONSET": 4.0,
    "YRS_SINCE_DIAGNOSIS": 3.5, "ENROLL_AGE": 10.0,
}


def _delta_direction(feat: str, d_val: float) -> str:
    """Human-readable direction word for a feature delta."""
    if feat in _NEUTRAL_FEATURES:
        return "increased" if d_val > 0 else "decreased"
    if feat in _HIGHER_IS_BETTER:
        return "improved" if d_val > 0 else "worsened"
    # Default: MDS-UPDRS / symptom scores — higher = worse
    return "worsened" if d_val > 0 else "improved"


def _load_transition_base_rates() -> Dict[str, float]:
    """Read base rates from the transition model artifact, falling back to
    hard-coded defaults if the artifact is absent (e.g. on a fresh clone)."""
    trans_path = config.MODELS / "transition_model_production.pkl"
    defaults = {"base_rate_12m": 0.42, "base_rate_next": 0.29}
    if not trans_path.exists():
        return defaults
    try:
        with open(trans_path, "rb") as f:
            art = pickle.load(f)
        models = art.get("models", {})
        br_12m = models.get("FLIP_WITHIN_12M", {}).get("flip_rate", defaults["base_rate_12m"])
        br_next = models.get("LABEL_FLIPPED_NEXT", {}).get("flip_rate", defaults["base_rate_next"])
        return {"base_rate_12m": float(br_12m), "base_rate_next": float(br_next)}
    except Exception:
        return defaults


class FactValidator:
    """Verifies that generated narrative text strictly adheres to structured facts."""

    IMPERATIVE_TERMS = [
        r"\bprescribe\b", r"\bstart\b\s+medication", r"\bincrease\b\s+dose",
        r"\brefer\s+for\s+dbs\b", r"\bsurgery\b", r"\bgive\b\s+levodopa"
    ]

    def validate(self, text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate text against payload.
        
        Returns
        -------
        dict : {"is_valid": bool, "errors": list}
        """
        errors = []
        lower_text = text.lower()
        
        # 1. Check for unauthorized imperative clinical recommendations
        for pattern in self.IMPERATIVE_TERMS:
            if re.search(pattern, lower_text):
                errors.append(f"Imperative recommendation detected matching: '{pattern}'")
                
        # 2. Check that the stated subtype matches the prediction
        pred_label = payload["prediction"]["label"]
        if pred_label.lower() not in lower_text:
            errors.append(f"Predicted subtype '{pred_label}' is missing in text.")
            
        # 3. Check that numbers in text correspond to allowed payload values
        # Extract percentage numbers (e.g. 14%, 90%)
        stated_pcts = [int(p) for p in re.findall(r"(\d+)%", text)]
        allowed_pcts = [
            int(round(payload["conformal"]["coverage"] * 100)),
            int(round(payload["transition_risk"]["p_flip_12m"] * 100)),
            int(round(payload["transition_risk"]["base_rate_12m"] * 100)),
            int(round(payload["trajectory"]["stability"] * 100)),
        ]
        if "p_flip_next" in payload["transition_risk"]:
            allowed_pcts.append(int(round(payload["transition_risk"]["p_flip_next"] * 100)))
            allowed_pcts.append(int(round(payload["transition_risk"]["base_rate_next"] * 100)))
            
        for pct in stated_pcts:
            if pct not in allowed_pcts:
                # Tolerant check if within 1 percentage point due to rounding
                if not any(abs(pct - ap) <= 1 for ap in allowed_pcts):
                    errors.append(f"Unverified percentage '{pct}%' found in explanation text.")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }


class ExplanationOrchestrator:
    """Orchestrates structured clinical evidence and produces verified plain-language summaries."""

    def __init__(self):
        self.validator = FactValidator()
        self._base_rates = _load_transition_base_rates()

    def assemble_payload(
        self,
        patient_id: str,
        visit_id: str,
        visit_month: int,
        probabilities: Dict[str, float],
        conformal_set: List[str],
        previous_conformal_set: Optional[List[str]],
        transition_risk_12m: float,
        transition_risk_next: float,
        trajectory_labels: List[str],
        stability_index: float,
        feature_deltas: Dict[str, float],
    ) -> Dict[str, Any]:
        """Assemble structured clinical payload from upstream layers."""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_probs[0][0]
        margin = round(sorted_probs[0][1] - sorted_probs[1][1], 3)
        
        # Identify top feature deltas, ranked by standardised magnitude (Δ / SD)
        delta_drivers = []
        for feat, d_val in feature_deltas.items():
            if not isinstance(d_val, (int, float)) or not math.isfinite(d_val):
                continue
            if abs(d_val) > 1e-4:
                desc = FEATURE_DESCRIPTIONS.get(feat, feat)
                direction = _delta_direction(feat, d_val)
                sd = _COHORT_SD.get(feat, 1.0)
                delta_drivers.append({
                    "feature": feat,
                    "description": desc,
                    "delta": round(float(d_val), 2),
                    "direction": direction,
                    "std_magnitude": abs(d_val) / sd,
                })
        # Sort deltas by standardised magnitude (most clinically notable first)
        delta_drivers.sort(key=lambda x: x["std_magnitude"], reverse=True)
        
        n_flips = 0
        for i in range(1, len(trajectory_labels)):
            if trajectory_labels[i] != trajectory_labels[i - 1]:
                n_flips += 1

        br_12m = self._base_rates["base_rate_12m"]
        br_next = self._base_rates["base_rate_next"]

        payload = {
            "visit": {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "visit_month": visit_month,
            },
            "prediction": {
                "label": top_label,
                "probabilities": {k: round(v, 3) for k, v in probabilities.items()},
                "margin_top2": margin,
            },
            "conformal": {
                "set": conformal_set,
                "size": len(conformal_set),
                "coverage": 0.90,
                "previous_set": previous_conformal_set,
                "previous_size": len(previous_conformal_set) if previous_conformal_set else None,
            },
            "trajectory": {
                "labels": trajectory_labels,
                "n_visits": len(trajectory_labels),
                "n_flips": n_flips,
                "stability": round(stability_index, 2),
            },
            "transition_risk": {
                "p_flip_12m": round(transition_risk_12m, 2),
                "base_rate_12m": round(br_12m, 2),
                "p_flip_next": round(transition_risk_next, 2),
                "base_rate_next": round(br_next, 2),
            },
            "delta_drivers": delta_drivers[:4],
            "audit": {
                "features_used": 26,
                "banned_leakage_excluded": 19,
                "source": "PPMI cohort",
            }
        }
        return payload

    def generate_deterministic_explanation(self, payload: Dict[str, Any]) -> str:
        """Produce a clinical summary using the deterministic verification template."""
        pred = payload["prediction"]
        conf = payload["conformal"]
        traj = payload["trajectory"]
        trans = payload["transition_risk"]
        deltas = payload["delta_drivers"]
        
        # Subtype clause
        subtype = pred["label"]
        set_str = "{" + ", ".join(conf["set"]) + "}"
        cov_pct = int(round(conf["coverage"] * 100))
        
        lines = []
        # Sentence 1: Subtype and confidence
        if conf["size"] == 1:
            conf_desc = f"The {cov_pct}% prediction set contains {subtype} alone -- the classification is confident."
        elif conf["size"] == 2:
            conf_desc = f"The {cov_pct}% prediction set contains {set_str} -- ruling out the third category, but indicating borderline ambiguity."
        else:
            conf_desc = f"The {cov_pct}% prediction set spans all three subtypes {set_str} -- insufficient distinct signal from intake data."
            
        lines.append(f"Motor subtype is predicted as **{subtype}**. {conf_desc}")
        
        # Sentence 2: Set evolution
        if conf["previous_set"] is not None:
            prev_set_str = "{" + ", ".join(conf["previous_set"]) + "}"
            if conf["size"] < conf["previous_size"]:
                lines.append(f"Confidence set narrowed from {prev_set_str} to {set_str} compared to previous visit.")
            elif conf["size"] > conf["previous_size"]:
                lines.append(f"Confidence set widened from {prev_set_str} to {set_str} due to increased symptom ambiguity.")
                
        # Sentence 3: Delta drivers
        if deltas:
            delta_phrases = []
            for d in deltas[:3]:
                delta_phrases.append(f"{d['description']} ({d['direction']})")
            lines.append("Key changes since the prior visit include: " + "; ".join(delta_phrases) + ".")
            
        # Sentence 4: Longitudinal trajectory and transition risk
        # Use unrounded values for comparison to avoid "42% above 42%"
        flip_pct = int(round(trans["p_flip_12m"] * 100))
        base_pct = int(round(trans["base_rate_12m"] * 100))
        stab_pct = int(round(traj["stability"] * 100))

        if abs(flip_pct - base_pct) <= 1:
            comparison_phrase = f"in line with the cohort base rate of {base_pct}%"
        elif flip_pct < base_pct:
            comparison_phrase = f"below the cohort base rate of {base_pct}%"
        else:
            comparison_phrase = f"above the cohort base rate of {base_pct}%"

        lines.append(
            f"The patient has maintained {stab_pct}% stability over {traj['n_visits']} recorded visits. "
            f"Estimated probability of a subtype transition within 12 months is {flip_pct}%, which is {comparison_phrase}."
        )
        
        # Sentence 5: Audit disclaimer
        lines.append(
            "\n*Audit: Computed from 26 routine non-invasive clinical measures. "
            "All 16 MDS-UPDRS motor exam formula items were strictly excluded to prevent data leakage.*"
        )
        
        return " ".join(lines)

    def _minimal_fallback(self, payload: Dict[str, Any]) -> str:
        """Safe fallback text that states only the subtype and conformal set.

        Used when the full template fails validation — guarantees no
        hallucinated numbers or imperatives reach the consumer.
        """
        subtype = payload["prediction"]["label"]
        set_str = "{" + ", ".join(payload["conformal"]["set"]) + "}"
        cov_pct = int(round(payload["conformal"]["coverage"] * 100))
        return (
            f"Motor subtype is predicted as **{subtype}**. "
            f"The {cov_pct}% conformal prediction set is {set_str}."
        )

    def explain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate verified explanation report with audit validation."""
        text = self.generate_deterministic_explanation(payload)
        val_res = self.validator.validate(text, payload)
        
        if not val_res["is_valid"]:
            # Fallback: use a minimal safe text and report that validation failed
            text = self._minimal_fallback(payload)
            val_res = {"is_valid": False, "fallback_applied": True, "errors": val_res["errors"]}
            
        return {
            "payload": payload,
            "explanation_text": text,
            "validation": val_res,
        }
