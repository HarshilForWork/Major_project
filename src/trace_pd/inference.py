"""Unified End-to-End Inference Pipeline for TRACE-PD.

Provides seamless integration of:
  1. Feature Assembly & Leakage Guard (26 routine clinical features)
  2. Motor Subtype Classifier (XGBoost)
  3. Conformal Prediction Set (90% Coverage Guarantee)
  4. Longitudinal Feature Deltas & Patient Trajectory Tracking
  5. 12-Month & Next-Visit Transition Risk Modeling
  6. Verified Plain-Language Explanation Orchestration

Usage:
  python -m trace_pd.inference --demo
  python -m trace_pd.inference --patient-id 3001 --visit V04
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from trace_pd import config
from trace_pd.models.conformal import ConformalPredictor
from trace_pd.tracking.tracker import PatientHistoryTracker, VisitRecord
from trace_pd.explanations.orchestrator import ExplanationOrchestrator


class TRACEPDInferencePipeline:
    """Production Inference Pipeline for TRACE-PD."""

    def __init__(self, history_file: Optional[Path] = None):
        # 1. Load Subtype & Conformal Artifacts
        conformal_path = config.MODELS / "conformal_model_production.pkl"
        if not conformal_path.exists():
            raise FileNotFoundError(f"Missing {conformal_path}. Run 'python -m trace_pd.models.conformal' first.")
            
        with open(conformal_path, "rb") as f:
            conf_artifacts = pickle.load(f)
            
        self.subtype_model = conf_artifacts["model_3c"]
        self.classes_3c = conf_artifacts["classes_3c"]
        self.conformal_lac = ConformalPredictor.from_dict(conf_artifacts["conformal_3c_lac"])
        self.conformal_aps = ConformalPredictor.from_dict(conf_artifacts["conformal_3c_aps"])
        self.cheap_features = conf_artifacts["features"]
        
        # 2. Load Transition Risk Artifacts
        trans_path = config.MODELS / "transition_model_production.pkl"
        if not trans_path.exists():
            raise FileNotFoundError(f"Missing {trans_path}. Run 'python -m trace_pd.models.train_transition' first.")
            
        with open(trans_path, "rb") as f:
            trans_artifacts = pickle.load(f)
            
        self.transition_models = trans_artifacts["models"]
        self.transition_features = trans_artifacts["features"]
        
        # 3. Longitudinal Tracker & Explanation Orchestrator
        self.history_file = history_file or (config.DATA / "interim" / "patient_tracker_store.json")
        self.tracker = PatientHistoryTracker.from_json(self.history_file)
        self.orchestrator = ExplanationOrchestrator()

    def predict_visit(
        self,
        patient_id: str,
        visit_id: str,
        visit_month: int,
        features: Dict[str, Any],
        persist: bool = True
    ) -> Dict[str, Any]:
        """Run full TRACE-PD assessment for a clinical visit."""
        # 1. Feature sanitization & assembly
        feature_vector = []
        for feat in self.cheap_features:
            val = features.get(feat, np.nan)
            try:
                feature_vector.append(float(val) if pd.notna(val) else np.nan)
            except (ValueError, TypeError):
                feature_vector.append(np.nan)
                
        X_visit = pd.DataFrame([feature_vector], columns=self.cheap_features)
        
        # 2. Subtype Probability Estimation
        probs_raw = self.subtype_model.predict_proba(X_visit)[0]
        prob_dict = {cls: float(probs_raw[i]) for i, cls in enumerate(self.classes_3c)}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_probs[0][0]
        margin_top2 = float(sorted_probs[0][1] - sorted_probs[1][1])
        
        # 3. Conformal Prediction Set (90% Coverage)
        pred_sets = self.conformal_lac.predict_sets(np.array([probs_raw]))
        conformal_set = pred_sets[0]
        
        # 4. Longitudinal Deltas & Trajectory
        history = self.tracker.get_history(patient_id)
        prev_record = history[-1] if history else None
        prev_conformal_set = prev_record.conformal_set if prev_record else None
        
        # Compute feature deltas
        deltas = self.tracker.compute_feature_deltas(patient_id, {
            k: float(features[k]) for k in features if k in self.cheap_features and pd.notna(features[k])
        })
        
        # 5. Transition Risk Estimation
        delta_cols = [
            "NP2RISE", "NP2TURN", "LEDD_TOTAL_MG", "MCATOT",
            "NP1RTOT", "NP1PTOT", "GDS_TOTAL", "SCOPA_AUT_TOTAL"
        ]
        trans_input_dict = {k: features.get(k, np.nan) for k in self.cheap_features}
        for c in delta_cols:
            trans_input_dict[f"DELTA_{c}"] = deltas.get(c, 0.0)
            
        trans_input_dict["PRIOR_VISITS_COUNT"] = len(history)
        trans_input_dict["PROB_TD"] = prob_dict["TD"]
        trans_input_dict["PROB_PIGD"] = prob_dict["PIGD"]
        trans_input_dict["PROB_INDETERMINATE"] = prob_dict["INDETERMINATE"]
        trans_input_dict["PROB_MARGIN_TOP2"] = margin_top2
        trans_input_dict["CONFORMAL_SET_SIZE"] = len(conformal_set)
        
        X_trans = pd.DataFrame([[trans_input_dict.get(c, np.nan) for c in self.transition_features]],
                               columns=self.transition_features).apply(pd.to_numeric, errors="coerce")
                               
        model_12m = self.transition_models["FLIP_WITHIN_12M"]["model"]
        model_next = self.transition_models["LABEL_FLIPPED_NEXT"]["model"]
        
        p_flip_12m = float(model_12m.predict_proba(X_trans)[0, 1])
        p_flip_next = float(model_next.predict_proba(X_trans)[0, 1])
        
        # 6. Update Tracker
        visit_record = VisitRecord(
            visit_id=visit_id,
            visit_month=visit_month,
            features={k: float(v) for k, v in features.items() if k in self.cheap_features and pd.notna(v)},
            probabilities=prob_dict,
            predicted_label=top_label,
            conformal_set=conformal_set,
            transition_risk_12m=p_flip_12m,
            transition_risk_next=p_flip_next,
        )
        if persist:
            self.tracker.add_visit(patient_id, visit_record)
            self.tracker.to_json(self.history_file)
            
        all_visits = self.tracker.get_history(patient_id)
        trajectory_labels = [r.predicted_label for r in all_visits]
        stability_index = self.tracker.compute_stability_index(patient_id)
        
        # 7. Explanation Orchestration
        payload = self.orchestrator.assemble_payload(
            patient_id=patient_id,
            visit_id=visit_id,
            visit_month=visit_month,
            probabilities=prob_dict,
            conformal_set=conformal_set,
            previous_conformal_set=prev_conformal_set,
            transition_risk_12m=p_flip_12m,
            transition_risk_next=p_flip_next,
            trajectory_labels=trajectory_labels,
            stability_index=stability_index,
            feature_deltas=deltas,
        )
        explanation = self.orchestrator.explain(payload)
        
        return {
            "patient_id": patient_id,
            "visit_id": visit_id,
            "visit_month": visit_month,
            "predicted_subtype": top_label,
            "probabilities": prob_dict,
            "conformal_confidence_set": conformal_set,
            "conformal_set_size": len(conformal_set),
            "transition_risk_12m": round(p_flip_12m, 3),
            "transition_risk_next": round(p_flip_next, 3),
            "stability_index": stability_index,
            "explanation_text": explanation["explanation_text"],
            "structured_payload": payload,
            "validation_status": explanation["validation"],
        }


def run_demo():
    """Run interactive demonstration on a sample patient trajectory from PPMI."""
    print("=" * 74)
    print("TRACE-PD END-TO-END DEMO: Longitudinal Patient Trajectory")
    print("=" * 74)
    
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    # Pick a patient with multiple visits
    sample_patno = 3001
    pat_df = df[df["PATNO"] == sample_patno].sort_values("VISIT_MONTH").copy()
    
    print(f"Tracking Patient ID: {sample_patno} ({len(pat_df)} total visits available in PPMI)")
    
    demo_store_path = config.DATA / "interim" / "demo_tracker.json"
    if demo_store_path.exists():
        demo_store_path.unlink()
        
    pipeline = TRACEPDInferencePipeline(history_file=demo_store_path)
    
    # Run through first 3 visits
    for idx, (_, row) in enumerate(pat_df.head(3).iterrows(), 1):
        visit_code = row["EVENT_ID"]
        month = int(row["VISIT_MONTH"])
        features = row.to_dict()
        true_label = row["LABEL"]
        
        result = pipeline.predict_visit(
            patient_id=str(sample_patno),
            visit_id=visit_code,
            visit_month=month,
            features=features,
            persist=True
        )
        
        print(f"\n>>> VISIT {idx}: {visit_code} (Month {month}) | Ground Truth Stebbins: {true_label}")
        print(f"Predicted Subtype:          {result['predicted_subtype']}")
        print(f"Probabilities:              TD: {result['probabilities']['TD']:.2f}, PIGD: {result['probabilities']['PIGD']:.2f}, IND: {result['probabilities']['INDETERMINATE']:.2f}")
        print(f"Conformal Set (90% conf):   {{{', '.join(result['conformal_confidence_set'])}}}")
        print(f"12-Month Transition Risk:   {result['transition_risk_12m']*100:.1f}%")
        print(f"Trajectory Stability:       {result['stability_index']*100:.1f}%")
        print("\nClinical Summary:")
        print(result["explanation_text"])
        print("-" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACE-PD Inference Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run multi-visit demo on sample PPMI patient")
    parser.add_argument("--patient-id", type=str, help="Patient identifier")
    parser.add_argument("--visit", type=str, default="BL", help="Visit code (e.g. BL, V04)")
    args = parser.parse_args()
    
    if args.demo or not args.patient_id:
        run_demo()
