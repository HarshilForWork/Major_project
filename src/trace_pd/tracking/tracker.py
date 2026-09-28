"""Longitudinal tracker and history store for patient visit trajectories."""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class VisitRecord:
    """Record of a single clinical visit assessment."""
    visit_id: str
    visit_month: int
    features: Dict[str, float]
    probabilities: Dict[str, float]
    predicted_label: str
    conformal_set: List[str]
    transition_risk_12m: Optional[float] = None
    transition_risk_next: Optional[float] = None
    true_label: Optional[str] = None


class PatientHistoryTracker:
    """Tracks sequential visits, feature deltas, and trajectory stability indices."""

    def __init__(self):
        self._store: Dict[str, List[VisitRecord]] = {}

    def add_visit(self, patient_id: str, record: VisitRecord) -> None:
        """Add a visit record for a patient in chronological order."""
        if patient_id not in self._store:
            self._store[patient_id] = []
        
        # Insert maintaining visit_month sort order
        history = self._store[patient_id]
        history.append(record)
        history.sort(key=lambda r: r.visit_month)

    def get_history(self, patient_id: str) -> List[VisitRecord]:
        """Get list of chronological visit records for a patient."""
        return self._store.get(patient_id, [])

    def get_last_visit(self, patient_id: str) -> Optional[VisitRecord]:
        """Get the most recent previous visit record for a patient."""
        history = self.get_history(patient_id)
        if len(history) >= 2:
            return history[-2]
        return None

    def compute_feature_deltas(self, patient_id: str, current_features: Dict[str, float]) -> Dict[str, float]:
        """Compute delta X = X_t - X_{t-1} against the preceding visit."""
        history = self.get_history(patient_id)
        if not history:
            return {k: 0.0 for k in current_features}
            
        last_rec = history[-1]
        deltas = {}
        for k, v in current_features.items():
            prev_v = last_rec.features.get(k)
            if prev_v is not None and v is not None:
                deltas[k] = float(v) - float(prev_v)
            else:
                deltas[k] = 0.0
        return deltas

    def compute_stability_index(self, patient_id: str) -> float:
        """Compute trajectory stability index:
        
        Stability = 1.0 - (number of subtype flips) / max(visits - 1, 1)
        Returns 1.0 if patient has <= 1 visit.
        """
        history = self.get_history(patient_id)
        if len(history) <= 1:
            return 1.0
            
        flips = 0
        for i in range(1, len(history)):
            prev_label = history[i - 1].predicted_label
            curr_label = history[i].predicted_label
            if prev_label != curr_label:
                flips += 1
                
        total_transitions = len(history) - 1
        stability = 1.0 - (flips / total_transitions)
        return round(max(0.0, min(1.0, stability)), 3)

    def to_json(self, filepath: Path) -> None:
        """Serialize tracker state to JSON file."""
        data = {
            pat_id: [asdict(rec) for rec in recs]
            for pat_id, recs in self._store.items()
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: Path) -> "PatientHistoryTracker":
        """Load tracker state from JSON file."""
        tracker = cls()
        if not filepath.exists():
            return tracker
            
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for pat_id, rec_dicts in data.items():
            for d in rec_dicts:
                rec = VisitRecord(**d)
                tracker.add_visit(pat_id, rec)
        return tracker
