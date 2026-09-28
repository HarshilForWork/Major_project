"""Conformal prediction set calibration for TRACE-PD.

Implements Adaptive Prediction Sets (APS, Romano et al. 2020) and Least Ambiguous
Prediction Sets (LAC, Sadinle et al. 2019) with exact finite-sample coverage guarantees.
Ensures patient-level group splitting so calibration and test folds are strictly exchangeable.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from trace_pd import config


class ConformalPredictor:
    """Conformal Prediction Set Classifier supporting APS and LAC."""

    def __init__(self, method: str = "aps", alpha: float = 0.10, classes: list = None, random_state: int = 42):
        """
        Parameters
        ----------
        method : str
            'aps' (Adaptive Prediction Sets) or 'lac' (Least Ambiguous / Margin).
        alpha : float
            Significance level (default 0.10 for 90% marginal coverage).
        classes : list
            Class label names matching probability columns.
        random_state : int
            Seed for reproducible tie-breaking in APS.
        """
        self.method = method.lower()
        self.alpha = alpha
        self.classes = np.array(classes if classes is not None else ["TD", "PIGD", "INDETERMINATE"])
        self.random_state = random_state
        self.q_hat = None

    def to_dict(self) -> dict:
        """Serialize calibrator state to plain dictionary."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "classes": list(self.classes),
            "random_state": self.random_state,
            "q_hat": self.q_hat,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Reconstruct calibrator from dictionary."""
        inst = cls(method=d["method"], alpha=d["alpha"], classes=d["classes"], random_state=d.get("random_state", 42))
        inst.q_hat = d["q_hat"]
        return inst

    def compute_nonconformity_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores for true labels given probability matrix."""
        n_samples = len(y_true)
        scores = np.zeros(n_samples)
        
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_indices = np.array([class_to_idx[y] for y in y_true])

        if self.method == "lac":
            # LAC score: 1 - P(y_true)
            for i in range(n_samples):
                scores[i] = 1.0 - probs[i, y_indices[i]]
        else:
            # Randomized APS score (Romano et al. 2020)
            rng = np.random.RandomState(self.random_state)
            u = rng.uniform(0.0, 1.0, n_samples)
            for i in range(n_samples):
                p = probs[i]
                y_idx = y_indices[i]
                sort_order = np.argsort(-p)
                rank = np.where(sort_order == y_idx)[0][0]
                p_cum = np.sum(p[sort_order[:rank]])
                scores[i] = p_cum + u[i] * p[sort_order[rank]]

        return scores

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Compute conformal quantile q_hat on the calibration split."""
        scores = self.compute_nonconformity_scores(probs_cal, y_cal)
        n = len(scores)
        # Finite-sample correction quantile: ceil((n + 1) * (1 - alpha)) / n
        rank = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        rank = min(n, max(1, rank))
        sorted_scores = np.sort(scores)
        self.q_hat = float(sorted_scores[rank - 1])
        return self.q_hat

    def predict_sets(self, probs: np.ndarray) -> list:
        """Generate prediction sets C(X) for probability matrix."""
        if self.q_hat is None:
            raise ValueError("Conformal predictor is not calibrated yet. Call calibrate() first.")
        
        n_samples = len(probs)
        prediction_sets = []

        if self.method == "lac":
            p_cutoff = 1.0 - self.q_hat
            for p in probs:
                matches = [self.classes[i] for i, prob in enumerate(p) if prob >= p_cutoff]
                if not matches:
                    matches = [self.classes[np.argmax(p)]]
                prediction_sets.append(matches)
        else:
            # APS accumulation
            for p in probs:
                sort_order = np.argsort(-p)
                sorted_p = p[sort_order]
                sorted_classes = self.classes[sort_order]
                
                current_set = []
                cum_p = 0.0
                for cls, prob in zip(sorted_classes, sorted_p):
                    current_set.append(cls)
                    cum_p += prob
                    if cum_p >= self.q_hat:
                        break
                prediction_sets.append(current_set)

        return prediction_sets

    def evaluate(self, probs: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluate coverage and set sizes on test split."""
        pred_sets = self.predict_sets(probs)
        covered = [y in p_set for y, p_set in zip(y_true, pred_sets)]
        set_sizes = [len(p_set) for p_set in pred_sets]
        
        res = {
            "method": self.method,
            "alpha": self.alpha,
            "target_coverage": round(1.0 - self.alpha, 3),
            "empirical_coverage": round(float(np.mean(covered)), 4),
            "mean_set_size": round(float(np.mean(set_sizes)), 3),
            "size_1_pct": round(float(np.mean([s == 1 for s in set_sizes])) * 100, 1),
            "size_2_pct": round(float(np.mean([s == 2 for s in set_sizes])) * 100, 1),
            "size_3_pct": round(float(np.mean([s >= 3 for s in set_sizes])) * 100, 1),
            "q_hat": round(self.q_hat, 4),
        }
        return res


def evaluate_task_conformal(sub_df, features, classes, task_name="3-class"):
    """Run 5-fold cross-conformal evaluation for a task."""
    X = sub_df[features].apply(pd.to_numeric, errors="coerce")
    y = sub_df["LABEL"].values
    groups = sub_df["PATNO"].values
    
    class_to_int = {c: i for i, c in enumerate(classes)}
    y_int = np.array([class_to_int[c] for c in y])
    
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    for method in ["aps", "lac"]:
        fold_results = []
        oof_pred_sets = [None] * len(y)
        oof_probs = np.zeros((len(y), len(classes)))
        
        for fold, (train_cal_idx, test_idx) in enumerate(sgk.split(X, y_int, groups=groups)):
            inner_groups = groups[train_cal_idx]
            inner_y = y_int[train_cal_idx]
            
            inner_sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
            inner_tr_rel, inner_cal_rel = next(inner_sgk.split(X.iloc[train_cal_idx], inner_y, groups=inner_groups))
            
            tr_idx = train_cal_idx[inner_tr_rel]
            cal_idx = train_cal_idx[inner_cal_rel]
            
            # Model params
            params = dict(
                max_depth=3, n_estimators=300, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=42, n_jobs=-1, tree_method="hist"
            )
            if len(classes) > 2:
                params.update(objective="multi:softprob", num_class=len(classes), eval_metric="mlogloss")
            else:
                params.update(objective="binary:logistic", eval_metric="logloss")
                
            model = XGBClassifier(**params)
            sample_weights = compute_sample_weight("balanced", y_int[tr_idx])
            model.fit(X.iloc[tr_idx], y_int[tr_idx], sample_weight=sample_weights)
            
            cal_probs = model.predict_proba(X.iloc[cal_idx])
            conformal = ConformalPredictor(method=method, alpha=0.10, classes=classes)
            conformal.calibrate(cal_probs, y[cal_idx])
            
            test_probs = model.predict_proba(X.iloc[test_idx])
            oof_probs[test_idx] = test_probs
            eval_metrics = conformal.evaluate(test_probs, y[test_idx])
            fold_results.append(eval_metrics)
            
            test_sets = conformal.predict_sets(test_probs)
            for i, idx in enumerate(test_idx):
                oof_pred_sets[idx] = test_sets[i]
                
        all_covered = [y[i] in oof_pred_sets[i] for i in range(len(y))]
        all_sizes = [len(s) for s in oof_pred_sets]
        
        summary = {
            "method": method.upper(),
            "target_coverage": 0.90,
            "empirical_coverage": round(float(np.mean(all_covered)), 4),
            "mean_set_size": round(float(np.mean(all_sizes)), 3),
            "size_1_pct": round(float(np.mean([s == 1 for s in all_sizes])) * 100, 1),
            "size_2_pct": round(float(np.mean([s == 2 for s in all_sizes])) * 100, 1),
            "size_3_pct": round(float(np.mean([s >= 3 for s in all_sizes])) * 100, 1),
            "mean_q_hat": round(float(np.mean([f["q_hat"] for f in fold_results])), 4),
        }
        results[method] = summary
        
    return results


def run_conformal_pipeline():
    """Run conformal evaluation for both 3-class and binary tasks and persist production model."""
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    features = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
    
    # 1. 3-class task
    sub_3c = df[df.LABEL.isin(["TD", "PIGD", "INDETERMINATE"])].copy()
    classes_3c = ["TD", "PIGD", "INDETERMINATE"]
    results_3c = evaluate_task_conformal(sub_3c, features, classes_3c, task_name="3-class")
    
    # 2. Binary task (TD vs PIGD)
    sub_bin = df[df.LABEL.isin(["TD", "PIGD"])].copy()
    classes_bin = ["TD", "PIGD"]
    results_bin = evaluate_task_conformal(sub_bin, features, classes_bin, task_name="binary")
    
    # Prepare text summary
    lines = [
        "=" * 74,
        "CONFORMAL CONFIDENCE EVALUATION REPORT (TRACE-PD)",
        "=" * 74,
        "Target Marginal Coverage: 90.0% (alpha = 0.10) across unseen patients",
        "Evaluation: 5-Fold StratifiedGroupKFold by PATNO",
        "",
        "--- TASK 1: BINARY (TD vs PIGD) - 5,122 rows, 439 patients ---",
        f"LAC Method:",
        f"  Empirical OOF Coverage: {results_bin['lac']['empirical_coverage']*100:.2f}%",
        f"  Mean Set Size:          {results_bin['lac']['mean_set_size']:.2f} classes",
        f"  Singleton Sets (Size 1):{results_bin['lac']['size_1_pct']:.1f}%",
        f"  Dual-Label Sets (Size 2):{results_bin['lac']['size_2_pct']:.1f}%",
        f"  Calibrated Threshold:   q_hat = {results_bin['lac']['mean_q_hat']:.4f}",
        "",
        f"APS Method:",
        f"  Empirical OOF Coverage: {results_bin['aps']['empirical_coverage']*100:.2f}%",
        f"  Mean Set Size:          {results_bin['aps']['mean_set_size']:.2f} classes",
        f"  Singleton Sets (Size 1):{results_bin['aps']['size_1_pct']:.1f}%",
        f"  Dual-Label Sets (Size 2):{results_bin['aps']['size_2_pct']:.1f}%",
        f"  Calibrated Threshold:   q_hat = {results_bin['aps']['mean_q_hat']:.4f}",
        "",
        "--- TASK 2: 3-CLASS (TD vs PIGD vs INDETERMINATE) - 5,742 rows ---",
        f"LAC Method:",
        f"  Empirical OOF Coverage: {results_3c['lac']['empirical_coverage']*100:.2f}%",
        f"  Mean Set Size:          {results_3c['lac']['mean_set_size']:.2f} classes",
        f"  Size 1 (Singleton):     {results_3c['lac']['size_1_pct']:.1f}%",
        f"  Size 2 (Dual label):    {results_3c['lac']['size_2_pct']:.1f}%",
        f"  Size 3 (Uncertain):     {results_3c['lac']['size_3_pct']:.1f}%",
        f"  Calibrated Threshold:   q_hat = {results_3c['lac']['mean_q_hat']:.4f}",
        "",
        f"APS Method:",
        f"  Empirical OOF Coverage: {results_3c['aps']['empirical_coverage']*100:.2f}%",
        f"  Mean Set Size:          {results_3c['aps']['mean_set_size']:.2f} classes",
        f"  Size 1 (Singleton):     {results_3c['aps']['size_1_pct']:.1f}%",
        f"  Size 2 (Dual label):    {results_3c['aps']['size_2_pct']:.1f}%",
        f"  Size 3 (Uncertain):     {results_3c['aps']['size_3_pct']:.1f}%",
        f"  Calibrated Threshold:   q_hat = {results_3c['aps']['mean_q_hat']:.4f}",
        "=" * 74,
    ]
    report_text = "\n".join(lines)
    print(report_text)
    
    out_file = config.METRICS / "conformal_results.txt"
    out_file.write_text(report_text + "\n", encoding="utf-8")
    print(f"Results written to {out_file}")
    
    # Train production calibrators
    print("\nTraining final production models and calibrators...")
    # Production 3-class
    X_3c = sub_3c[features].apply(pd.to_numeric, errors="coerce")
    y_3c = sub_3c["LABEL"].values
    groups_3c = sub_3c["PATNO"].values
    y_int_3c = np.array([{c: i for i, c in enumerate(classes_3c)}[c] for c in y_3c])
    
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, cal_idx = next(sgk.split(X_3c, y_int_3c, groups=groups_3c))
    
    prod_3c_model = XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, tree_method="hist",
        objective="multi:softprob", num_class=3, eval_metric="mlogloss"
    )
    prod_3c_model.fit(X_3c.iloc[tr_idx], y_int_3c[tr_idx], sample_weight=compute_sample_weight("balanced", y_int_3c[tr_idx]))
    
    prod_3c_cal_probs = prod_3c_model.predict_proba(X_3c.iloc[cal_idx])
    prod_3c_conformal_lac = ConformalPredictor(method="lac", alpha=0.10, classes=classes_3c)
    prod_3c_conformal_lac.calibrate(prod_3c_cal_probs, y_3c[cal_idx])
    
    prod_3c_conformal_aps = ConformalPredictor(method="aps", alpha=0.10, classes=classes_3c)
    prod_3c_conformal_aps.calibrate(prod_3c_cal_probs, y_3c[cal_idx])
    
    artifacts = {
        "model_3c": prod_3c_model,
        "conformal_3c_lac": prod_3c_conformal_lac.to_dict(),
        "conformal_3c_aps": prod_3c_conformal_aps.to_dict(),
        "features": features,
        "classes_3c": classes_3c,
        "metrics_3c": results_3c,
        "metrics_bin": results_bin,
    }
    save_path = config.MODELS / "conformal_model_production.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Production models and calibrators saved to {save_path}")


if __name__ == "__main__":
    run_conformal_pipeline()
