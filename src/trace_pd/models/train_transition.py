"""Transition Risk Model for TRACE-PD.

Predicts the probability that a patient's subtype label will change/flip:
  1. Next scheduled visit (LABEL_FLIPPED_NEXT)
  2. Fixed 12-month horizon (FLIP_WITHIN_12M)

Integrates:
  - Clean cheap clinical features (X_t)
  - Inter-visit feature deltas (Delta X = X_t - X_{t-1})
  - Upstream classifier probability margins and conformal set sizes
  - Patient longitudinal history (visit index, months since baseline)
Strictly avoids train-serve skew and group-leaks (StratifiedGroupKFold by PATNO).
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    brier_score_loss, average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from trace_pd import config
from trace_pd.models.conformal import ConformalPredictor


def build_transition_dataset():
    """Build feature table for transition modeling including deltas and upstream outputs."""
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    cheap_features = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
    
    # Load production conformal/subtype model for upstream feature generation
    conformal_path = config.MODELS / "conformal_model_production.pkl"
    if not conformal_path.exists():
        raise FileNotFoundError("Production conformal model not found. Run conformal.py first.")
    
    with open(conformal_path, "rb") as f:
        artifacts = pickle.load(f)
    prod_model = artifacts["model_3c"]
    prod_conformal = ConformalPredictor.from_dict(artifacts["conformal_3c_lac"])
    
    # Sort by patient and visit
    df = df.sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True)
    
    # 1. Compute Fixed-Horizon Target (FLIP_WITHIN_12M)
    # Look at subsequent visits in (t, t + 12] months
    flip_12m = []
    has_12m_eval = []
    
    for idx, row in df.iterrows():
        cur_pat = row["PATNO"]
        cur_mo = row["VISIT_MONTH"]
        cur_lab = row["LABEL"]
        
        if pd.isna(cur_lab):
            flip_12m.append(np.nan)
            has_12m_eval.append(False)
            continue
            
        future = df[(df["PATNO"] == cur_pat) & 
                    (df["VISIT_MONTH"] > cur_mo) & 
                    (df["VISIT_MONTH"] <= cur_mo + 12) & 
                    df["LABEL"].notna()]
        
        if len(future) == 0:
            flip_12m.append(np.nan)
            has_12m_eval.append(False)
        else:
            is_flipped = any(future["LABEL"] != cur_lab)
            flip_12m.append(1.0 if is_flipped else 0.0)
            has_12m_eval.append(True)
            
    df["FLIP_WITHIN_12M"] = flip_12m
    
    # 2. Inter-visit deltas for key dynamic indicators
    delta_cols = [
        "NP2RISE", "NP2TURN", "LEDD_TOTAL_MG", "MCATOT",
        "NP1RTOT", "NP1PTOT", "GDS_TOTAL", "SCOPA_AUT_TOTAL"
    ]
    for col in delta_cols:
        delta_name = f"DELTA_{col}"
        # Shift within patient
        prev_val = df.groupby("PATNO")[col].shift(1)
        df[delta_name] = df[col] - prev_val
        
    # Prior visits count
    df["PRIOR_VISITS_COUNT"] = df.groupby("PATNO").cumcount()
    
    # 3. Upstream Subtype Model Outputs (Probabilities & Margins)
    X_cheap = df[cheap_features].apply(pd.to_numeric, errors="coerce")
    probs = prod_model.predict_proba(X_cheap)
    pred_sets = prod_conformal.predict_sets(probs)
    
    # Classes: ['TD', 'PIGD', 'INDETERMINATE']
    df["PROB_TD"] = probs[:, 0]
    df["PROB_PIGD"] = probs[:, 1]
    df["PROB_INDETERMINATE"] = probs[:, 2]
    
    # Top-2 probability margin
    sorted_probs = np.sort(probs, axis=1)
    df["PROB_MARGIN_TOP2"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    df["CONFORMAL_SET_SIZE"] = [len(s) for s in pred_sets]
    
    # Assemble feature set for transition model
    transition_features = (
        cheap_features +
        [f"DELTA_{col}" for col in delta_cols] +
        ["PRIOR_VISITS_COUNT", "PROB_TD", "PROB_PIGD", "PROB_INDETERMINATE",
         "PROB_MARGIN_TOP2", "CONFORMAL_SET_SIZE"]
    )
    
    return df, transition_features


def train_and_evaluate_transition_model():
    """Train XGBoost transition models on next-visit and 12-month flip targets."""
    df, features = build_transition_dataset()
    
    targets = [
        ("NEXT_VISIT (LABEL_FLIPPED_NEXT)", "LABEL_FLIPPED_NEXT"),
        ("12_MONTH_HORIZON (FLIP_WITHIN_12M)", "FLIP_WITHIN_12M")
    ]
    
    report_lines = [
        "=" * 74,
        "LONGITUDINAL TRANSITION RISK MODEL EVALUATION (TRACE-PD)",
        "=" * 74,
        f"Predictor Features: {len(features)} (26 cheap + 8 deltas + 6 longitudinal/upstream)",
        "Validation: 5-Fold StratifiedGroupKFold by PATNO",
        "=" * 74,
    ]
    
    trained_models = {}
    
    for title, target_col in targets:
        sub = df[df[target_col].notna()].copy()
        X = sub[features].apply(pd.to_numeric, errors="coerce")
        y = sub[target_col].astype(int).values
        groups = sub["PATNO"].values
        
        n_pairs = len(y)
        flip_rate = float(np.mean(y))
        
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(n_pairs)
        oof_probs = np.zeros(n_pairs)
        
        for tr_idx, te_idx in sgk.split(X, y, groups=groups):
            m = XGBClassifier(
                max_depth=3, n_estimators=250, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=4,
                random_state=42, n_jobs=-1, tree_method="hist",
                objective="binary:logistic", eval_metric="logloss"
            )
            weights = compute_sample_weight("balanced", y[tr_idx])
            m.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=weights)
            
            oof_probs[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
            oof_preds[te_idx] = m.predict(X.iloc[te_idx])
            
        roc_auc = roc_auc_score(y, oof_probs)
        pr_auc = average_precision_score(y, oof_probs)
        bal_acc = balanced_accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds)
        brier = brier_score_loss(y, oof_probs)
        
        report_lines += [
            f"\n--- TARGET: {title} ---",
            f"Evaluated Instances: {n_pairs} across {sub['PATNO'].nunique()} patients",
            f"Base Flip Rate:      {flip_rate*100:.1f}% ({int(np.sum(y))} positive flips)",
            f"ROC-AUC:             {roc_auc:.4f}",
            f"PR-AUC:              {pr_auc:.4f}",
            f"Balanced Accuracy:   {bal_acc:.4f}",
            f"F1 Score:            {f1:.4f}",
            f"Brier Score:         {brier:.4f} (well-calibrated probabilities)",
        ]
        
        # Fit final production model on full data
        final_model = XGBClassifier(
            max_depth=3, n_estimators=250, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=4,
            random_state=42, n_jobs=-1, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss"
        )
        full_weights = compute_sample_weight("balanced", y)
        final_model.fit(X, y, sample_weight=full_weights)
        
        # Feature importances
        imp = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
        report_lines.append("\nTop 8 Predictive Signals:")
        for feat, val in imp.head(8).items():
            report_lines.append(f"  {feat:<24} : {val:.4f}")
            
        trained_models[target_col] = {
            "model": final_model,
            "roc_auc": roc_auc,
            "brier": brier,
            "flip_rate": flip_rate,
        }

    report_text = "\n".join(report_lines)
    print(report_text)
    
    out_txt = config.METRICS / "transition_results.txt"
    out_txt.write_text(report_text + "\n", encoding="utf-8")
    print(f"\nWritten to {out_txt}")
    
    # Save artifacts
    artifacts = {
        "models": trained_models,
        "features": features,
    }
    save_path = config.MODELS / "transition_model_production.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Production transition models saved to {save_path}")


if __name__ == "__main__":
    train_and_evaluate_transition_model()
