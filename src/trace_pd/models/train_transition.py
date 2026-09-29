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
from trace_pd.evaluation.splits import assign_patient_folds, log_run


SUBTYPE_PARAMS = dict(
    max_depth=3, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    random_state=42, n_jobs=-1, tree_method="hist",
    objective="multi:softprob", num_class=3, eval_metric="mlogloss",
)
CLASSES_3C = ["TD", "PIGD", "INDETERMINATE"]


def _oof_upstream_outputs(df, X_cheap):
    """Out-of-fold subtype probabilities and conformal set sizes for every row.

    For each patient fold k: fit the subtype model on the labelled visits of the OTHER
    folds, hold out a quarter of those training patients to calibrate LAC, then score
    fold k. Rows of patients in fold k are therefore scored by a model and a calibrator
    that never saw them -- the same situation as a new patient at serving time.

    Unlabelled rows still need a score (they can be the t-1 visit of a labelled pair),
    so they are scored by the model of the fold their patient belongs to.
    """
    fold_of = assign_patient_folds(df)
    folds = np.array([fold_of.get(p, 0) for p in df["PATNO"].to_numpy()])

    labelled = df["LABEL"].isin(CLASSES_3C).to_numpy()
    y_int = np.full(len(df), -1)
    c2i = {c: i for i, c in enumerate(CLASSES_3C)}
    y_int[labelled] = [c2i[c] for c in df.loc[labelled, "LABEL"]]

    probs = np.full((len(df), 3), np.nan)
    set_sizes = np.full(len(df), np.nan)

    for k in sorted(set(folds.tolist())):
        te_rows = np.flatnonzero(folds == k)
        tr_rows = np.flatnonzero((folds != k) & labelled)
        if len(te_rows) == 0 or len(tr_rows) == 0:
            continue

        # carve a calibration slice out of the TRAINING patients only
        tr_groups = df["PATNO"].to_numpy()[tr_rows]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        fit_rel, cal_rel = next(inner.split(X_cheap.iloc[tr_rows], y_int[tr_rows], groups=tr_groups))
        fit_rows, cal_rows = tr_rows[fit_rel], tr_rows[cal_rel]

        m = XGBClassifier(**SUBTYPE_PARAMS)
        m.fit(X_cheap.iloc[fit_rows], y_int[fit_rows],
              sample_weight=compute_sample_weight("balanced", y_int[fit_rows]))

        cal_probs = m.predict_proba(X_cheap.iloc[cal_rows])
        cp = ConformalPredictor(method="lac", alpha=0.10, classes=CLASSES_3C)
        cp.calibrate(cal_probs, np.array(CLASSES_3C)[y_int[cal_rows]])

        te_probs = m.predict_proba(X_cheap.iloc[te_rows])
        probs[te_rows] = te_probs
        set_sizes[te_rows] = [len(s) for s in cp.predict_sets(te_probs)]

    return probs, set_sizes


def build_transition_dataset():
    """Build feature table for transition modeling including deltas and upstream outputs."""
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    cheap_features = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"].tolist())
    
    # Upstream features are now generated out-of-fold inside this script, so the
    # production conformal pickle is no longer an input (it used to be, and using it
    # was the leak).
    #
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
    
    # 3. Upstream subtype-model outputs -- OUT OF FOLD.
    #
    #    Previously these came from the production model, which was fitted on 80% of
    #    patients and then scored ALL of them: for those patients PROB_* encoded their
    #    own true label. They are the top transition features, so the reported AUC was
    #    optimistic. Now each patient is scored by a model that never saw that patient,
    #    using the SAME patient folds the transition CV will use.
    X_cheap = df[cheap_features].apply(pd.to_numeric, errors="coerce")
    probs, set_sizes = _oof_upstream_outputs(df, X_cheap)

    # Classes: ['TD', 'PIGD', 'INDETERMINATE']
    df["PROB_TD"] = probs[:, 0]
    df["PROB_PIGD"] = probs[:, 1]
    df["PROB_INDETERMINATE"] = probs[:, 2]

    # Top-2 probability margin
    sorted_probs = np.sort(probs, axis=1)
    df["PROB_MARGIN_TOP2"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    df["CONFORMAL_SET_SIZE"] = set_sizes

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
    fold_of = assign_patient_folds(df)

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
        
        # SAME patient folds that generated the out-of-fold upstream features.
        sub_folds = np.array([fold_of.get(p, 0) for p in groups])

        tx_params = dict(
            max_depth=3, n_estimators=250, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=4,
            random_state=42, n_jobs=-1, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss",
        )

        oof_preds = np.zeros(n_pairs)
        oof_probs = np.zeros(n_pairs)      # balanced-weight model -> for ranking metrics
        oof_probs_unw = np.zeros(n_pairs)  # unweighted model      -> for Brier

        for k in sorted(set(sub_folds.tolist())):
            te_idx = np.flatnonzero(sub_folds == k)
            tr_idx = np.flatnonzero(sub_folds != k)
            if len(te_idx) == 0 or len(np.unique(y[tr_idx])) < 2:
                continue

            m = XGBClassifier(**tx_params)
            m.fit(X.iloc[tr_idx], y[tr_idx],
                  sample_weight=compute_sample_weight("balanced", y[tr_idx]))
            oof_probs[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
            oof_preds[te_idx] = m.predict(X.iloc[te_idx])

            # Balanced weights deliberately distort the probability scale toward 0.5,
            # so a Brier score computed from them is not a calibration measurement.
            m_unw = XGBClassifier(**tx_params)
            m_unw.fit(X.iloc[tr_idx], y[tr_idx])
            oof_probs_unw[te_idx] = m_unw.predict_proba(X.iloc[te_idx])[:, 1]

        roc_auc = roc_auc_score(y, oof_probs)
        pr_auc = average_precision_score(y, oof_probs)
        bal_acc = balanced_accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds)
        brier = brier_score_loss(y, oof_probs_unw)
        brier_weighted = brier_score_loss(y, oof_probs)
        brier_base = brier_score_loss(y, np.full(n_pairs, flip_rate))

        report_lines += [
            f"\n--- TARGET: {title} ---",
            f"Evaluated Instances: {n_pairs} across {sub['PATNO'].nunique()} patients",
            f"Base Flip Rate:      {flip_rate*100:.1f}% ({int(np.sum(y))} positive flips)",
            f"ROC-AUC:             {roc_auc:.4f}",
            f"PR-AUC:              {pr_auc:.4f}",
            f"Balanced Accuracy:   {bal_acc:.4f}",
            f"F1 Score:            {f1:.4f}",
            f"Brier (unweighted):  {brier:.4f}  vs base-rate {brier_base:.4f} "
            f"({'better' if brier < brier_base else 'NO BETTER'} than always predicting the base rate)",
            f"Brier (bal. weights):{brier_weighted:.4f}  <- not a calibration measure, weights distort the scale",
        ]
        for mname, mval in [("roc_auc", roc_auc), ("pr_auc", pr_auc), ("bal_acc", bal_acc),
                            ("brier_unweighted", brier), ("brier_base_rate", brier_base)]:
            log_run("1", f"transition:{target_col}", "xgboost", tx_params, "dev+test CV (oof upstream)",
                    mname, mval, n=n_pairs, n_patients=sub["PATNO"].nunique(),
                    note="upstream PROB_*/set size now out-of-fold")
        
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
