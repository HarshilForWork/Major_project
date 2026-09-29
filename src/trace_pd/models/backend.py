"""One switch for the black-box learner used across evaluation scripts.

    make_classifier("xgb" | "hgb", n_classes)   -> unfitted classifier
    fit_classifier(model, X, y)                 -> fitted, with balanced sample weights

"xgb" needs `pip install xgboost`; "hgb" (sklearn HistGradientBoosting) always works.
Hyper-parameters mirror models/train_xgboost.py and models/train_baseline.py.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


def available_backends():
    try:
        import xgboost  # noqa: F401
        return ["xgb", "hgb"]
    except ImportError:
        return ["hgb"]


def make_classifier(backend="hgb", n_classes=2):
    if backend == "xgb":
        from xgboost import XGBClassifier
        p = dict(max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1,
                 tree_method="hist")
        if n_classes > 2:
            p.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
        else:
            p.update(objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(**p)
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.06, random_state=42)


def fit_classifier(model, X, y, balanced=True):
    """balanced=True for classification metrics; balanced=False when the predicted
    PROBABILITIES are reported to a user (balanced weights push them toward 0.5)."""
    y = np.asarray(y).astype(int)
    if balanced:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def backend_from_argv(argv):
    """--backend xgb|hgb (default: xgb if installed, else hgb)."""
    b = None
    if "--backend" in argv:
        i = argv.index("--backend")
        if i + 1 >= len(argv):
            raise SystemExit("usage: --backend xgb|hgb")
        b = argv[i + 1]
    avail = available_backends()
    if b is None:
        b = avail[0]
    if b not in avail:
        raise SystemExit(f"backend '{b}' not available (installed: {avail}). Try: pip install xgboost")
    return b
