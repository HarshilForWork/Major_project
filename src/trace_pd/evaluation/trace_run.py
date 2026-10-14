"""TRACE runner: models and cohorts (Stage 1) and the per-patient cache (Stage 2).

Brief: docs/TRACE_BRIEF.md. Config: configs/trace.yaml.

Models (all `backend.make_classifier("xgb", 2)` + balanced weights, trained on the locked
DEVELOPMENT split only, no tuning on test):
    ppmi         TD vs PIGD, 26 features         (== models/ar1_shap_artifacts.pkl classifier)
    adni_minus     CN vs Dementia without MMSE_TOTAL, FAQ_TOTAL                  -- ADNI PRIMARY (M_minus)
    adni_minus_lm  ... also without LOGMEM_DELAYED                               -- sensitivity (M_minus_LM)
    adni_minus2    ... also without LOGMEM_DELAYED, LOGMEM_IMMEDIATE             -- sensitivity (M_minus2)
    adni_full    all 23 features             (== models/ar1_shap_artifacts_adni.pkl) -- sensitivity
(User decisions, 14 Oct 2026, results/PROGRESS.md.)

Analysis patients: locked-test patients with >= min_visits visits, sorted by VISIT_MONTH.
visits="all" (primary) keeps every visit the model scores; visits="labelled" (sensitivity)
keeps only visits with a non-missing LABEL, and the >= min_visits rule is applied after
filtering. Patients are referred to only by an anonymous index = position in the sorted
list of eligible patients; IDs never leave this process.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_run stage1
      PYTHONPATH=src python -m trace_pd.evaluation.trace_run compute [model[:labelled] ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from xgboost import Booster

from trace_pd import config
from trace_pd.explain import trace_engine as te
from trace_pd.models.backend import make_classifier, fit_classifier

CFG = yaml.safe_load(open(config.ROOT / "configs" / "trace.yaml", encoding="utf-8"))
MODEL_DIR = config.ROOT / "models" / "trace"
CACHE = config.ROOT / "results" / "cache"

MODELS = {
    "ppmi": dict(cohort="ppmi", drop=[]),
    "adni_minus": dict(cohort="adni", drop=["MMSE_TOTAL", "FAQ_TOTAL"]),
    "adni_minus_lm": dict(cohort="adni", drop=["MMSE_TOTAL", "FAQ_TOTAL", "LOGMEM_DELAYED"]),
    "adni_minus2": dict(cohort="adni", drop=["MMSE_TOTAL", "FAQ_TOTAL", "LOGMEM_DELAYED", "LOGMEM_IMMEDIATE"]),
    "adni_full": dict(cohort="adni", drop=[]),
}
COHORT = {
    "ppmi": dict(id="PATNO", pos="PIGD", neg="TD"),
    "adni": dict(id="RID", pos="Dementia", neg="CN"),
}


# ------------------------------------------------------------------------------ data
_DATA = {}


def load_cohort(cohort):
    """(dev, test, features) with the project's own loading rules, memoised."""
    if cohort in _DATA:
        return _DATA[cohort]
    if cohort == "ppmi":
        from trace_pd.evaluation.splits import split_dev_test
        df = pd.read_csv(config.LONG_TABLE, low_memory=False)
        dd = pd.read_csv(config.DICTIONARY)
        F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
        df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    else:
        from trace_pd.evaluation.splits_adni import split_dev_test
        df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
        dd = pd.read_csv(config.ADNI_DICTIONARY)
        F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
        for c in F:                      # same encoding as evaluate_ar1_shap_adni.load()
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.factorize(df[c], sort=True)[0].astype(float)
                df.loc[df[c] < 0, c] = np.nan
        df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    dev, test = split_dev_test(df)
    _DATA[cohort] = (dev, test, F)
    return _DATA[cohort]


def features(model_name):
    spec = MODELS[model_name]
    _, _, F = load_cohort(spec["cohort"])
    missing = [c for c in spec["drop"] if c not in F]
    assert not missing, f"cannot drop absent features {missing}"
    return [c for c in F if c not in spec["drop"]]


def analysis_patients(cohort, visits="all"):
    """List of (anon_index, visits frame) for locked-test patients with >= min_visits."""
    assert visits in ("all", "labelled"), visits
    _, test, _ = load_cohort(cohort)
    if visits == "labelled":
        test = test[test.LABEL.notna()]
    idc = COHORT[cohort]["id"]
    out = []
    for i, (_, g) in enumerate(
            (pid, g) for pid, g in sorted(test.groupby(idc), key=lambda kv: kv[0])
            if len(g) >= CFG["min_visits"]):
        out.append((i, g.sort_values("VISIT_MONTH", kind="mergesort").reset_index(drop=True)))
    return out


# ---------------------------------------------------------------------------- models
def get_model(name):
    """Load models/trace/<name>.json, or train it on the development split and save it."""
    path = MODEL_DIR / f"{name}.json"
    if path.exists():
        b = Booster()
        b.load_model(path)
        return b
    spec = MODELS[name]
    dev, _, _ = load_cohort(spec["cohort"])
    c = COHORT[spec["cohort"]]
    F = features(name)
    s = dev[dev.LABEL.isin([c["neg"], c["pos"]])]
    clf = fit_classifier(make_classifier("xgb", 2), s[F].to_numpy(float), (s.LABEL == c["pos"]).astype(int))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clf.get_booster().save_model(path)
    b = Booster()
    b.load_model(path)
    return b


def locked_test_ba(name, b):
    spec = MODELS[name]
    _, test, _ = load_cohort(spec["cohort"])
    c = COHORT[spec["cohort"]]
    tt = test[test.LABEL.isin([c["neg"], c["pos"]])]
    thr = 0.0   # predict() at probability 0.5 == margin 0 (what every reported BA used)
    pred = (te.margin(b, tt[features(name)].to_numpy(float)) > thr).astype(int)
    return balanced_accuracy_score((tt.LABEL == c["pos"]).astype(int), pred), len(tt)


def stage1():
    import pickle

    lines = ["Stage 1: models and cohorts", ""]
    for name in MODELS:
        b = get_model(name)
        bs = b.attributes().get("best_iteration")
        ba, n = locked_test_ba(name, b)
        lines.append(f"{name:<14} features {len(features(name)):>2}  trees {b.num_boosted_rounds()}  "
                     f"best_iteration {bs}  locked-test BA {ba:.3f} (n visits {n})  "
                     f"output range bound R {te.tree_output_range(b):.3f}")
    # the saved TRACE models must equal the models behind the AR(1)-SHAP results
    for name, pkl in [("ppmi", "ar1_shap_artifacts.pkl"), ("adni_full", "ar1_shap_artifacts_adni.pkl")]:
        art = pickle.load(open(config.ROOT / "models" / pkl, "rb"))
        _, test, _ = load_cohort(MODELS[name]["cohort"])
        X = test[features(name)].to_numpy(float)
        diff = np.abs(art["classifier"].predict(X, output_margin=True) - te.margin(get_model(name), X)).max()
        lines.append(f"{name}: max |margin difference| vs {pkl} classifier = {diff:.2e}")
    lines.append("")
    for cohort in COHORT:
        for visits in ("all", "labelled"):
            pats = analysis_patients(cohort, visits)
            T = np.array([len(g) for _, g in pats])
            lines.append(f"{cohort} [{visits} visits]: analysis patients (locked test, >= {CFG['min_visits']} "
                         f"visits) = {len(pats)}; visits median {np.median(T):.0f}, range {T.min()}-{T.max()}, "
                         f"total {T.sum()}; pairs per model (T^2) = {int((T ** 2).sum())}")
    print("\n".join(lines))
    return lines


# ----------------------------------------------------------------------------- cache
def cache_dir(name, visits="all"):
    d = CACHE / (name if visits == "all" else f"{name}__{visits}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute(name, force=False, visits="all"):
    """Per patient: F (T,T,d), diag_max, margins f (T,), u_bg (background route). Saved to
    results/cache/<name>[__labelled]/p<idx>.npz (gitignored)."""
    b = get_model(name)
    F_cols = features(name)
    pats = analysis_patients(MODELS[name]["cohort"], visits)
    t0 = time.time()
    for k, (i, g) in enumerate(pats):
        out = cache_dir(name, visits) / f"p{i:04d}.npz"
        if out.exists() and not force:
            continue
        X = g[F_cols].to_numpy(float)
        F, diag_max = te.pairwise_F(b, X)
        tmp = out.with_name(out.stem + ".tmp.npz")          # atomic: a killed run never leaves a half file
        np.savez(tmp, F=F, diag_max=diag_max, f=te.margin(b, X), u_bg=te.trace_via_background(b, X), X=X)
        os.replace(tmp, out)
        if (k + 1) % 50 == 0:
            print(f"  {name}: {k + 1}/{len(pats)} patients, {time.time() - t0:.0f}s", flush=True)
    print(f"{name} [{visits}]: cache complete ({len(pats)} patients, {time.time() - t0:.0f}s)")


def load_cache(name, visits="all"):
    """List of dicts (one per analysis patient, in anonymous-index order)."""
    pats = analysis_patients(MODELS[name]["cohort"], visits)
    out = []
    for i, _ in pats:
        p = cache_dir(name, visits) / f"p{i:04d}.npz"
        if not p.exists():
            compute(name, visits=visits)
        z = np.load(p)
        out.append({k: z[k] for k in z.files} | {"idx": i})
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if cmd == "stage1":
        stage1()
    elif cmd == "compute":
        for spec in (sys.argv[2:] or list(MODELS)):
            n, _, v = spec.partition(":")
            compute(n, visits=v or "all")
