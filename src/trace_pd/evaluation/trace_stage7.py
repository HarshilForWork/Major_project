"""TRACE Stage 7: stability (brief metrics 8 and 9). docs/TRACE_BRIEF.md.

(a) Delete one visit q (real records only; no replacement visits) and measure, for every pair
    (s, t) with q not in {s, t}, how much each method's explanation A(s->t) moves:
    max_j |Delta A_j| and mean_j |Delta A_j|. Reported separately for q = first, intermediate,
    last visit. Patients with T >= 4. Exact closed forms (deleting q only removes row/column q
    of F, because phi(r->t) depends only on visits r and t):
        trace        u'(t) = (T u(t) - F[q, t]) / (T - 1)
        chain        changes only when s < q < t: + F[q-1,q+1] - F[q-1,q] - F[q,q+1]
        fixed_first  changes only when q = 0 (new reference = visit 1)
        fixed_last   changes only when q = T-1 (new reference = visit T-2)
        direct, population, path_dep: never change (per-pair or per-visit quantities)
        ar1          changes visit q+1's attribution (its previous visit becomes q-1); for q = 0,
                     visit 1 loses its previous visit, so pairs involving it are undefined.
    Bound check (Proposition 4): max_j |Delta psi_j(s->t)| <= 4R/(T-1), R = sum over trees of
    (max leaf - min leaf). A violation means a bug.
(b) Retraining stability: the 10 seed models of Stage 4; per pair, the Spearman correlation
    across features of a method's explanation between two retrained models, averaged over the
    45 model pairs, then within patient, then over patients. AR(1) is excluded here (about 1 s
    per visit per model; deviation).

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_stage7 [deletion|retrain] [model ...]
Out:  results/stage7/  (aggregates only)
"""
import json
import sys
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.evaluation import trace_baselines as tb
from trace_pd.evaluation.trace_stage4 import boot, seed_models
from trace_pd.explain import trace_engine as te

OUT = config.ROOT / "results" / "stage7"
MODELS = ["ppmi", "adni_minus", "adni_minus_lm", "adni_minus2"]
DEL_METHODS = ["trace", "chain", "fixed_first", "fixed_last", "direct", "population", "path_dep", "ar1"]
CASES = ["first", "intermediate", "last"]


# ------------------------------------------------------------------------------ deletion
def deletion_patient(F, ar=None):
    """Per case and method: list of (max_j|dA|, mean_j|dA|) over (pair, q). Also the bound ratio."""
    T, _, d = F.shape
    u, psi, _, _ = te.trace_from_F(F)
    out = {c: {m: [] for m in DEL_METHODS} for c in CASES}
    trace_on_ar1 = {c: [] for c in CASES}          # TRACE restricted to pairs where AR(1) is defined
    worst_ratio = 0.0
    for q in range(T):
        case = "first" if q == 0 else "last" if q == T - 1 else "intermediate"
        keep = np.array([i for i in range(T) if i != q])
        s_, t_ = np.triu_indices(T, 1)
        ok = (s_ != q) & (t_ != q)
        s_, t_ = s_[ok], t_[ok]
        u2 = (T * u - F[q]) / (T - 1)
        d_trace = (u2[t_] - u2[s_]) - psi[s_, t_]
        d_chain = np.zeros((len(s_), d))
        mid = (s_ < q) & (q < t_)
        if mid.any():
            d_chain[mid] = F[q - 1, q + 1] - F[q - 1, q] - F[q, q + 1]
        d_ff = np.zeros((len(s_), d))
        if q == 0:
            d_ff = (F[1, t_] - F[1, s_]) - (F[0, t_] - F[0, s_])
        d_fl = np.zeros((len(s_), d))
        if q == T - 1:
            d_fl = (F[T - 2, t_] - F[T - 2, s_]) - (F[T - 1, t_] - F[T - 1, s_])
        zero = np.zeros((len(s_), d))
        D = {"trace": d_trace, "chain": d_chain, "fixed_first": d_ff, "fixed_last": d_fl,
             "direct": zero, "population": zero, "path_dep": zero}
        for m, arr in D.items():
            out[case][m].extend(zip(np.abs(arr).max(1), np.abs(arr).mean(1)))
        worst_ratio = max(worst_ratio, float(np.abs(d_trace).max() if len(d_trace) else 0.0))
        if ar is not None:
            valid = s_ >= 1                                   # AR(1) undefined at visit 0
            if q == 0:
                valid &= (s_ != 1) & (t_ != 1)                # visit 1 loses its previous visit
            d_ar = np.zeros((len(s_), d))
            if 1 <= q <= T - 2:
                delta = ar["del_phi"][q + 1] - ar["phi"][q + 1]
                d_ar[t_ == q + 1] += delta
                d_ar[s_ == q + 1] -= delta
            out[case]["ar1"].extend(zip(np.abs(d_ar[valid]).max(1), np.abs(d_ar[valid]).mean(1)))
            trace_on_ar1[case].extend(zip(np.abs(d_trace[valid]).max(1), np.abs(d_trace[valid]).mean(1)))
    return out, trace_on_ar1, worst_ratio


def deletion(name):
    pats = [p for p in tr.load_cache(name) if len(p["f"]) >= 4]
    AR = tb.load_ar1(name) if name in tb.PRIMARY else {}
    R = te.tree_output_range(tr.get_model(name))
    per = {c: {m: {"max": [], "mean": []} for m in DEL_METHODS} for c in CASES}
    paired = {c: {m: [] for m in DEL_METHODS} for c in CASES}       # per patient: trace - method (max metric)
    bound = []
    for p in pats:
        T = len(p["f"])
        out, tr_ar, worst = deletion_patient(p["F"], AR.get(p["idx"]))
        bound.append(worst / (4 * R / (T - 1)))
        for c in CASES:
            tmax = np.mean([a for a, _ in out[c]["trace"]]) if out[c]["trace"] else None
            for m in DEL_METHODS:
                v = out[c][m]
                if not v:
                    per[c][m]["max"].append(None); per[c][m]["mean"].append(None); paired[c][m].append(None)
                    continue
                mx = float(np.mean([a for a, _ in v]))
                per[c][m]["max"].append(mx)
                per[c][m]["mean"].append(float(np.mean([b for _, b in v])))
                base = np.mean([a for a, _ in tr_ar[c]]) if m == "ar1" else tmax
                paired[c][m].append(None if base is None else float(base - mx))
    res = {"model": name, "n_patients": len(pats), "R_bound": R,
           "bound_max_ratio": float(np.max(bound)), "bound_violations": int(np.sum(np.array(bound) > 1 + 1e-9))}
    for c in CASES:
        res[c] = {m: dict(max=boot(per[c][m]["max"]), mean=boot(per[c][m]["mean"]),
                          trace_minus_method=boot(paired[c][m])) for m in DEL_METHODS}
    if AR:
        reps = np.concatenate([np.abs(a["rep"] - a["phi"]).max(1)[np.isfinite(a["rep"]).all(1)] for a in AR.values()])
        res["ar1_mc_noise_max_abs"] = dict(mean=float(reps.mean()), p95=float(np.quantile(reps, .95)), n=int(len(reps)))
    return res


# ------------------------------------------------------------------------------- retrain
def _row_spearman(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


def retrain(name):
    pats = tr.load_cache(name)
    models = seed_models(name)
    Xe, Xr = [], []
    for p in pats:                                # every ordered pair of every patient
        X = p["X"]
        T = len(X)
        for r in range(T):
            for t in range(T):
                Xe.append(X[t]); Xr.append(X[r])
    Xe, Xr = np.array(Xe), np.array(Xr)
    X_all = np.vstack([p["X"] for p in pats])
    bg = tb.population_background(name)
    A = {m: [] for m in ["trace", "direct", "chain", "fixed_first", "fixed_last", "population", "path_dep"]}
    for b in models:
        Fall = te.baseline_shapley_batch(b, Xe, Xr)
        pop = te.explainer(b, bg).shap_values(X_all)
        pdv = te.explainer(b, perturbation="tree_path_dependent").shap_values(X_all)
        off = voff = 0
        per_m = {m: [] for m in A}
        for p in pats:
            T = len(p["X"])
            F = Fall[off:off + T * T].reshape(T, T, -1).copy()
            F[np.arange(T), np.arange(T)] = 0.0
            off += T * T
            E = tb.explanations(F, pop[voff:voff + T], pdv[voff:voff + T])
            voff += T
            s, t = np.triu_indices(T, 1)
            for m in per_m:
                per_m[m].append(E[m][s, t])
        for m in A:
            A[m].append(per_m[m])
    res = {"model": name, "n_patients": len(pats), "n_models": len(models)}
    pairs_ab = list(combinations(range(len(models)), 2))
    per = {}
    for m in A:
        per_patient = []
        for pi in range(len(pats)):
            vals = np.array([_row_spearman(A[m][a][pi], A[m][b][pi]) for a, b in pairs_ab])  # (45, n_pairs)
            v = np.nanmean(vals, axis=0) if np.isfinite(vals).any() else np.array([np.nan])
            v = v[np.isfinite(v)]
            per_patient.append(float(v.mean()) if len(v) else None)
        per[m] = per_patient
        res[m] = boot(per_patient)
    # paired: TRACE minus method, per patient (> 0 = TRACE more stable across retraining)
    res["trace_minus_method"] = {
        m: boot([None if a is None or b is None else a - b for a, b in zip(per["trace"], per[m])])
        for m in A if m != "trace"}
    return res


# ------------------------------------------------------------------- never-changing credit
def never_changing(name):
    """Share of each method's credit (over all pairs s < t) given to features whose value is
    identical (NaN-aware) at every visit of the patient. TRACE guarantees 0 (Theorem 1c)."""
    pats = tr.load_cache(name)
    V = tb.load_visit(name)
    res = {}
    for m in ["trace", "chain", "direct", "fixed_first", "fixed_last", "population", "path_dep"]:
        share, absmax = [], []
        for p in pats:
            X = p["X"]
            T = len(X)
            const = np.all(te.nan_equal(X, X[[0]]), axis=0)
            A = tb.explanations(p["F"], V[p["idx"]]["pop"], V[p["idx"]]["pd"])[m]
            s, t = np.triu_indices(T, 1)
            a = np.abs(A[s, t])
            if not const.any() or a.sum() == 0:
                continue
            share.append(float(a[:, const].sum() / a.sum()))
            absmax.append(float(a[:, const].max()))
        res[m] = dict(share=boot(share), max_abs=float(np.max(absmax)), patients=len(share))
    return res


# ------------------------------------------------------------------------------ report
def fmt(b, scale=1.0, nd=3):
    return f"{b['mean'] * scale:.{nd}f} [{b['lo'] * scale:.{nd}f}, {b['hi'] * scale:.{nd}f}]"


def write_md():
    L = ["# TRACE Stage 7: stability", "",
         "Patient means with patient-bootstrap 95% CIs (1,000 draws). Aggregates only. Units: log-odds.", ""]
    for name in MODELS:
        p = OUT / f"deletion_{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        L += [f"## {name}: delete one visit ({r['n_patients']} patients with >= 4 visits)", "",
              f"Bound check (Prop. 4): worst max_j|dpsi| / (4R/(T-1)) = {r['bound_max_ratio']:.4f}; "
              f"violations = {r['bound_violations']} (R = {r['R_bound']:.2f}).", ""]
        if "ar1_mc_noise_max_abs" in r:
            n = r["ar1_mc_noise_max_abs"]
            L.append(f"AR(1) own Monte Carlo noise (same visit, two seeds), max_j |diff|: mean {n['mean']:.4f}, "
                     f"p95 {n['p95']:.4f} (n = {n['n']}).")
            L.append("")
        L += ["| Deleted visit | Method | mean over pairs of max_j abs(dA) | mean over pairs of mean_j abs(dA) | "
              "TRACE minus method (max metric; < 0 = TRACE moves less) |", "|---|---|---|---|---|"]
        for c in CASES:
            for m in DEL_METHODS:
                x = r[c][m]
                if x["max"]["n_patients"] == 0:
                    continue
                L.append(f"| {c} | {m} | {fmt(x['max'])} | {fmt(x['mean'], nd=4)} | {fmt(x['trace_minus_method'])} |")
        L.append("")
    rows = []
    for name in MODELS:
        p = OUT / f"retrain_{name}.json"
        if p.exists():
            r = json.loads(p.read_text())
            for m in ["trace", "direct", "chain", "fixed_first", "fixed_last", "population", "path_dep"]:
                pd_ = r.get("trace_minus_method", {}).get(m)
                rows.append(f"| {name} | {m} | {fmt(r[m])} | {'' if pd_ is None else fmt(pd_)} |")
    if rows:
        L += ["## Retraining stability (10 seed models; Spearman across features, same pair)", "",
              "| Model | Method | Mean Spearman [95% CI] | TRACE minus method, paired (> 0 = TRACE more stable) |",
              "|---|---|---|---|"] + rows
    (OUT / "STAGE7.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = sys.argv[2:] or MODELS
    if what == "never":
        (OUT / "never_changing_credit.json").write_text(json.dumps({n: never_changing(n) for n in names}, indent=2))
        print("never-changing credit done")
        sys.exit(0)
    for n in names:
        if what in ("deletion", "all"):
            (OUT / f"deletion_{n}.json").write_text(json.dumps(deletion(n), indent=2))
            print(f"deletion done: {n}", flush=True)
        if what in ("retrain", "all"):
            (OUT / f"retrain_{n}.json").write_text(json.dumps(retrain(n), indent=2))
            print(f"retrain done: {n}", flush=True)
    write_md()
