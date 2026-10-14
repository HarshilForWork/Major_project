"""TRACE Stage 5: baseline change explanations A(s->t) (docs/TRACE_BRIEF.md, Stage 5).

Methods (all return a vector over features for a pair of visits s < t of one patient):
    trace          psi(s,t) = u(t) - u(s)
    direct         F[s,t]                         (Baseline Shapley; upper reference on its own pair)
    chain          sum_{k=s}^{t-1} F[k,k+1]       ("DeltaSHAP-style")
    fixed_first    F[0,t] - F[0,s]                (simplest consistent fix)
    fixed_last     F[T-1,t] - F[T-1,s]
    population     phi_pop(t) - phi_pop(s)        interventional TreeSHAP, fixed background of
                                                  n_background_population TRAINING visits (seed fixed)
    path_dep       phi_pd(t) - phi_pd(s)          shap default (tree_path_dependent) TreeSHAP
    ar1            phi_ar1(t) - phi_ar1(s)        the team's conditional AR(1)-SHAP (explain/ar1.py,
                                                  explain/conditional_shapley.py), 48 permutations x 24
                                                  draws, transition model fitted on the development split.
                                                  Undefined for a patient's first visit (no previous visit),
                                                  so pairs with s = 0 are excluded for this method only.
Asymmetric Shapley: skipped (no implementation in the repo; user decision). FA-SHAP: no
forecasts in the repo, skipped. DynSHAP / C-SHAP: out of scope per the brief.

Per-visit attributions for population / path_dep are cached in results/cache/<model>_visit/;
AR(1) in results/cache/<model>_ar1/ together with the Stage 7 deletion variants:
    del_phi[k] = AR(1) attribution of visit k when visit k-1 is deleted (k >= 2): the only
    AR(1) quantity a single deletion changes.
AR(1) is computed for the PRIMARY models only (ppmi, adni_minus): about 1 s per visit (deviation).
Common random numbers: the AR(1) RNG for (patient, visit) is the same in the base and deletion runs.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_baselines visit <model>
      PYTHONPATH=src python -m trace_pd.evaluation.trace_baselines ar1 <model> <chunk> <n_chunks>
      PYTHONPATH=src python -m trace_pd.evaluation.trace_baselines stage5
"""
import json
import os
import sys
import warnings

import numpy as np

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.explain import trace_engine as te

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = config.ROOT / "results" / "stage5"
AR1_PERM, AR1_MC = 48, 24
PRIMARY = ["ppmi", "adni_minus"]
METHODS = ["trace", "direct", "chain", "fixed_first", "fixed_last", "population", "path_dep", "ar1"]


# --------------------------------------------------------------- per-visit attributions
def population_background(name, seed=None):
    """Fixed random sample of the model's TRAINING visits (development, labelled pos/neg)."""
    spec = tr.MODELS[name]
    c = tr.COHORT[spec["cohort"]]
    dev, _, _ = tr.load_cohort(spec["cohort"])
    s = dev[dev.LABEL.isin([c["neg"], c["pos"]])][tr.features(name)].to_numpy(float)
    rng = np.random.default_rng(tr.CFG["seeds"]["global"] if seed is None else seed)
    return s[rng.choice(len(s), size=min(tr.CFG["n_background_population"], len(s)), replace=False)]


def visit_attributions(booster, name, X_all):
    """(phi_pop, phi_pd) for the stacked visits X_all, via the verified shap model copy."""
    pop = te.explainer(booster, population_background(name)).shap_values(X_all)
    pd_ = te.explainer(booster, perturbation="tree_path_dependent").shap_values(X_all)
    return pop, pd_


def compute_visit(name, visits="all"):
    b = tr.get_model(name)
    pats = tr.load_cache(name, visits)
    X_all = np.vstack([p["X"] for p in pats])
    pop, pd_ = visit_attributions(b, name, X_all)
    d = tr.cache_dir(f"{name}_visit", visits)
    off = 0
    for p in pats:
        T = len(p["X"])
        np.savez(d / f"p{p['idx']:04d}.npz", pop=pop[off:off + T], pd=pd_[off:off + T])
        off += T
    print(f"{name} [{visits}]: population / path-dependent attributions cached ({len(X_all)} visits)")


def load_visit(name, visits="all"):
    d = tr.cache_dir(f"{name}_visit", visits)
    return {int(p.stem[1:]): dict(np.load(p)) for p in d.glob("p*.npz")}


# ------------------------------------------------------------------------- AR(1)-SHAP
def _ar1_setup(name):
    from trace_pd.explain import ar1
    from trace_pd.explain.conditional_shapley import AR1Filler

    coh = tr.MODELS[name]["cohort"]
    idc = tr.COHORT[coh]["id"]
    dev, _, _ = tr.load_cohort(coh)
    F = tr.features(name)
    tm = ar1.TransitionModel(F, "full_dt", id_col=idc).fit(ar1.build_pairs(dev, F, id_col=idc), dev)
    return AR1Filler(tm, F), F, idc


def compute_ar1(name, chunk=0, n_chunks=1):
    """AR(1) attributions for every visit with a previous visit, plus deletion variants and a
    second-seed replicate of up to 3 visits per patient (to measure AR(1)'s own MC noise)."""
    from trace_pd.explain import ar1
    from trace_pd.explain.conditional_shapley import shapley_instance

    fl, F, idc = _ar1_setup(name)
    b = tr.get_model(name)
    f = lambda Z: te.margin(b, Z)
    d = tr.cache_dir(f"{name}_ar1")
    pats = tr.analysis_patients(tr.MODELS[name]["cohort"])
    g0 = tr.CFG["seeds"]["global"]

    def attr(row_frame, k_seed, i, k):
        x = row_frame[F].to_numpy(float)[0]
        fl.for_instance(x, row_frame)
        return shapley_instance(f, x, fl, AR1_PERM, AR1_MC, np.random.default_rng([g0, i, k, k_seed]))

    for n, (i, g) in enumerate(pats):
        if n % n_chunks != chunk:
            continue
        out = d / f"p{i:04d}.npz"
        if out.exists():
            continue
        T = len(g)
        d_ = len(F)
        phi = np.full((T, d_), np.nan); base = np.full(T, np.nan)
        del_phi = np.full((T, d_), np.nan); del_base = np.full(T, np.nan)
        rep = np.full((T, d_), np.nan)
        P = ar1.build_pairs(g, F, id_col=idc)              # rows = visits 1..T-1, in order
        assert len(P) == T - 1
        Xg = g[F].to_numpy(float)
        assert np.all(te.nan_equal(P[F].to_numpy(float), Xg[1:])), "pair rows out of visit order"
        for k in range(1, T):
            phi[k], base[k] = attr(P.iloc[[k - 1]], 0, i, k)
            if k <= 3:
                rep[k], _ = attr(P.iloc[[k - 1]], 1, i, k)  # replicate with another seed
        for k in range(2, T):                               # delete visit k-1
            Pq = ar1.build_pairs(g.drop(index=k - 1).reset_index(drop=True), F, id_col=idc)
            assert np.all(te.nan_equal(Pq.iloc[k - 2][F].to_numpy(float), Xg[k])), "deletion row mismatch"
            del_phi[k], del_base[k] = attr(Pq.iloc[[k - 2]], 0, i, k)   # same RNG stream as base
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, phi=phi, base=base, del_phi=del_phi, del_base=del_base, rep=rep)
        os.replace(tmp, out)
    print(f"{name}: AR(1) chunk {chunk}/{n_chunks} done", flush=True)


def load_ar1(name):
    d = tr.cache_dir(f"{name}_ar1")
    return {int(p.stem[1:]): dict(np.load(p)) for p in d.glob("p*.npz") if "tmp" not in p.name}


# ------------------------------------------------------------------- change explanations
def explanations(F, pop=None, pd_=None, ar1=None):
    """dict method -> array A[s, t, d] for s < t (other entries unused)."""
    T = F.shape[0]
    u, psi, _, _ = te.trace_from_F(F)
    steps = np.stack([F[k, k + 1] for k in range(T - 1)])
    C = np.vstack([np.zeros((1, F.shape[2])), np.cumsum(steps, 0)])
    A = {"trace": psi, "direct": F, "chain": C[None, :, :] - C[:, None, :],
         "fixed_first": F[0][None, :, :] - F[0][:, None, :],
         "fixed_last": F[T - 1][None, :, :] - F[T - 1][:, None, :]}
    if pop is not None:
        A["population"] = pop[None, :, :] - pop[:, None, :]
    if pd_ is not None:
        A["path_dep"] = pd_[None, :, :] - pd_[:, None, :]
    if ar1 is not None:
        A["ar1"] = ar1[None, :, :] - ar1[:, None, :]       # NaN rows/cols for visit 0
    return A


# ------------------------------------------------------------------------ Stage 5 report
def stage5(name):
    b = tr.get_model(name)
    pats = tr.load_cache(name)
    V = load_visit(name)
    AR = load_ar1(name) if name in PRIMARY else {}
    gaps = {m: [] for m in METHODS}
    gaps_mean = {m: [] for m in METHODS}
    n_pairs = {m: 0 for m in METHODS}
    for p in pats:
        F, X = p["F"], p["X"]
        f = te.exact_margin(b, X)
        T = len(f)
        ar = AR.get(p["idx"])
        A = explanations(F, V[p["idx"]]["pop"], V[p["idx"]]["pd"], None if ar is None else ar["phi"])
        s, t = np.triu_indices(T, 1)
        df = f[t] - f[s]
        for m, arr in A.items():
            e = np.abs(arr[s, t].sum(1) - df)
            ok = np.isfinite(e)
            if ok.any():
                gaps[m].append(float(e[ok].max()))
                gaps_mean[m].append(float(e[ok].mean()))
                n_pairs[m] += int(ok.sum())
    res = {m: dict(n_pairs=n_pairs[m], max=float(np.max(gaps[m])) if gaps[m] else None,
                   mean_of_patient_means=float(np.mean(gaps_mean[m])) if gaps_mean[m] else None)
           for m in METHODS}
    if AR:
        reps = [np.abs(a["rep"] - a["phi"])[np.isfinite(a["rep"])] for a in AR.values()]
        reps = np.concatenate(reps)
        res["ar1_mc_noise"] = dict(n_values=int(len(reps)), mean_abs=float(reps.mean()), p95_abs=float(np.quantile(reps, .95)))
    return res


def write_stage5(results):
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# TRACE Stage 5: baselines and their efficiency gap", "",
         "Efficiency gap = |sum_j A_j(s->t) - (f(x^t) - f(x^s))|, with f the exact float64 margin, over all "
         "pairs s < t of the analysed locked-test patients. Aggregates only.", "",
         "| Model | Method | Pairs | Max gap | Mean gap (patient means) |", "|---|---|---|---|---|"]
    for name, r in results.items():
        for m in METHODS:
            x = r[m]
            if x["n_pairs"]:
                L.append(f"| {name} | {m} | {x['n_pairs']} | {x['max']:.2e} | {x['mean_of_patient_means']:.2e} |")
            else:
                L.append(f"| {name} | {m} | 0 | not computed | |")
        if "ar1_mc_noise" in r:
            n = r["ar1_mc_noise"]
            L.append(f"| {name} | ar1 Monte Carlo noise (same visit, two seeds) | {n['n_values']} values | "
                     f"p95 {n['p95_abs']:.3f} | mean {n['mean_abs']:.4f} |")
    (OUT / "STAGE5.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "stage5.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "visit":
        for spec in sys.argv[2:]:
            n, _, v = spec.partition(":")
            compute_visit(n, v or "all")
    elif cmd == "ar1":
        compute_ar1(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "stage5":
        res = {n: stage5(n) for n in (sys.argv[2:] or PRIMARY)}
        write_stage5(res)
        print((OUT / "STAGE5.md").read_text(encoding="utf-8"))
