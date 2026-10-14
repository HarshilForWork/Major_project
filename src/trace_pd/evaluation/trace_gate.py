"""TRACE Stage 3 gate: exactness checks G1-G4 on real locked-test patients.

Brief: docs/TRACE_BRIEF.md, Stage 3. Tolerances: configs/trace.yaml (never loosened).
Tests: tests/test_trace_gate.py. Results (maxima and counts only, no patient data):
results/gate/<model>.json and results/gate/GATE.md.

G4 is the independent check: it never calls shap. For a sample of visit pairs with at most
12 differing features it builds every 2^k hybrid visit, scores them with
booster.predict(output_margin=True), and computes the exact Shapley values directly.

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_gate [model ...]
"""
import json
import sys
import time
from math import factorial

import numpy as np

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.explain import trace_engine as te

TOL = tr.CFG["tolerances"]
GATE_DIR = config.ROOT / "results" / "gate"
GATE_MODELS = ["ppmi", "adni_minus", "adni_minus_lm", "adni_minus2", "adni_full"]


# ---------------------------------------------------------------------------- G1-G3
def exactness(name):
    pats = tr.load_cache(name)
    b = tr.get_model(name)
    used = te.used_feature_mask(b, len(tr.features(name)))
    g1_pair = g1_traj = g1_pair32 = g1_traj32 = xgb32_vs_exact = 0.0
    g2_tri = g2_lsq = g2_bg = 0.0
    g3_const = g3_unused = 0.0
    n_const = 0
    diag = 0.0
    for p in pats:
        F, f32, X = p["F"], p["f"], p["X"]
        f = te.exact_margin(b, X)          # official G1 reference (user decision, 14 Oct 2026)
        u, psi, _, _ = te.trace_from_F(F)
        # G1: psi adds up for every ordered pair; trajectory decomposition at every visit.
        # Official: against the exact float64 tree sum. Also reported: against XGBoost's
        # float32 predict (results/gate/G1_diagnosis.md).
        g1_pair = max(g1_pair, np.abs(psi.sum(2) - (f[None, :] - f[:, None])).max())
        g1_traj = max(g1_traj, np.abs(f - f.mean() - u.sum(1)).max())
        g1_pair32 = max(g1_pair32, np.abs(psi.sum(2) - (f32[None, :] - f32[:, None])).max())
        g1_traj32 = max(g1_traj32, np.abs(f32 - f32.mean() - u.sum(1)).max())
        xgb32_vs_exact = max(xgb32_vs_exact, np.abs(f32 - f).max())
        # G2: every triple s,t,q; closed form vs least squares vs background route
        tri = psi[:, :, None, :] + psi[None, :, :, :] - psi[:, None, :, :]   # [s,t,q] = psi(s,t)+psi(t,q)-psi(s,q)
        g2_tri = max(g2_tri, np.abs(tri).max())
        g2_lsq = max(g2_lsq, np.abs(u - te.trace_via_lstsq(F)).max())
        g2_bg = max(g2_bg, np.abs(u - p["u_bg"]).max())
        # G3a: NaN-aware constant across all visits -> u == 0
        const = np.all(te.nan_equal(X, X[[0]]), axis=0)
        n_const += int(const.sum())
        if const.any():
            g3_const = max(g3_const, np.abs(u[:, const]).max())
        if (~used).any():
            g3_unused = max(g3_unused, np.abs(u[:, ~used]).max())
        diag = max(diag, float(p["diag_max"]))
    return dict(
        n_patients=len(pats), n_visits=int(sum(len(p["f"]) for p in pats)),
        G1_pair_max=float(g1_pair), G1_trajectory_max=float(g1_traj),
        G1_pair_max_xgb_float32=float(g1_pair32), G1_trajectory_max_xgb_float32=float(g1_traj32),
        xgb_float32_predict_vs_exact_max=float(xgb32_vs_exact),
        G2_triple_max=float(g2_tri), G2_closed_vs_lstsq_max=float(g2_lsq),
        G2_closed_vs_background_max=float(g2_bg),
        G3_constant_feature_max=float(g3_const), G3_constant_feature_cases=n_const,
        G3_unused_feature_max=float(g3_unused), G3_unused_features=int((~used).sum()),
        F_diagonal_from_treeshap_max=diag,
    )


# --------------------------------------------------------------------------------- G4
def _nan_flags(xr, xt):
    nr, nt = np.isnan(xr), np.isnan(xt)
    return dict(nan_ref=bool((nr & ~nt).any()), nan_expl=bool((~nr & nt).any()),
                nan_both=bool((nr & nt).any()), no_nan=bool(~(nr | nt).any()))


def sample_pairs(pats, rng, min_total, per_cat=500, max_k=12):
    """Stratified sample of (patient position, r, t), r < t, with 1 <= k <= max_k differing
    features: up to `per_cat` pairs from each NaN category (NaN only in the reference,
    only in the explicand, NaN in both at some feature, no NaN), then random fill-up to
    `min_total`. A pair may belong to several categories."""
    elig, flags = [], []
    for pi, p in enumerate(pats):
        X = p["X"]
        T = len(X)
        for r in range(T):
            for t in range(r + 1, T):
                k = int((~te.nan_equal(X[r], X[t])).sum())
                if 1 <= k <= max_k:
                    elig.append((pi, r, t))
                    flags.append(_nan_flags(X[r], X[t]))
    chosen = set()
    avail = {}
    for cat in ["nan_ref", "nan_expl", "nan_both", "no_nan"]:
        idx = [i for i, fl in enumerate(flags) if fl[cat]]
        avail[cat] = len(idx)
        for i in rng.permutation(idx)[:per_cat]:
            chosen.add(int(i))
    rest = [i for i in rng.permutation(len(elig)) if int(i) not in chosen]
    for i in rest[:max(0, min_total - len(chosen))]:
        chosen.add(int(i))
    chosen = sorted(chosen)
    counts = {cat: sum(flags[i][cat] for i in chosen) for cat in avail}
    return [elig[i] for i in chosen], counts, avail, len(elig)


def brute_force_shapley(booster, xr, xt):
    """Exact Shapley values of v(S) = f(x^t on S, x^r elsewhere) - f(x^r), players = the
    differing features (NaN-aware). Returns a full-length vector (0 for non-players)."""
    D = np.flatnonzero(~te.nan_equal(xr, xt))
    k = len(D)
    masks = np.arange(2 ** k)
    bits = ((masks[:, None] >> np.arange(k)) & 1).astype(bool)          # (2^k, k)
    H = np.repeat(xr[None, :], 2 ** k, axis=0)
    H[:, D] = np.where(bits, xt[D][None, :], xr[D][None, :])
    v = te.margin(booster, H)
    v = v - v[0]
    size = bits.sum(1)
    w = np.array([factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)])
    phi = np.zeros(len(xr))
    for i in range(k):
        without = ~bits[:, i]
        m0 = masks[without]
        phi[D[i]] = float((w[size[without]] * (v[m0 | (1 << i)] - v[m0])).sum())
    return phi


def brute_force(name, seed):
    pats = tr.load_cache(name)
    b = tr.get_model(name)
    rng = np.random.default_rng(seed)
    max_k = tr.CFG["g4"].get("max_k_overrides", {}).get(name, tr.CFG["g4"]["max_k"])
    pairs, counts, avail, n_elig = sample_pairs(pats, rng, tr.CFG["g4"]["min_pairs_per_cohort"], max_k=max_k)
    err = {c: 0.0 for c in counts}
    err_all = anti = 0.0
    ks = []
    for pi, r, t in pairs:
        X, F = pats[pi]["X"], pats[pi]["F"]
        exact = brute_force_shapley(b, X[r], X[t])
        e = float(np.abs(exact - F[r, t]).max())
        err_all = max(err_all, e)
        for cat, on in _nan_flags(X[r], X[t]).items():
            if on:
                err[cat] = max(err[cat], e)
        anti = max(anti, float(np.abs(F[r, t] + F[t, r]).max()))
        ks.append(int((~te.nan_equal(X[r], X[t])).sum()))
    return dict(G4_max_k=max_k, G4_pairs=len(pairs), G4_eligible_pairs=n_elig, G4_pairs_by_nan_category=counts,
                G4_available_by_nan_category=avail, G4_k_median=float(np.median(ks)), G4_k_max=int(max(ks)),
                G4_max=err_all, G4_max_by_nan_category=err, G4_antisymmetry_max=anti)


# ---------------------------------------------------------------- pilot cross-check
def _pilot_module():
    """notebooks/trace_pilot.py: the pilot's own exact tree-Shapley code (leaf-path
    enumeration with closed-form unanimity-game weights). Independent of shap and of the
    2^k brute force, and has no limit on the number of differing features k."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("trace_pilot", config.ROOT / "notebooks" / "trace_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _AsClassifier:                       # the pilot reads trees via model.get_booster()
    def __init__(self, booster):
        self._b = booster

    def get_booster(self):
        return self._b


def pilot_crosscheck(name, seed, n_pairs=2000):
    """Compare F[r, t] with the pilot's exact algorithm on a random sample of ordered pairs
    (r != t, any k). Supplementary check added at the user's request (14 Oct 2026)."""
    pilot = _pilot_module()
    pats = tr.load_cache(name)
    b = tr.get_model(name)
    leaves = pilot.leaf_paths(_AsClassifier(b), tr.features(name))
    rng = np.random.default_rng(seed + 1)
    allp = [(pi, r, t) for pi, p in enumerate(pats) for r in range(len(p["f"])) for t in range(len(p["f"])) if r != t]
    pick = [allp[i] for i in rng.choice(len(allp), size=min(n_pairs, len(allp)), replace=False)]
    Xs = np.array([pats[pi]["X"][t] for pi, r, t in pick])
    Rs = np.array([pats[pi]["X"][r] for pi, r, t in pick])
    phi = pilot.baseline_shapley(leaves, Xs, Rs, Xs.shape[1])
    Fs = np.array([pats[pi]["F"][r, t] for pi, r, t in pick])
    ks = (~te.nan_equal(Xs, Rs)).sum(1)
    return dict(pilot_pairs=len(pick), pilot_k_median=float(np.median(ks)), pilot_k_max=int(ks.max()),
                pilot_max=float(np.abs(phi - Fs).max()))


# ------------------------------------------------------------------------------ run
def run_gate(name):
    t0 = time.time()
    res = {"model": name, "shap_version": __import__("shap").__version__,
           "xgboost_version": __import__("xgboost").__version__}
    res |= exactness(name)
    res |= brute_force(name, seed=tr.CFG["seeds"]["global"])
    res |= pilot_crosscheck(name, seed=tr.CFG["seeds"]["global"])
    checks = {
        "G1": max(res["G1_pair_max"], res["G1_trajectory_max"]) <= TOL["add_up"],
        "G2": max(res["G2_triple_max"], res["G2_closed_vs_lstsq_max"],
                  res["G2_closed_vs_background_max"]) <= TOL["consistency"],
        "G3": max(res["G3_constant_feature_max"], res["G3_unused_feature_max"]) <= TOL["zero"],
        "G4": max(res["G4_max"], res["G4_antisymmetry_max"]) <= TOL["brute_force"]
              and res["G4_pairs"] >= tr.CFG["g4"]["min_pairs_per_cohort"],
        "pilot": res["pilot_max"] <= TOL["brute_force"],      # supplementary, same tolerance as G4
    }
    res["checks"] = {k: ("PASS" if v else "FAIL") for k, v in checks.items()}
    res["seconds"] = round(time.time() - t0, 1)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res


def write_table(results):
    rows = ["| Model | G1 add-up vs exact float64 (pair / trajectory) [vs XGBoost float32] | "
            "G2 consistency (triples / lstsq / background) | "
            "G3 zero (constant / unused) | G4 brute force (max / antisymmetry, pairs) | Pilot code (max, pairs) | Result |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        unused = ("n/a, every feature is used" if r["G3_unused_features"] == 0
                  else f"{r['G3_unused_feature_max']:.1e} ({r['G3_unused_features']} features)")
        rows.append(
            f"| {r['model']} | {r['G1_pair_max']:.1e} / {r['G1_trajectory_max']:.1e} "
            f"[{r['G1_pair_max_xgb_float32']:.1e} / {r['G1_trajectory_max_xgb_float32']:.1e}] | "
            f"{r['G2_triple_max']:.1e} / {r['G2_closed_vs_lstsq_max']:.1e} / {r['G2_closed_vs_background_max']:.1e} | "
            f"{r['G3_constant_feature_max']:.1e} ({r['G3_constant_feature_cases']} cases) / {unused} | "
            f"{r['G4_max']:.1e} / {r['G4_antisymmetry_max']:.1e}, {r['G4_pairs']} (k <= {r['G4_max_k']}) | "
            f"{r['pilot_max']:.1e}, {r['pilot_pairs']} | "
            f"{' '.join(f'{k} {v}' for k, v in r['checks'].items())} |")
    (GATE_DIR / "GATE.md").write_text(
        "# TRACE Stage 3 gate\n\nTolerances: G1, G2, G4 <= 1e-5; G3 <= 1e-6 (configs/trace.yaml).\n"
        "G1 is measured against the exact float64 tree sum (user decision, 14 Oct 2026); the number in\n"
        "brackets is against XGBoost's float32 predict (results/gate/G1_diagnosis.md). G4 uses k <= 13\n"
        "for PPMI (user decision: only 1,488 PPMI pairs have k <= 12) and k <= 12 otherwise.\n\n"
        + "\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    names = sys.argv[1:] or GATE_MODELS
    out = []
    for n in names:
        r = run_gate(n)
        out.append(r)
        print(json.dumps(r, indent=2))
    write_table(out)
