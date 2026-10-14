"""TRACE Stage 4: size of the problem (brief metrics 5 and 6). docs/TRACE_BRIEF.md.

Runs (configs/trace.yaml stage_plan + the labelled-only sensitivity):
    ppmi, adni_minus (primary); adni_minus_lm, adni_minus2 (sensitivity);
    ppmi:labelled, adni_minus:labelled (labelled visits only)

Definitions (reused EXACTLY from the pilot, notebooks/trace_pilot.py, where it defines them):
    direct(s,t) = F[s,t]; chain(s,t) = sum_{k=s}^{t-1} F[k,k+1]; top = argmax |.| (ties -> lowest
    feature index); a vector with max |.| <= 1e-9 has no top cause and is skipped.
    top-cause disagreement  top(chain) != top(direct)
    clear winner            direct's 2nd largest |credit| < 0.8 x its largest
    total disagreement      sum_j |chain_j - direct_j| / sum_j |direct_j|
    TRACE change rate       top(psi(s,t)) != top(direct(s,t)); pilot version over pairs r<t with
                            max_j |F[r,t,j]| > 0.05 ("real change")
    locality price          unchanged feature (NaN-aware) with |psi_j| > 10% of sum_j |psi_j|
    trust score I           sum_{r<t} R^2 / sum_{r<t} F^2 (patients with sum F^2 = 0 skipped)
Brief additions: the same metrics over ALL pairs (s,t) with at least one visit in between
(t >= s+2), averaged within patient first, then over patients; patient-level bootstrap
95% CIs (n_bootstrap in configs/trace.yaml).

Seed-noise baseline: 10 models retrained on the same development split with seeds 0-9
(subsample/colsample 0.8, so the seed matters; checked). Noise rate = how often the top
feature of direct(s,t) differs between two retrained models, averaged over the 45 model
pairs. Explanations of retrained models use trace_engine.baseline_shapley_batch (exact; equals
the shap route to <= 4.3e-8 on every pair of every patient).

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_stage4
Out:  results/stage4/  (aggregates only; no patient identifiers)
"""
import json
import sys
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.explain import trace_engine as te
from trace_pd.models.backend import make_classifier, fit_classifier

OUT = config.ROOT / "results" / "stage4"
SEED_DIR = config.ROOT / "models" / "trace" / "seeds"
RUNS = [("ppmi", "all"), ("adni_minus", "all"), ("adni_minus_lm", "all"), ("adni_minus2", "all"),
        ("ppmi", "labelled"), ("adni_minus", "labelled")]
PILOT = {   # pilot numbers (scratchpad run, 12 Oct 2026) for the reproduction check
    "ppmi": dict(top_differs=0.307, clear_top_differs=0.114, rel_gap_median=0.518, sign_flip=0.0,
                 trace_change_real=0.173, trace_change_real_n=10997, locality_10pct=0.001, I_median=0.030),
    "adni_minus": dict(top_differs=0.229, clear_top_differs=0.088, rel_gap_median=0.386, sign_flip=0.0,
                       trace_change_real=0.077, trace_change_real_n=24099, locality_10pct=0.012, I_median=0.015),
}
EPS_TOP = 1e-9


# ---------------------------------------------------------------------------- helpers
def top(v):
    return int(np.argmax(np.abs(v))) if np.abs(v).max() > EPS_TOP else -1


def boot(per_patient, n=None, seed=None):
    """Mean over patients with a patient-level percentile bootstrap 95% CI."""
    v = np.asarray([x for x in per_patient if x is not None and np.isfinite(x)], float)
    n = n or tr.CFG["n_bootstrap"]
    rng = np.random.default_rng(tr.CFG["seeds"]["global"] if seed is None else seed)
    if len(v) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_patients=0)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(v.mean()), lo=float(lo), hi=float(hi), n_patients=int(len(v)))


def boot_spearman(a, b, n=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(tr.CFG["seeds"]["global"])
    n = n or tr.CFG["n_bootstrap"]
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        r = spearmanr(a[i], b[i]).correlation
        if np.isfinite(r):
            rs.append(r)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return dict(rho=float(spearmanr(a, b).correlation), lo=float(lo), hi=float(hi), n_patients=int(len(a)))


def chain_of(F):
    """chain[s, t] = sum_{k=s}^{t-1} F[k, k+1] for s < t (via cumulative sums)."""
    T = F.shape[0]
    steps = np.stack([F[k, k + 1] for k in range(T - 1)])
    C = np.vstack([np.zeros((1, F.shape[2])), np.cumsum(steps, 0)])
    return C[None, :, :] - C[:, None, :]


# --------------------------------------------------------------------- per patient
def patient_metrics(F, X):
    T = F.shape[0]
    u, psi, R, I = te.trace_from_F(F)
    ch = chain_of(F)
    m = {"T": T, "I": I}
    # pilot first->last
    direct, chain = F[0, T - 1], ch[0, T - 1]
    jd, jc = top(direct), top(chain)
    if jd >= 0:
        a = np.sort(np.abs(direct))[::-1]
        m["fl_top_differs"] = float(jd != jc)
        m["fl_clear_top_differs"] = float(jd != jc and a[1] < 0.8 * a[0])
        m["fl_clear"] = float(a[1] < 0.8 * a[0])
        m["fl_sign_flip"] = float(np.sign(direct[jd]) != np.sign(chain[jd]))
        m["fl_rel_gap"] = float(np.abs(direct - chain).sum() / np.abs(direct).sum())
        m["fl_trace_change"] = float(top(psi[0, T - 1]) != jd)
    # all pairs with a visit in between (t >= s+2)
    td, rg, trc, trc_all = [], [], [], []
    for s in range(T):
        for t in range(s + 1, T):
            d = F[s, t]
            jd = top(d)
            if jd < 0:
                continue
            trc_all.append(top(psi[s, t]) != jd)
            if t >= s + 2:
                td.append(top(ch[s, t]) != jd)
                rg.append(np.abs(d - ch[s, t]).sum() / np.abs(d).sum())
                trc.append(top(psi[s, t]) != jd)
    m["gap_top_differs"] = float(np.mean(td)) if td else None
    m["gap_rel_gap"] = float(np.mean(rg)) if rg else None
    m["gap_trace_change"] = float(np.mean(trc)) if trc else None
    m["all_trace_change"] = float(np.mean(trc_all)) if trc_all else None
    # pilot pair-level counts (pooled later): real-change pairs, locality price
    iu = np.triu_indices(T, 1)
    Fe, Pe = F[iu], psi[iu]
    big = np.abs(Fe).max(1) > 0.05
    m["_real_change"] = (np.abs(Fe[big]).argmax(1) != np.abs(Pe[big]).argmax(1)).tolist()
    same = te.nan_equal(X[iu[0]], X[iu[1]])
    share = np.abs(Pe) / (np.abs(Pe).sum(1, keepdims=True) + 1e-12)
    m["_loc_share"] = share[same].tolist()
    m["_loc_abs"] = np.abs(Pe)[same].tolist()
    return m


# ------------------------------------------------------------------------ seed noise
def seed_models(name):
    """10 retrained models (seeds 0-9), same data and recipe; saved under models/trace/seeds."""
    from xgboost import Booster

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    spec = tr.MODELS[name]
    c = tr.COHORT[spec["cohort"]]
    F_cols = tr.features(name)
    out = []
    for sd in tr.CFG["seeds"]["retrain_seeds"]:
        p = SEED_DIR / f"{name}_s{sd}.json"
        if not p.exists():
            dev, _, _ = tr.load_cohort(spec["cohort"])
            s = dev[dev.LABEL.isin([c["neg"], c["pos"]])]
            clf = make_classifier("xgb", 2)
            clf.set_params(random_state=sd)
            fit_classifier(clf, s[F_cols].to_numpy(float), (s.LABEL == c["pos"]).astype(int))
            clf.get_booster().save_model(p)
        b = Booster()
        b.load_model(p)
        out.append(b)
    return out


def seed_noise(name, visits, pats):
    models = seed_models(name)
    # does the seed change the model? (subsample/colsample 0.8 -> it should)
    Xall = np.vstack([p["X"] for p in pats])
    margins = np.array([te.exact_margin(b, Xall) for b in models])
    seed_effect = float(np.abs(margins - margins[0]).max())
    # one batched call per model over every patient's pairs: (0, T-1) first, then all t >= s+2
    Xe, Xr, owner, is_fl = [], [], [], []
    for pi, p in enumerate(pats):
        X = p["X"]
        T = len(X)
        for s, t in [(0, T - 1)] + [(s, t) for s in range(T) for t in range(s + 2, T)]:
            Xe.append(X[t]); Xr.append(X[s]); owner.append(pi); is_fl.append(len(owner) == 1 or owner[-2] != pi)
    Xe, Xr, owner, is_fl = np.array(Xe), np.array(Xr), np.array(owner), np.array(is_fl)
    tops = np.empty((len(models), len(Xe)), int)
    for m, b in enumerate(models):
        phi = te.baseline_shapley_batch(b, Xe, Xr)
        mx = np.abs(phi).max(1)
        tops[m] = np.where(mx > EPS_TOP, np.abs(phi).argmax(1), -1)
    pairs_ab = list(combinations(range(len(models)), 2))
    a_idx, b_idx = np.array([a for a, _ in pairs_ab]), np.array([b for _, b in pairs_ab])
    valid = (tops[a_idx] >= 0) & (tops[b_idx] >= 0)                     # (45, n_rows)
    diff = (tops[a_idx] != tops[b_idx]) & valid
    with np.errstate(invalid="ignore"):
        row_rate = np.where(valid.sum(0) > 0, diff.sum(0) / np.maximum(valid.sum(0), 1), np.nan)
    fl_noise, gap_noise = [], []
    for pi in range(len(pats)):
        rows = owner == pi
        fl = row_rate[rows & is_fl]
        fl_noise.append(float(fl[0]) if len(fl) and np.isfinite(fl[0]) else None)
        g = row_rate[rows & ~is_fl]
        g = g[np.isfinite(g)]
        gap_noise.append(float(g.mean()) if len(g) else None)
    return dict(seed_effect_max_margin_diff=seed_effect, n_models=len(models)), fl_noise, gap_noise


# ------------------------------------------------------------------------------ run
def run(name, visits):
    pats = tr.load_cache(name, visits)
    M = [patient_metrics(p["F"], p["X"]) for p in pats]
    col = lambda k: [m.get(k) for m in M]
    res = {"model": name, "visits": visits, "n_patients": len(M),
           "visits_median": float(np.median(col("T")))}
    for k in ["fl_top_differs", "fl_clear_top_differs", "fl_sign_flip", "fl_rel_gap", "fl_trace_change",
              "gap_top_differs", "gap_rel_gap", "gap_trace_change", "all_trace_change"]:
        res[k] = boot(col(k))
    rg = np.array([x for x in col("fl_rel_gap") if x is not None])
    res["fl_rel_gap_median"] = float(np.median(rg))
    real = [x for m in M for x in m["_real_change"]]
    loc_share = np.array([x for m in M for x in m["_loc_share"]])
    loc_abs = np.array([x for m in M for x in m["_loc_abs"]])
    res["pilot_trace_change_real"] = dict(rate=float(np.mean(real)), n_pairs=len(real))
    res["locality"] = dict(n_cases=int(len(loc_share)), share_gt_10pct=float(np.mean(loc_share > 0.10)),
                           share_gt_5pct=float(np.mean(loc_share > 0.05)), abs_gt_0_05=float(np.mean(loc_abs > 0.05)),
                           abs_p99=float(np.quantile(loc_abs, 0.99)), abs_max=float(loc_abs.max()))
    I = np.array([x for x in col("I") if np.isfinite(x)])
    T = np.array(col("T"))
    res["I"] = dict(median=float(np.median(I)), q25=float(np.quantile(I, .25)), q75=float(np.quantile(I, .75)),
                    q90=float(np.quantile(I, .90)), min=float(I.min()), max=float(I.max()),
                    by_T={f"{lo}-{hi}": dict(n=int(((T >= lo) & (T <= hi)).sum()),
                                             median=float(np.median(np.array(col("I"))[(T >= lo) & (T <= hi)])))
                          for lo, hi in [(3, 4), (5, 9), (10, 14), (15, 99)] if ((T >= lo) & (T <= hi)).sum()})
    res["I_vs_disagreement"] = dict(
        first_last=boot_spearman(col("I"), [np.nan if x is None else x for x in col("fl_rel_gap")]),
        gap_pairs=boot_spearman(col("I"), [np.nan if x is None else x for x in col("gap_rel_gap")]))
    info, fl_noise, gap_noise = seed_noise(name, visits, pats)
    res["seed_noise"] = info | dict(first_last=boot(fl_noise), gap_pairs=boot(gap_noise))
    # paired: chain-vs-direct disagreement minus seed noise, per patient
    d_fl = [None if a is None or b is None else a - b for a, b in zip(col("fl_top_differs"), fl_noise)]
    d_gap = [None if a is None or b is None else a - b for a, b in zip(col("gap_top_differs"), gap_noise)]
    res["chain_minus_noise"] = dict(first_last=boot(d_fl), gap_pairs=boot(d_gap))
    if visits == "all" and name in PILOT:
        res["pilot_reproduction"] = {
            "top_differs": (res["fl_top_differs"]["mean"], PILOT[name]["top_differs"]),
            "clear_top_differs": (res["fl_clear_top_differs"]["mean"], PILOT[name]["clear_top_differs"]),
            "rel_gap_median": (res["fl_rel_gap_median"], PILOT[name]["rel_gap_median"]),
            "sign_flip": (res["fl_sign_flip"]["mean"], PILOT[name]["sign_flip"]),
            "trace_change_real": (res["pilot_trace_change_real"]["rate"], PILOT[name]["trace_change_real"]),
            "trace_change_real_n": (res["pilot_trace_change_real"]["n_pairs"], PILOT[name]["trace_change_real_n"]),
            "locality_10pct": (res["locality"]["share_gt_10pct"], PILOT[name]["locality_10pct"]),
            "I_median": (res["I"]["median"], PILOT[name]["I_median"]),
        }
    hist(I, name, visits)
    return res


def hist(I, name, visits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(I, bins=30, color="#4C72B0")
    ax.set_xlabel("trust score I (share of pairwise explanation energy that is contradiction)")
    ax.set_ylabel("patients")
    ax.set_title(f"{name} [{visits} visits], n = {len(I)}")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_I_hist_{name}_{visits}.png", dpi=150)
    plt.close(fig)


def fmt(b):
    return f"{100 * b['mean']:.1f}% [{100 * b['lo']:.1f}, {100 * b['hi']:.1f}]"


def write_md(results):
    L = ["# TRACE Stage 4: size of the problem", "",
         "Patient-level means with patient-bootstrap 95% CIs (1,000 draws). Definitions are in the module "
         "docstring of `src/trace_pd/evaluation/trace_stage4.py`. Aggregates only.", ""]
    L += ["## Pilot reproduction (all visits)", "", "| Model | Metric | Now | Pilot |", "|---|---|---|---|"]
    for r in results:
        for k, (now, pil) in r.get("pilot_reproduction", {}).items():
            f = (lambda x: f"{x}") if k.endswith("_n") else (lambda x: f"{100 * x:.1f}%" if k != "I_median" else f"{x:.3f}")
            L.append(f"| {r['model']} | {k} | {f(now)} | {f(pil)} |")
    L += ["", "## Chain vs direct, compared with seed noise", "",
          "| Run | Patients | First->last: top differs | ... seed noise | ... difference | Gap pairs: top differs | "
          "... seed noise | ... difference | First->last total disagreement (median) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['model']} [{r['visits']}] | {r['n_patients']} | {fmt(r['fl_top_differs'])} | "
                 f"{fmt(r['seed_noise']['first_last'])} | {fmt(r['chain_minus_noise']['first_last'])} | "
                 f"{fmt(r['gap_top_differs'])} | {fmt(r['seed_noise']['gap_pairs'])} | "
                 f"{fmt(r['chain_minus_noise']['gap_pairs'])} | {100 * r['fl_rel_gap_median']:.1f}% |")
    L += ["", "## TRACE change rate and locality price", "",
          "| Run | TRACE vs direct top differs (first->last) | (gap pairs) | (all pairs) | "
          "unchanged feature > 10% of pair explanation | > 5% | abs > 0.05 log-odds | 99th pct abs |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lc = r["locality"]
        L.append(f"| {r['model']} [{r['visits']}] | {fmt(r['fl_trace_change'])} | {fmt(r['gap_trace_change'])} | "
                 f"{fmt(r['all_trace_change'])} | {100 * lc['share_gt_10pct']:.2f}% | {100 * lc['share_gt_5pct']:.2f}% | "
                 f"{100 * lc['abs_gt_0_05']:.2f}% | {lc['abs_p99']:.3f} |")
    L += ["", "## Trust score I", "",
          "| Run | median [IQR] | 90th pct | by visits T (n: median) | Spearman(I, first->last disagreement) | "
          "Spearman(I, gap-pair disagreement) |", "|---|---|---|---|---|---|"]
    for r in results:
        I = r["I"]
        byT = "; ".join(f"{k}: {v['n']}: {v['median']:.3f}" for k, v in I["by_T"].items())
        s1, s2 = r["I_vs_disagreement"]["first_last"], r["I_vs_disagreement"]["gap_pairs"]
        L.append(f"| {r['model']} [{r['visits']}] | {I['median']:.3f} [{I['q25']:.3f}, {I['q75']:.3f}] | "
                 f"{I['q90']:.3f} | {byT} | {s1['rho']:.2f} [{s1['lo']:.2f}, {s1['hi']:.2f}] | "
                 f"{s2['rho']:.2f} [{s2['lo']:.2f}, {s2['hi']:.2f}] |")
    L += ["", "Seed check: max margin difference between retrained seeds = "
          + ", ".join(f"{r['model']}: {r['seed_noise']['seed_effect_max_margin_diff']:.2f}" for r in results
                      if r["visits"] == "all"), ""]
    (OUT / "STAGE4.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    runs = RUNS if len(sys.argv) == 1 else [(s.partition(":")[0], s.partition(":")[2] or "all") for s in sys.argv[1:]]
    results = []
    for name, visits in runs:
        r = run(name, visits)
        (OUT / f"stage4_{name}_{visits}.json").write_text(json.dumps(r, indent=2))
        results.append(r)
        print(f"done {name} [{visits}]", flush=True)
    write_md(results)
    print((OUT / "STAGE4.md").read_text(encoding="utf-8"))
