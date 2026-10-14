"""TRACE Stages 6 (faithfulness), 8 (shuffled-label sanity check) and 9 (descriptive clinical
analyses). docs/TRACE_BRIEF.md. Primary models: ppmi, adni_minus.

Stage 6: for every pair s < t and method, rank features by |A_j(s->t)| (ties -> lower feature
  index), revert the top-k features of visit t to visit s's values (real recorded values,
  NaN copied, nothing interpolated), recovered_k = (f(t) - f(reverted)) / (f(t) - f(s)).
  Pairs with |f(t) - f(s)| < eps are excluded. eps was still OPEN when this ran: main results
  use the proposed 0.05, with 0.01 / 0.1 / 0.25 as sensitivity. Within-patient mean, then
  patient bootstrap. Direct Baseline Shapley wins on its own pair by construction (upper reference).
Stage 8: retrain on the development split with PATIENT-level shuffled labels (each patient's
  baseline label permuted across patients, applied to all its training visits; real visits,
  so not synthetic), same hyperparameters. Report locked-test BA and the Spearman correlation
  of per-feature mean |u_j| with the real model's.
Stage 9 (descriptive only, not causal; feature groups = DRAFT configs/feature_groups.yaml):
  9.2 PPMI model-predicted flips (margin crosses 0 between consecutive visits): share of
      TRACE credit psi(k->k+1) by feature and group; agreement with label flips.
  9.3 measurement accounting: share of |u| from comparisons where a feature's
      recorded/missing status differs; ADNI by study phase (COLPROT).
  9.4 within/between: b_j(t) = phi_pop_j(t) - u_j(t); identity sum_j b_j(t) constant over a
      patient's visits; per-feature mean |u|, mean |b|, within-patient SD of b.
Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_stage689 [6|8|9] ...
Out:  results/stage6/, results/stage8/, results/stage9/ (aggregates only)
"""
import json
import sys

import numpy as np
import yaml
from scipy.stats import spearmanr

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.evaluation import trace_baselines as tb
from trace_pd.evaluation.trace_stage4 import boot
from trace_pd.explain import trace_engine as te
from trace_pd.models.backend import make_classifier, fit_classifier

PRIMARY = ["ppmi", "adni_minus"]
METHODS = ["trace", "direct", "chain", "fixed_first", "fixed_last", "population", "path_dep", "ar1"]
EPS_MAIN, EPS_SENS = 0.05, [0.01, 0.1, 0.25]
KS = [1, 2, 3, 4, 5]
R = config.ROOT / "results"


# ------------------------------------------------------------------------- Stage 6
def stage6(name):
    b = tr.get_model(name)
    pats = tr.load_cache(name)
    V = tb.load_visit(name)
    AR = tb.load_ar1(name)
    rows = {m: dict(pat=[], Xt=[], Xs=[], A=[], df=[], ft=[]) for m in METHODS}
    for pi, p in enumerate(pats):
        F, X = p["F"], p["X"]
        f = te.exact_margin(b, X)
        T = len(X)
        E = tb.explanations(F, V[p["idx"]]["pop"], V[p["idx"]]["pd"], AR[p["idx"]]["phi"])
        s, t = np.triu_indices(T, 1)
        for m in METHODS:
            A = E[m][s, t]
            ok = np.isfinite(A).all(1)
            rows[m]["pat"].append(np.full(ok.sum(), pi)); rows[m]["Xt"].append(X[t[ok]]); rows[m]["Xs"].append(X[s[ok]])
            rows[m]["A"].append(A[ok]); rows[m]["df"].append(f[t[ok]] - f[s[ok]]); rows[m]["ft"].append(f[t[ok]])
    rec = {}
    for m in METHODS:
        d = {k: np.concatenate(v) for k, v in rows[m].items()}
        order = np.argsort(-np.abs(d["A"]), axis=1, kind="stable")
        out = np.zeros((len(d["df"]), len(KS)))
        for i, k in enumerate(KS):
            mask = np.zeros_like(d["A"], bool)
            np.put_along_axis(mask, order[:, :k], True, axis=1)
            Xr = np.where(mask, d["Xs"], d["Xt"])
            with np.errstate(divide="ignore", invalid="ignore"):
                out[:, i] = (d["ft"] - te.exact_margin(b, Xr)) / d["df"]
        rec[m] = dict(pat=d["pat"], df=d["df"], r=out)
    res = {"model": name, "n_patients": len(pats)}
    for eps in [EPS_MAIN] + EPS_SENS:
        key = f"eps_{eps}"
        res[key] = {}
        per_pat = {}
        for m in METHODS:
            keep = np.abs(rec[m]["df"]) >= eps
            pm = np.full((len(pats), len(KS)), np.nan)
            for pi in range(len(pats)):
                sel = keep & (rec[m]["pat"] == pi)
                if sel.any():
                    pm[pi] = rec[m]["r"][sel].mean(0)
            per_pat[m] = pm
            res[key][m] = dict(n_pairs=int(keep.sum()), n_excluded=int((~keep).sum()),
                               recovered={k: boot(pm[:, i]) for i, k in enumerate(KS)})
        # paired: TRACE minus method (> 0 = TRACE recovers more). For ar1, TRACE on ar1's pairs.
        tr_on_ar1 = np.full((len(pats), len(KS)), np.nan)
        keep_a = np.abs(rec["ar1"]["df"]) >= eps
        # trace restricted to s >= 1 pairs: recompute from trace rows matching ar1's pair set
        tpat, tdf = rec["trace"]["pat"], rec["trace"]["df"]
        for pi in range(len(pats)):
            T = len(pats[pi]["f"])
            s, t = np.triu_indices(T, 1)
            sel_t = np.flatnonzero(tpat == pi)
            okp = (s >= 1) & (np.abs(tdf[sel_t]) >= eps)
            if okp.any():
                tr_on_ar1[pi] = rec["trace"]["r"][sel_t][okp].mean(0)
        res[key]["trace_minus_method"] = {
            m: {k: boot(((tr_on_ar1 if m == "ar1" else per_pat["trace"]) - per_pat[m])[:, i]) for i, k in enumerate(KS)}
            for m in METHODS if m != "trace"}
    return res


# ------------------------------------------------------------------------- Stage 8
def stage8(name):
    from xgboost import Booster
    from sklearn.metrics import balanced_accuracy_score

    spec = tr.MODELS[name]
    c = tr.COHORT[spec["cohort"]]
    dev, test, _ = tr.load_cohort(spec["cohort"])
    F = tr.features(name)
    s = dev[dev.LABEL.isin([c["neg"], c["pos"]])].sort_values([c["id"], "VISIT_MONTH"])
    base = s.groupby(c["id"])["LABEL"].first()
    rng = np.random.default_rng(tr.CFG["seeds"]["global"])
    perm = dict(zip(base.index, rng.permutation(base.to_numpy())))
    y = (s[c["id"]].map(perm) == c["pos"]).astype(int)
    path = tr.MODEL_DIR / f"{name}_shuffled.json"
    if not path.exists():
        clf = fit_classifier(make_classifier("xgb", 2), s[F].to_numpy(float), y)
        clf.get_booster().save_model(path)
    bs = Booster()
    bs.load_model(path)
    tt = test[test.LABEL.isin([c["neg"], c["pos"]])]
    ba = balanced_accuracy_score((tt.LABEL == c["pos"]).astype(int), (te.margin(bs, tt[F].to_numpy(float)) > 0).astype(int))
    pats = tr.load_cache(name)
    leaves = te._leaf_paths(bs)
    u_real = np.vstack([te.trace_from_F(p["F"])[0] for p in pats])
    u_shuf = np.vstack([te.trace_from_F(te.pairwise_F_fast(bs, p["X"], leaves))[0] for p in pats])
    mr, ms = np.abs(u_real).mean(0), np.abs(u_shuf).mean(0)
    rho = spearmanr(mr, ms).correlation
    # per-visit agreement of tallies (Spearman across features, averaged over visits)
    per_visit = [spearmanr(a, b_).correlation for a, b_ in zip(u_real, u_shuf) if np.std(a) > 0 and np.std(b_) > 0]
    return dict(model=name, shuffled_test_ba=float(ba), n_test_visits=int(len(tt)),
                spearman_mean_abs_u=float(rho), mean_abs_u_total_real=float(mr.sum()), mean_abs_u_total_shuffled=float(ms.sum()),
                per_visit_spearman_mean=float(np.nanmean(per_visit)), features=F,
                mean_abs_u_real=mr.tolist(), mean_abs_u_shuffled=ms.tolist())


# ------------------------------------------------------------------------- Stage 9
def groups(name):
    g = yaml.safe_load(open(config.ROOT / "configs" / "feature_groups.yaml", encoding="utf-8"))[name]
    F = tr.features(name)
    assert sorted(f for v in g.values() for f in v) == sorted(F)
    return {k: [F.index(f) for f in v] for k, v in g.items()}


def stage9(name):
    b = tr.get_model(name)
    F = tr.features(name)
    G = groups(name)
    pats = tr.load_cache(name)
    frames = dict(tr.analysis_patients(tr.MODELS[name]["cohort"]))
    V = tb.load_visit(name)
    res = {"model": name, "feature_groups_status": "DRAFT (not yet confirmed by the user)"}
    # ---- 9.2 flips (all models; brief asks PPMI, ADNI shown for completeness)
    thr = tr.CFG["decision_threshold_margin"]
    c = tr.COHORT[tr.MODELS[name]["cohort"]]
    shares, up, down = [], [], []
    n_model_flips = n_label_flips = n_both = n_lab_pairs = 0
    for p in pats:
        f = te.exact_margin(b, p["X"])
        _, psi, _, _ = te.trace_from_F(p["F"])
        lab = frames[p["idx"]]["LABEL"].to_numpy()
        for k in range(len(f) - 1):
            mflip = (f[k] > thr) != (f[k + 1] > thr)
            if mflip:
                n_model_flips += 1
                a = psi[k, k + 1]
                sh = np.abs(a) / np.abs(a).sum()
                shares.append(sh)
                (up if f[k + 1] > f[k] else down).append(a)
            if lab[k] in (c["neg"], c["pos"]) and lab[k + 1] in (c["neg"], c["pos"]):
                n_lab_pairs += 1
                lflip = lab[k] != lab[k + 1]
                n_label_flips += int(lflip)
                n_both += int(lflip and mflip)
    S = np.array(shares)
    res["flips"] = dict(
        n_model_flips=n_model_flips, n_up=len(up), n_down=len(down),
        label_pairs_both_labelled=n_lab_pairs, label_flips=n_label_flips, label_flips_also_model_flips=n_both,
        mean_share_by_feature={F[j]: float(S[:, j].mean()) for j in np.argsort(-S.mean(0))} if len(S) else {},
        mean_share_by_group={g: float(S[:, idx].sum(1).mean()) for g, idx in G.items()} if len(S) else {},
        mean_signed_credit_up_by_group={g: float(np.array(up)[:, idx].sum(1).mean()) for g, idx in G.items()} if up else {},
        mean_signed_credit_down_by_group={g: float(np.array(down)[:, idx].sum(1).mean()) for g, idx in G.items()} if down else {})
    # ---- 9.3 measurement accounting
    tot_missdiff = tot_all = 0.0
    by_phase = {}
    for p in pats:
        F_, X = p["F"], p["X"]
        T = len(X)
        miss = np.isnan(X)
        differs = miss[:, None, :] != miss[None, :, :]               # [r, t, j]
        u_md = (F_ * differs).sum(0) / T                              # [t, j]
        u = F_.mean(0)
        tot_missdiff += np.abs(u_md).sum()
        tot_all += np.abs(u).sum()
        if "COLPROT" in frames[p["idx"]]:
            for t, ph in enumerate(frames[p["idx"]]["COLPROT"].to_numpy()):
                d = by_phase.setdefault(str(ph), [0.0, 0.0, 0])
                d[0] += np.abs(u_md[t]).sum(); d[1] += np.abs(u[t]).sum(); d[2] += 1
    res["measurement"] = dict(
        share_abs_u_from_missingness_differences=float(tot_missdiff / tot_all),
        note="share of sum |u_j(t)| contributed by comparisons where feature j's recorded/missing status differs "
             "(the two parts are not sign-separated, so the |.| shares are indicative)",
        by_phase={k: dict(share=float(v[0] / v[1]), visits=v[2]) for k, v in sorted(by_phase.items())})
    # ---- 9.4 within / between
    bg = tb.population_background(name)
    fbar_bg = float(te.exact_margin(b, bg).mean())
    ident, sd_b, abs_u, abs_b = 0.0, [], [], []
    for p in pats:
        u = p["F"].mean(0)
        pop = V[p["idx"]]["pop"]
        bb = pop - u
        sb = bb.sum(1)
        f = te.exact_margin(b, p["X"])
        ident = max(ident, float(np.abs(sb - (f.mean() - fbar_bg)).max()))
        abs_u.append(np.abs(u).mean(0)); abs_b.append(np.abs(bb).mean(0)); sd_b.append(bb.std(0))
    abs_u, abs_b, sd_b = np.array(abs_u).mean(0), np.array(abs_b).mean(0), np.array(sd_b).mean(0)
    res["within_between"] = dict(
        identity_max_error=ident, identity_tolerance=tr.CFG["tolerances"]["between_identity"],
        identity_pass=bool(ident <= tr.CFG["tolerances"]["between_identity"]),
        share_within=float(abs_u.sum() / (abs_u.sum() + abs_b.sum())),
        by_feature={F[j]: dict(mean_abs_within_u=float(abs_u[j]), mean_abs_between_b=float(abs_b[j]),
                               within_patient_sd_of_b=float(sd_b[j])) for j in np.argsort(-(abs_u + abs_b))},
        by_group={g: dict(mean_abs_within_u=float(abs_u[idx].sum()), mean_abs_between_b=float(abs_b[idx].sum()))
                  for g, idx in G.items()})
    return res


# ------------------------------------------------------------------------- figures
def faithfulness_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.8), squeeze=False)
    for ax, r in zip(axes[0], results):
        for m in METHODS:
            v = r[f"eps_{EPS_MAIN}"][m]["recovered"]
            y = [v[k]["mean"] for k in KS]
            lo = [v[k]["lo"] for k in KS]
            hi = [v[k]["hi"] for k in KS]
            lw = 2.4 if m == "trace" else 1.2
            ax.plot(KS, y, marker="o", lw=lw, label=m)
            ax.fill_between(KS, lo, hi, alpha=0.12)
        ax.set_xlabel("top-k features reverted to the earlier visit's values")
        ax.set_ylabel("share of the score change recovered")
        ax.set_title(f"{r['model']} (eps = {EPS_MAIN}; {r['n_patients']} patients)", fontsize=9)
        ax.set_xticks(KS)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    (R / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(R / "figures" / "fig_faithfulness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1]
    names = sys.argv[2:] or PRIMARY
    if what == "6":
        (R / "stage6").mkdir(parents=True, exist_ok=True)
        out = [stage6(n) for n in names]
        for r in out:
            (R / "stage6" / f"faithfulness_{r['model']}.json").write_text(json.dumps(r, indent=2))
        faithfulness_figure(out)
    elif what == "8":
        (R / "stage8").mkdir(parents=True, exist_ok=True)
        for n in names:
            r = stage8(n)
            (R / "stage8" / f"sanity_{n}.json").write_text(json.dumps(r, indent=2))
            print(n, {k: r[k] for k in ["shuffled_test_ba", "spearman_mean_abs_u", "per_visit_spearman_mean",
                                        "mean_abs_u_total_real", "mean_abs_u_total_shuffled"]})
    elif what == "9":
        (R / "stage9").mkdir(parents=True, exist_ok=True)
        for n in names:
            r = stage9(n)
            (R / "stage9" / f"clinical_{n}.json").write_text(json.dumps(r, indent=2))
            print(n, "done")
