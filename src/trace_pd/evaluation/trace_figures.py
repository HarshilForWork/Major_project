"""TRACE headline-patient figures (brief Stage 9.5), one per cohort.

Selection rule (fixed in advance, not chosen by eye): among analysed locked-test patients
whose first->last step-by-step (chain) and direct explanations name CLEARLY different top
causes (direct's 2nd-largest |credit| < 80% of its largest), take the one with the largest
|change in model score| between first and last visit. Shown as "Patient A"; x-axis = visit
number only. No IDs, dates, ages or raw feature values appear (feature NAMES only).

Figures (results/figures/):
    fig_headline_<model>_routes.png  first->last: step-by-step vs direct vs TRACE, per feature
    fig_headline_<model>_curves.png  TRACE feature curves u_j(t), stacked; their sum reproduces
                                     the patient's score relative to the patient's own average

Run:  PYTHONPATH=src python -m trace_pd.evaluation.trace_figures [model ...]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trace_pd import config
from trace_pd.evaluation import trace_run as tr
from trace_pd.evaluation.trace_stage4 import chain_of, top
from trace_pd.explain import trace_engine as te

FIG = config.ROOT / "results" / "figures"
LABEL = {"ppmi": "PPMI (TD vs PIGD model)", "adni_minus": "ADNI (CN vs dementia, M_minus)"}


MIN_RECORDED = {"ppmi": 0.8, "adni_minus": 0.6}   # "well-recorded" variant; ADNI has no pair >= 75%


def pick(name, min_recorded=0.0):
    b = tr.get_model(name)
    best = None
    for p in tr.load_cache(name):
        F = p["F"]
        T = F.shape[0]
        rec = (~np.isnan(p["X"])).mean(1)
        if min(rec[0], rec[-1]) < min_recorded:
            continue
        direct, chain = F[0, T - 1], chain_of(F)[0, T - 1]
        jd, jc = top(direct), top(chain)
        if jd < 0 or jd == jc:
            continue
        a = np.sort(np.abs(direct))[::-1]
        if not a[1] < 0.8 * a[0]:
            continue
        f = te.exact_margin(b, p["X"])
        score = abs(f[-1] - f[0])
        if best is None or score > best[0]:
            best = (score, p, f)
    return best


def figures(name, min_recorded=0.0, tag="", who="Patient A"):
    FIG.mkdir(parents=True, exist_ok=True)
    feats = tr.features(name)
    score, p, f = pick(name, min_recorded)
    F = p["F"]
    T = F.shape[0]
    rec = (~np.isnan(p["X"])).sum(1)
    rec_note = f"features recorded at first / last visit: {rec[0]} / {rec[-1]} of {len(feats)}"
    u, psi, _, _ = te.trace_from_F(F)
    direct, chain, trace = F[0, T - 1], chain_of(F)[0, T - 1], psi[0, T - 1]
    # ---------------------------------------------------------------- routes figure
    idx = set()
    for v in (direct, chain, trace):
        idx |= set(np.argsort(-np.abs(v))[:4].tolist())
    idx = sorted(idx, key=lambda j: -abs(trace[j]))
    y = np.arange(len(idx))
    h = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(idx) + 1.6))
    ax.barh(y - h, chain[idx], h, label=f"step-by-step (top: {feats[top(chain)]})", color="#DD8452")
    ax.barh(y, direct[idx], h, label=f"direct first-to-last (top: {feats[top(direct)]})", color="#4C72B0")
    ax.barh(y + h, trace[idx], h, label=f"TRACE (top: {feats[top(trace)]})", color="#55A868")
    ax.set_yticks(y, [feats[j] for j in idx])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("credit for the change in model score (log-odds)")
    ax.set_title(f"{LABEL.get(name, name)}: {who}, first to last visit ({T} visits)\n"
                 f"all three explanations add up to the same change: {f[-1] - f[0]:+.2f} log-odds\n{rec_note}",
                 fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_headline_{name}{tag}_routes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # ---------------------------------------------------------------- curves figure
    k = 5
    main = list(np.argsort(-np.abs(u).max(0))[:k])
    rest = [j for j in range(len(feats)) if j not in main]
    series = [(feats[j], u[:, j]) for j in main] + [("all other features", u[:, rest].sum(1))]
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pos = np.zeros(T)
    neg = np.zeros(T)
    colors = plt.cm.tab10.colors
    for c, (lab, v) in enumerate(series):
        vp, vn = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.bar(x, vp, bottom=pos, color=colors[c % 10], width=0.8, label=lab)
        ax.bar(x, vn, bottom=neg, color=colors[c % 10], width=0.8)
        pos += vp
        neg += vn
    ax.plot(x, f - f.mean(), "k.-", lw=1.4, label="model score minus patient's own average")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{i}\n({r})" for i, r in zip(x, rec)])
    ax.set_xlabel(f"visit number (features recorded, of {len(feats)})")
    ax.set_ylabel("log-odds")
    ax.set_title(f"{LABEL.get(name, name)}: {who}, TRACE feature curves (stacked bars sum exactly to the black line)",
                 fontsize=8.5)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_headline_{name}{tag}_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    err = float(np.abs(u.sum(1) - (f - f.mean())).max())
    print(f"{name}{tag}: {who} has {T} visits; {rec_note}; score change {f[-1] - f[0]:+.2f}; top causes: "
          f"step-by-step {feats[top(chain)]}, direct {feats[top(direct)]}, TRACE {feats[top(trace)]}; "
          f"curve add-up error {err:.1e}")


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["ppmi", "adni_minus"]):
        figures(n)
        figures(n, MIN_RECORDED[n], "_wellrecorded", "Patient B")
