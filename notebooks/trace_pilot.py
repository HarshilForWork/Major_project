"""TRACE feasibility pilot: how intransitive are pairwise Baseline-Shapley explanations
of within-patient prediction changes, for our real XGBoost models on locked-test patients?

Exact Baseline Shapley for one (explicand x, reference r) pair on a tree ensemble:
each leaf is reachable by hybrid h(S) iff every feature on its path is satisfied by x
(if in S) or r (if not). Per leaf: A = feats only x satisfies, B = only r satisfies,
any feat satisfied by neither -> unreachable. Leaf game w*1[A subset S, S cap B empty];
Shapley: j in A -> w (|A|-1)!|B|!/(|A|+|B|)!, j in B -> -w |A|!(|B|-1)!/(|A|+|B|)!.
Prints aggregates only.
"""
import sys
from math import factorial
import numpy as np
import pandas as pd

from trace_pd import config
from trace_pd.models.backend import make_classifier, fit_classifier

try:                                   # pytest's captured stdout has no reconfigure()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def leaf_paths(model, F):
    """Exact float32 thresholds/leaf values from the JSON model (text dumps round)."""
    import json
    js = json.loads(model.get_booster().save_raw("json"))
    leaves = []
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        L, R = t["left_children"], t["right_children"]
        si, sc, dl = t["split_indices"], t["split_conditions"], t["default_left"]
        def walk(n, conds):
            if L[n] == -1:
                leaves.append((conds, float(np.float32(sc[n]))))
                return
            s = np.float32(sc[n])
            walk(L[n], conds + [(si[n], s, True, bool(dl[n]))])
            walk(R[n], conds + [(si[n], s, False, not bool(dl[n]))])
        walk(0, [])
    return leaves


def sat(v, s, yes, miss_here):
    v = v.astype(np.float32)
    nan = np.isnan(v)
    with np.errstate(invalid="ignore"):
        ok = (v < s) if yes else (v >= s)
    return np.where(nan, miss_here, ok)


def baseline_shapley(leaves, X, R, d):
    """X, R: (P, d). Returns (P, d) exact Baseline Shapley of f(X)-f(R)."""
    P = len(X)
    phi = np.zeros((P, d))
    for conds, w in leaves:
        feats = sorted({c[0] for c in conds})
        sx = {j: np.ones(P, bool) for j in feats}
        sr = {j: np.ones(P, bool) for j in feats}
        for j, s, yes, mh in conds:
            sx[j] &= sat(X[:, j], s, yes, mh)
            sr[j] &= sat(R[:, j], s, yes, mh)
        reach = np.ones(P, bool)
        nA = np.zeros(P, int); nB = np.zeros(P, int)
        for j in feats:
            reach &= sx[j] | sr[j]
            nA += sx[j] & ~sr[j]
            nB += ~sx[j] & sr[j]
        tot = nA + nB
        ca = w * FACT[np.maximum(nA - 1, 0)] * FACT[nB] / FACT[tot]
        cb = w * FACT[nA] * FACT[np.maximum(nB - 1, 0)] / FACT[tot]
        for j in feats:
            a = reach & sx[j] & ~sr[j]
            b = reach & ~sx[j] & sr[j]
            phi[:, j] += np.where(a, ca, 0.0) - np.where(b, cb, 0.0)
    return phi


FACT = np.array([float(factorial(k)) for k in range(40)])


def analyse(name, model, test, F, id_col):
    leaves = leaf_paths(model, F)
    d = len(F)
    rows = {"I": [], "top_direct_vs_chain": [], "sign_direct_vs_chain": [],
            "top_pair_vs_trace": [], "loc_viol": [], "T": []}
    eff_err = 0.0
    pats = []
    for pid, g in test.groupby(id_col):
        X = g.sort_values("VISIT_MONTH")[F].to_numpy(float)
        if len(X) >= 3:
            pats.append(X)
    XS, RS = [], []
    for X in pats:
        T = len(X)
        ii, tt = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        XS.append(X[tt.ravel()]); RS.append(X[ii.ravel()])
    ALL = baseline_shapley(leaves, np.vstack(XS), np.vstack(RS), d)
    off = 0
    for X in pats:
        T = len(X)
        fm = model.predict(X, output_margin=True)
        phi = ALL[off:off + T * T].reshape(T, T, d)  # phi[r,t,j] = phi_j(r->t)
        off += T * T
        eff_err = max(eff_err, np.abs(phi.sum(2) - (fm[None, :] - fm[:, None])).max())
        u = phi.mean(0)                                   # (T, d) TRACE potentials
        psi = u[None, :, :] - u[:, None, :]               # psi[s,t,j]
        iu = np.triu_indices(T, 1)
        Fe, Pe = phi[iu], psi[iu]
        den = (Fe ** 2).sum()
        if den < 1e-12:
            continue
        rows["I"].append(((Fe - Pe) ** 2).sum() / den)
        rows["T"].append(T)
        direct = phi[0, T - 1]
        chain = sum(phi[k, k + 1] for k in range(T - 1))
        if np.abs(direct).max() > 1e-9:
            j1, j2 = np.abs(direct).argmax(), np.abs(chain).argmax()
            rows["top_direct_vs_chain"].append(j1 != j2)
            a = np.sort(np.abs(direct))[::-1]
            rows.setdefault("clear_top_differs", []).append((j1 != j2) and a[1] < 0.8 * a[0])
            rows.setdefault("rel_gap", []).append(np.abs(direct - chain).sum() / np.abs(direct).sum())
            rows["sign_direct_vs_chain"].append(np.sign(direct[j1]) != np.sign(chain[j1]))
        # per-pair: top feature of pairwise BShap vs TRACE, pairs with a real change
        big = np.abs(Fe).max(1) > 0.05
        if big.any():
            rows["top_pair_vs_trace"].extend(np.abs(Fe[big]).argmax(1) != np.abs(Pe[big]).argmax(1))
        # locality price: unchanged feature, |psi| relative to total |change|
        same = (X[iu[0]] == X[iu[1]]) | (np.isnan(X[iu[0]]) & np.isnan(X[iu[1]]))
        dtot = np.abs(Pe).sum(1, keepdims=True) + 1e-12
        rows["loc_viol"].extend((np.abs(Pe) / dtot)[same])
    I = np.array(rows["I"]); Tn = np.array(rows["T"])
    lv = np.array(rows["loc_viol"])
    print(f"\n== {name}  (patients with >=3 visits: {len(I)}, median visits {np.median(Tn):.0f})")
    print(f"  efficiency check, max |sum phi - delta f|: {eff_err:.2e}")
    print(f"  intransitivity index I: median {np.median(I):.3f}, 75th {np.quantile(I,.75):.3f}, 90th {np.quantile(I,.9):.3f}")
    for lo, hi in [(3, 4), (5, 9), (10, 99)]:
        m = (Tn >= lo) & (Tn <= hi)
        if m.sum():
            print(f"    T {lo}-{hi}: n={m.sum()}, median I {np.median(I[m]):.3f}")
    print(f"  first->last: top feature differs, direct vs chained consecutive: {np.mean(rows['top_direct_vs_chain']):.1%}")
    print(f"  ... of which the direct top feature was a CLEAR winner (2nd < 80% of 1st): {np.mean(rows['clear_top_differs']):.1%}")
    print(f"  first->last: total |direct - chained| as share of total |direct|: median {np.median(rows['rel_gap']):.1%}")
    print(f"  first->last: sign of direct top feature flips in chained version: {np.mean(rows['sign_direct_vs_chain']):.1%}")
    print(f"  pairs with real change: top feature pairwise vs TRACE differs: {np.mean(rows['top_pair_vs_trace']):.1%} (n={len(rows['top_pair_vs_trace'])})")
    print(f"  locality price: unchanged features getting >10% of a pair's explanation: {np.mean(lv > 0.10):.1%} of (pair,feature) cases")


def ppmi():
    from trace_pd.evaluation.splits import split_dev_test
    df = pd.read_csv(config.LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
    df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    dev, test = split_dev_test(df)
    s = dev[dev.LABEL.isin(["TD", "PIGD"])]
    m = fit_classifier(make_classifier("xgb", 2), s[F].to_numpy(float), (s.LABEL == "PIGD").astype(int))
    analyse("PPMI TD vs PIGD", m, test, F, "PATNO")


def adni():
    from trace_pd.evaluation.splits_adni import split_dev_test
    df = pd.read_csv(config.ADNI_LONG_TABLE, low_memory=False)
    dd = pd.read_csv(config.ADNI_DICTIONARY)
    F = sorted(dd.loc[dd.bucket == "CHEAP_FEATURE", "column"])
    for c in F:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c], sort=True)[0].astype(float)
            df.loc[df[c] < 0, c] = np.nan
    df[F] = df[F].apply(pd.to_numeric, errors="coerce")
    dev, test = split_dev_test(df)
    s = dev[dev.LABEL.isin(["CN", "Dementia"])]
    y = (s.LABEL == "Dementia").astype(int)
    m = fit_classifier(make_classifier("xgb", 2), s[F].to_numpy(float), y)
    analyse("ADNI CN vs Dementia, full features", m, test, F, "RID")
    Fm = [f for f in F if not any(k in f.upper() for k in ("MMSE", "FAQ"))]
    print(f"  (M_minus drops: {sorted(set(F) - set(Fm))})")
    m2 = fit_classifier(make_classifier("xgb", 2), s[Fm].to_numpy(float), y)
    tt = test[test.LABEL.isin(["CN", "Dementia"])]
    from sklearn.metrics import balanced_accuracy_score
    print(f"  M_minus locked-test balanced accuracy: {balanced_accuracy_score((tt.LABEL=='Dementia').astype(int), m2.predict(tt[Fm].to_numpy(float))):.3f}")
    analyse("ADNI CN vs Dementia, M_minus (no MMSE/FAQ)", m2, test, Fm, "RID")


if __name__ == "__main__":
    ppmi()
    adni()
