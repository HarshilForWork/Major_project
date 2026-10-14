"""TRACE: path-consistent explanations of within-patient prediction changes.

Brief: docs/TRACE_BRIEF.md (Section 3 has the maths). For one patient with visits
x^1..x^T and a single-visit model f (raw XGBoost margin):

    F[r, t, j] = phi_j(r->t)   Baseline Shapley of f(x^t) - f(x^r), exact interventional
                               TreeSHAP with the single reference row x^r
    u[t, j]    = mean_r F[r, t, j]                       the per-visit "tally"
    psi[s, t]  = u[t] - u[s]                             consistent change explanation
    R          = F - psi                                 the contradiction removed
    I          = sum_{r<t} R^2 / sum_{r<t} F^2           per-patient trust score

u is the least-squares projection of F onto contradiction-free (gradient) flows and equals
interventional TreeSHAP with the patient's own T visits as the background; the two
independent routes (`trace_via_background`, `trace_via_lstsq`) exist only to verify
`trace_from_F` in the Stage 3 gate.

NaN stays NaN everywhere (no imputation): shap and XGBoost route it by the learned
default direction.
"""
import numpy as np

__all__ = [
    "explainer", "shap_model_copy",
    "nan_equal", "margin", "exact_margin", "pairwise_F", "pairwise_F_fast", "baseline_shapley_batch", "trace_from_F", "trace_via_background",
    "trace_via_lstsq", "tree_output_range", "used_feature_mask",
]


# ------------------------------------------------------------------ shap compatibility
# shap 0.49.1 (the newest release, checked 14 Oct 2026) reads XGBoost's base_score with
# float(), but XGBoost 3.x serialises it as a one-element vector string "[4.9999997E-1]".
# Fix: explain a COPY of the model whose UBJSON serialisation carries the scalar string
# "4.9999997E-1". The original booster is never modified, shap itself is not patched, and
# `shap_model_copy` proves that the copy decodes to exactly the original model apart from
# that one string. All shap explainers in TRACE (engine and baselines) go through `explainer`.
_UBJ_INT = {b"i": (1, True), b"U": (1, False), b"I": (2, True), b"l": (4, True), b"L": (8, True)}


def _scalar_base_score_ubj(raw):
    key = b"base_score"
    i = raw.find(key)
    if i < 0 or raw.find(key, i + 1) >= 0:
        raise ValueError("expected exactly one base_score in the model")
    j = i + len(key)
    if raw[j:j + 1] != b"S":
        return raw                                    # already numeric: nothing to fix
    marker = raw[j + 1:j + 2]
    n_len, signed = _UBJ_INT[marker]
    n = int.from_bytes(raw[j + 2:j + 2 + n_len], "big", signed=signed)
    s0, s1 = j + 2 + n_len, j + 2 + n_len + n
    text = raw[s0:s1].decode()
    if not (text.startswith("[") and text.endswith("]")):
        return raw
    vals = [v.strip() for v in text[1:-1].split(",") if v.strip()]
    if len(vals) != 1:
        raise ValueError(f"multi-element base_score {text!r} is not supported")
    new = vals[0].encode()
    return raw[:j + 2] + len(new).to_bytes(n_len, "big", signed=signed) + new + raw[s1:]


def _shap_compat_class():
    from xgboost import Booster

    class ShapCompatBooster(Booster):
        """Copy of an XGBoost 3.x booster whose UBJSON reports a scalar base_score."""

        def save_raw(self, raw_format="ubj"):
            raw = super().save_raw(raw_format)
            return bytearray(_scalar_base_score_ubj(bytes(raw))) if raw_format == "ubj" else raw

    return ShapCompatBooster


def shap_model_copy(booster):
    """Patched copy of `booster` for shap, verified identical apart from base_score's format."""
    import io
    from shap.explainers._tree import decode_ubjson_buffer

    copy = _shap_compat_class()(model_file=bytearray(booster.save_raw("ubj")))
    a = decode_ubjson_buffer(io.BytesIO(bytes(booster.save_raw("ubj"))))
    b = decode_ubjson_buffer(io.BytesIO(bytes(copy.save_raw("ubj"))))
    pa, pb = a["learner"]["learner_model_param"], b["learner"]["learner_model_param"]
    assert float(str(pa["base_score"]).strip("[]")) == float(pb["base_score"]), "base_score value changed"
    pa = dict(pa, base_score=None)
    pb = dict(pb, base_score=None)
    assert pa == pb and {k: v for k, v in a["learner"].items() if k != "learner_model_param"} == \
        {k: v for k, v in b["learner"].items() if k != "learner_model_param"}, "copy differs from original"
    return copy


_COPIES = {}


def explainer(booster, background=None, perturbation="interventional"):
    """shap.TreeExplainer on the verified compatibility copy (cached per booster)."""
    import shap

    key = id(booster)
    if key not in _COPIES:
        _COPIES[key] = (booster, shap_model_copy(booster))     # keep booster alive so id() stays unique
    model = _COPIES[key][1]
    if perturbation == "interventional":
        bg = np.asarray(background, float)
        # shap silently subsamples a plain-array background to 100 rows; pass a masker that
        # keeps every row (found 14 Oct 2026 when the Stage 9.4 identity failed).
        data = bg if len(bg) <= 100 else shap.maskers.Independent(bg, max_samples=len(bg))
        return shap.TreeExplainer(model, data=data, feature_perturbation="interventional", model_output="raw")
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")


def _explainer(booster, background):
    return explainer(booster, background)


# -------------------------------------------------------------------------- helpers
def nan_equal(a, b):
    """Elementwise equality that treats NaN == NaN as equal (numpy's == does not)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a == b) | (np.isnan(a) & np.isnan(b))


def margin(booster, X):
    """Raw margin (log-odds) of every row; NaN passed through as missing."""
    from xgboost import DMatrix

    return booster.predict(DMatrix(np.asarray(X, float), missing=np.nan), output_margin=True).astype(float)


def exact_margin(booster, X):
    """The model's raw margin computed exactly: XGBoost's own routing (float32 values and
    thresholds, x < split goes left, NaN follows default_left) but leaf values summed in
    float64. `booster.predict` sums the trees in float32 and is off by up to ~1e-5 on
    margins near 10 (results/gate/G1_diagnosis.md); G1 is measured against this instead
    (user decision, 14 Oct 2026)."""
    import json

    import scipy.special

    js = json.loads(booster.save_raw("json"))
    X32 = np.asarray(X, float).astype(np.float32)
    n = len(X32)
    out = np.zeros(n)
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        L, R = np.asarray(t["left_children"]), np.asarray(t["right_children"])
        si = np.asarray(t["split_indices"])
        sc = np.asarray(t["split_conditions"], np.float32)
        dl = np.asarray(t["default_left"]).astype(bool)
        node = np.zeros(n, int)
        while True:
            internal = L[node] != -1
            if not internal.any():
                break
            rows = np.flatnonzero(internal)
            nd = node[rows]
            x = X32[rows, si[nd]]
            with np.errstate(invalid="ignore"):
                go_left = np.where(np.isnan(x), dl[nd], x < sc[nd])
            node[rows] = np.where(go_left, L[nd], R[nd])
        out += sc[node].astype(np.float64)
    p = js["learner"]["learner_model_param"]
    base = float(str(p["base_score"]).strip("[]"))
    if js["learner"]["objective"]["name"] in ("binary:logistic", "reg:logistic"):
        base = float(scipy.special.logit(base))
    return out + base


def used_feature_mask(booster, d):
    """True for features the model splits on at least once."""
    import json

    js = json.loads(booster.save_raw("json"))
    used = np.zeros(d, bool)
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        left = np.asarray(t["left_children"])
        idx = np.asarray(t["split_indices"])[left != -1]
        used[idx] = True
    return used


def tree_output_range(booster):
    """R = sum over trees of (max leaf - min leaf): a valid upper bound on the range of f."""
    import json

    js = json.loads(booster.save_raw("json"))
    R = 0.0
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        left = np.asarray(t["left_children"])
        leaves = np.asarray(t["split_conditions"], float)[left == -1]   # leaf values live here
        R += float(leaves.max() - leaves.min())
    return R


# ------------------------------------------------------------------------- the engine
def pairwise_F(booster, X):
    """F[r, t, j] = phi_j(r->t), exact Baseline Shapley via interventional TreeSHAP with the
    single reference row x^r. float64, shape (T, T, d). The diagonal is set to exactly 0;
    `diag_max` (returned) records what TreeSHAP produced there, as a check."""
    X = np.asarray(X, float)
    T, d = X.shape
    F = np.zeros((T, T, d))
    for r in range(T):
        F[r] = _explainer(booster, X[[r]]).shap_values(X)
    diag_max = float(np.abs(F[np.arange(T), np.arange(T)]).max()) if T else 0.0
    F[np.arange(T), np.arange(T)] = 0.0
    return F, diag_max


def _leaf_paths(booster):
    """Every leaf of every tree as (conditions, value); condition = (feature, float32
    threshold, goes_left, missing_goes_here). Read from the JSON model (exact float32)."""
    import json

    js = json.loads(booster.save_raw("json"))
    leaves = []
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        L, R = t["left_children"], t["right_children"]
        si, sc, dl = t["split_indices"], t["split_conditions"], t["default_left"]
        stack = [(0, [])]
        while stack:
            n, conds = stack.pop()
            if L[n] == -1:
                leaves.append((conds, float(np.float32(sc[n]))))
                continue
            s = np.float32(sc[n])
            stack.append((L[n], conds + [(si[n], s, True, bool(dl[n]))]))
            stack.append((R[n], conds + [(si[n], s, False, not bool(dl[n]))]))
    return leaves


def _sat(v, s, left, miss_here):
    v = v.astype(np.float32)
    with np.errstate(invalid="ignore"):
        ok = (v < s) if left else (v >= s)
    return np.where(np.isnan(v), miss_here, ok)


def baseline_shapley_batch(booster, X, Rf, leaves=None):
    """Exact Baseline Shapley phi(r->x) for many (x, r) rows at once, by leaf-path
    enumeration (the pilot's algorithm, notebooks/trace_pilot.py). A leaf is reachable by a
    hybrid iff each feature on its path is satisfied by x (feature swapped in) or by r (not);
    with A = features only x satisfies and B = only r satisfies, the leaf is the unanimity
    game w*1[A in S, S disjoint B], whose Shapley values are closed form. float64 sums.
    Equals interventional TreeSHAP with a single reference row (gate 'pilot' column,
    max diff <= 6.5e-8 on every model); used where thousands of retrained models or
    deletions make per-reference shap explainers too slow (Stages 4 and 7)."""
    from math import factorial

    X, Rf = np.asarray(X, float), np.asarray(Rf, float)
    P, d = X.shape
    if leaves is None:
        leaves = _leaf_paths(booster)
    fact = np.array([float(factorial(k)) for k in range(d + 2)])
    phi = np.zeros((P, d))
    for conds, w in leaves:
        feats = sorted({c[0] for c in conds})
        sx = {j: np.ones(P, bool) for j in feats}
        sr = {j: np.ones(P, bool) for j in feats}
        for j, s, left, mh in conds:
            sx[j] &= _sat(X[:, j], s, left, mh)
            sr[j] &= _sat(Rf[:, j], s, left, mh)
        reach = np.ones(P, bool)
        nA = np.zeros(P, int)
        nB = np.zeros(P, int)
        for j in feats:
            reach &= sx[j] | sr[j]
            nA += sx[j] & ~sr[j]
            nB += ~sx[j] & sr[j]
        tot = nA + nB
        ca = w * fact[np.maximum(nA - 1, 0)] * fact[nB] / fact[tot]
        cb = w * fact[nA] * fact[np.maximum(nB - 1, 0)] / fact[tot]
        for j in feats:
            a = reach & sx[j] & ~sr[j]
            b = reach & ~sx[j] & sr[j]
            phi[:, j] += np.where(a, ca, 0.0) - np.where(b, cb, 0.0)
    return phi


def pairwise_F_fast(booster, X, leaves=None):
    """F[r, t] for all ordered pairs of one patient via baseline_shapley_batch."""
    X = np.asarray(X, float)
    T = len(X)
    r, t = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    F = baseline_shapley_batch(booster, X[t.ravel()], X[r.ravel()], leaves).reshape(T, T, -1)
    F[np.arange(T), np.arange(T)] = 0.0
    return F


def trace_from_F(F):
    """Closed-form TRACE from the pairwise flows. Returns u (T, d), psi (T, T, d) with
    psi[s, t] = u[t] - u[s], R (T, T, d) and the trust score I (scalar)."""
    T = F.shape[0]
    u = F.mean(axis=0)
    psi = u[None, :, :] - u[:, None, :]
    R = F - psi
    iu = np.triu_indices(T, 1)
    den = float((F[iu] ** 2).sum())
    I = float((R[iu] ** 2).sum() / den) if den > 0 else float("nan")
    return u, psi, R, I


def trace_via_background(booster, X):
    """Independent route: interventional TreeSHAP of every visit with the patient's whole
    set of T visits as the background. Equals u from trace_from_F (linearity of Shapley)."""
    X = np.asarray(X, float)
    return _explainer(booster, X).shap_values(X)


def trace_via_lstsq(F):
    """Independent route: least-squares fit of u(t) - u(r) to F[r, t] over all r < t on the
    complete visit graph. lstsq returns the minimum-norm solution, which is orthogonal to
    the constant vector, i.e. exactly the gauge sum_t u(t) = 0 used by the closed form."""
    T, _, d = F.shape
    r, t = np.triu_indices(T, 1)
    B = np.zeros((len(r), T))
    B[np.arange(len(r)), t] = 1.0
    B[np.arange(len(r)), r] = -1.0
    u, *_ = np.linalg.lstsq(B, F[r, t], rcond=None)
    return u
