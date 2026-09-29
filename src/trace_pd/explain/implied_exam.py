"""C1 -- Implied-Exam Explanation of the subtype classifier.

The classifier g maps 26 cheap features x to TD / PIGD. The label it imitates is a
known formula of two hidden scores (tremor T, gait P). We explain g by estimating
the scores it *behaves as if* it had seen:

    h_T(x) ~ log T,   h_P(x) ~ log P        (concept heads, trained on the true scores)
    l_hat(x) = h_T(x) - h_P(x)               (implied log-ratio)
    decision  = zone(l_hat) against a cutoff c*

c* is fitted so the implied decision reproduces g on the training data (fidelity
calibration). Because l_hat is a difference of two heads, feature attributions split
EXACTLY into a tremor channel and a gait channel:

    phi_j(l_hat) = phi_j(h_T) - phi_j(h_P)

and are reported in the formula's own units: how far feature j moves the patient
toward / away from the cutoff.

Identifiability note: the formula only depends on the ratio through three zones, so
g's output alone cannot identify T and P separately. The heads are therefore trained
with concept supervision (true scores) and fidelity to g is MEASURED, not assumed.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from .shapley import shapley_mc
from . import formula as F

HEAD_KW = dict(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42)


class ImpliedExamExplainer:
    def __init__(self, feature_names, head_kw=None):
        self.features = list(feature_names)
        kw = HEAD_KW if head_kw is None else head_kw
        self.hT = HistGradientBoostingRegressor(**kw)
        self.hP = HistGradientBoostingRegressor(**kw)
        self.cut = 0.0                                  # binary TD-vs-PIGD cutoff on l_hat

    # ------------------------------------------------------------------ fitting
    def fit(self, X, logT, logP):
        self.hT.fit(X, logT)
        self.hP.fit(X, logP)
        return self

    def calibrate_fidelity(self, X, g_pigd):
        """Pick the cutoff c* on l_hat whose decision best reproduces g's decisions.
        g_pigd: classifier's binary decision (1 = PIGD) on the SAME rows."""
        l = self.implied(X)[2]
        grid = np.quantile(l, np.linspace(0.02, 0.98, 97))
        agree = [np.mean((l <= c).astype(int) == g_pigd) for c in grid]
        self.cut = float(grid[int(np.argmax(agree))])
        return self

    # ---------------------------------------------------------------- inference
    def implied(self, X):
        lT, lP = self.hT.predict(X), self.hP.predict(X)
        return lT, lP, lT - lP

    def decide(self, X):
        return (self.implied(X)[2] <= self.cut).astype(int)           # 1 = PIGD

    # --------------------------------------------------------------- explaining
    def explain(self, X, background, n_perm=32, seed=0):
        """Per-feature attributions to the tremor channel, gait channel and ratio.
        All three are exactly additive:  sum_j phi_T = h_T(x) - E h_T,  etc."""
        X, B = np.asarray(X, float), np.asarray(background, float)
        phiT, baseT = shapley_mc(self.hT.predict, X, B, n_perm, rng=np.random.default_rng(seed))
        phiP, baseP = shapley_mc(self.hP.predict, X, B, n_perm, rng=np.random.default_rng(seed))
        lT, lP, l = self.implied(X)
        return dict(phi_tremor=phiT, phi_gait=phiP, phi_ratio=phiT - phiP,
                    base_tremor=baseT, base_gait=baseP, base_ratio=baseT - baseP,
                    log_tremor=lT, log_gait=lP, log_ratio=l,
                    margin=l - self.cut)                                # <0 -> PIGD side

    def narrative(self, e, i, top=3):
        """Plain-language explanation for row i of an explain() result."""
        T, P, l, m = (np.exp(e["log_tremor"][i]) - F.EPS, np.exp(e["log_gait"][i]) - F.EPS,
                      e["log_ratio"][i], e["margin"][i])
        side = "PIGD" if m <= 0 else "TD"
        phi = e["phi_ratio"][i]
        order = np.argsort(phi if side == "PIGD" else -phi)[:top]            # features pushing toward the decision
        drivers = []
        for j in order:
            ch = "gait" if abs(e["phi_gait"][i, j]) >= abs(e["phi_tremor"][i, j]) else "tremor"
            share = abs(phi[j]) / max(np.abs(phi).sum(), 1e-9) * 100
            drivers.append(f"{self.features[j]} (via the {ch} channel, {share:.0f}% of the total attribution)")
        T, P = max(T, 0.0), max(P, 0.0)
        ratio = f"{T / P:.2f}" if P > 0 else "undefined (gait 0)"
        return (f"Predicted {side}. The model behaves as if tremor ~ {T:.2f} and gait ~ {P:.2f} "
                f"(implied ratio {ratio}; model-calibrated cutoff {np.exp(self.cut):.2f} on the smoothed ratio; "
                f"margin {abs(m):.2f} log-units). "
                f"Main drivers: " + "; ".join(drivers) + ".")
