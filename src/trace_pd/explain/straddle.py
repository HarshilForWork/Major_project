"""C2 -- Cutoff-straddle explanation of the conformal prediction set.

Per channel, a predictive half-width from quantile heads (5% / 95%):
    hw_T(x), hw_P(x)          uncertainty of the implied log T and log P
The implied log-ratio interval is the Minkowski sum of the two channels:
    l in  l_hat(x)  +/-  kappa * (hw_T(x) + hw_P(x))

kappa is FIDELITY-CALIBRATED: chosen on held-out calibration rows so that
"interval straddles the decision cutoff" reproduces the black-box conformal set's
ambiguity ({TD, PIGD} vs singleton) as closely as possible. (With raw 95% channel
intervals the Minkowski interval straddles almost always, which explains nothing;
the first version of this module did exactly that.)

EXPLANATION, with d = |l_hat - cutoff|:
  straddle                 d <  kappa (hw_T + hw_P)
  resolves if tremor known d >= kappa hw_P      (tremor uncertainty removed)
  resolves if gait known   d >= kappa hw_T
  blame = the channel whose removal resolves it; if both do, the one with the larger
          half-width; if neither, "both" (one channel alone would not settle it).
VERIFICATION (only possible because the true scores exist): replace the implied
value of the blamed channel with the TRUE value and check the straddle really resolves.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

Q_KW = dict(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42, loss="quantile")


class CutoffStraddleExplainer:
    def __init__(self, alpha=0.10):
        self.a = alpha; self.m = {}; self.kappa = 1.0

    def fit(self, X, logT, logP):
        for name, yv in (("T", logT), ("P", logP)):
            lo = HistGradientBoostingRegressor(quantile=self.a / 2, **Q_KW).fit(X, yv)
            hi = HistGradientBoostingRegressor(quantile=1 - self.a / 2, **Q_KW).fit(X, yv)
            self.m[name] = (lo, hi)
        return self

    def half_widths(self, X):
        return {k: np.maximum((hi.predict(X) - lo.predict(X)) / 2, 1e-6) for k, (lo, hi) in self.m.items()}

    def calibrate_fidelity(self, X, l_hat, cut, ambiguous):
        """Choose kappa so that straddling reproduces the black-box set ambiguity."""
        hw = self.half_widths(X); d = np.abs(l_hat - cut); w = hw["T"] + hw["P"]
        grid = np.quantile(d / w, np.linspace(0.01, 0.99, 99))
        agree = [np.mean((d < k * w) == ambiguous) for k in grid]
        self.kappa = float(grid[int(np.argmax(agree))])
        return self

    def explain(self, X, l_hat, cut, point_T, point_P, true_T=None, true_P=None):
        hw = self.half_widths(X); k = self.kappa
        d = np.abs(l_hat - cut)
        straddle = d < k * (hw["T"] + hw["P"])
        res_T = straddle & (d >= k * hw["P"])
        res_P = straddle & (d >= k * hw["T"])
        blame = np.where(~straddle, "none",
                 np.where(res_T & ~res_P, "tremor",
                 np.where(res_P & ~res_T, "gait",
                 np.where(res_T & res_P, np.where(hw["T"] >= hw["P"], "tremor", "gait"), "both"))))
        out = dict(straddle=straddle, blame=blame, dist=d, hw_tremor=hw["T"], hw_gait=hw["P"],
                   tremor_share=hw["T"] / (hw["T"] + hw["P"]),
                   ratio_lo=l_hat - k * (hw["T"] + hw["P"]), ratio_hi=l_hat + k * (hw["T"] + hw["P"]))
        if true_T is not None:
            lt, lg = true_T - point_P, point_T - true_P          # ratio with ONE channel set to truth
            out["really_resolves_tremor"] = straddle & (np.abs(lt - cut) >= k * hw["P"])
            out["really_resolves_gait"] = straddle & (np.abs(lg - cut) >= k * hw["T"])
            out["covered_T"] = np.abs(true_T - point_T) <= k * hw["T"]
            out["covered_P"] = np.abs(true_P - point_P) <= k * hw["P"]
        return out
