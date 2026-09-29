"""C3 -- Proximity / Drift / Noise explanation of the transition-risk model.

In formula coordinates a subtype flip between visit t and t+1 can only come from:
  PROXIMITY  the patient's underlying log-ratio eta sits near a cutoff
  DRIFT      eta itself moves (mu = expected change to the next visit)
  NOISE      the observed l scatters around eta from visit to visit (sigma)

Structural surrogate (first-passage form):
  eta ~ N(eta_hat, tau^2),  l_t ~ N(eta, sigma^2),  l_{t+1} ~ N(eta + mu, sigma^2)
  p_pdn = E_eta [ 1 - sum_z  pi_z(eta, sigma) * pi_z(eta + mu, sigma) ]
where pi_z(u, s) is the probability that N(u, s^2) lies in subtype zone z.

Heads (trained on cheap features against targets built from the TRUE log-ratio):
  eta_hat(x): smoothed true l (mean of the neighbouring visits -> leave-visit-out)
  mu_hat(x) : change of the smoothed l to the next visit
  sigma(x)  : |l - eta| * sqrt(pi/2)   (mean-absolute-deviation -> SD)
  tau       : residual SD of eta_hat

Faithfulness to the black-box transition model g3 -- COMPONENT-CALIBRATED:
  1. exact 3-player Shapley of the SURROGATE itself on logit p_pdn gives s_P, s_D, s_N
     (heads in S at the patient's value, the others at a cohort reference)
  2. regress the black box on those components:
        logit g3(x) ~= alpha + b_P s_P(x) + b_D s_D(x) + b_N s_N(x)
  3. black-box attribution:  phi_c = b_c * s_c(x)
The black box now enters through one weight PER component, so two models that rely on
different mechanisms get different explanations (checked by the planted-model test in
evaluation/validate_c3.py). An earlier single-slope version, logit g3 ~ a + b logit p_pdn,
gave nearly identical splits for any black box -- the shares were a property of the
surrogate, not of the model being explained.
Efficiency: phi_P + phi_D + phi_N = prediction - alpha - sum_c b_c s_c(reference) = pred - base.
"""
import itertools
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from . import formula as F

KW = dict(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(20)


def zone_probs(u, s):
    a, b = F.CUT_PIGD, F.CUT_TD
    pa, pb = norm.cdf((a - u) / s), norm.cdf((b - u) / s)
    return pa, pb - pa, 1 - pb                         # PIGD, IND, TD


def flip_prob(eta, tau, mu, sigma):
    eta, tau, mu, sigma = (np.asarray(v, float)[..., None] for v in (eta, tau, mu, sigma))
    e = eta + np.sqrt(2) * tau * _GH_X                 # quadrature over eta
    z0, z1 = zone_probs(e, sigma), zone_probs(e + mu, sigma)
    stay = sum(p0 * p1 for p0, p1 in zip(z0, z1))
    return np.clip(((1 - stay) * _GH_W).sum(-1) / np.sqrt(np.pi), 1e-6, 1 - 1e-6)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PDNExplainer:
    def __init__(self):
        self.h_eta = HistGradientBoostingRegressor(**KW)
        self.h_mu = HistGradientBoostingRegressor(**KW)
        self.h_sig = HistGradientBoostingRegressor(**KW)
        self.tau = None; self.alpha = 0.0; self.beta = 1.0; self.ref = None

    def fit(self, X, eta, mu, abs_noise):
        m_eta, m_mu, m_sig = np.isfinite(eta), np.isfinite(mu), np.isfinite(abs_noise)
        self.h_eta.fit(X[m_eta], eta[m_eta])
        self.tau = float(np.std(eta[m_eta] - self.h_eta.predict(X[m_eta])))
        self.h_mu.fit(X[m_mu], mu[m_mu])
        self.h_sig.fit(X[m_sig], abs_noise[m_sig] * np.sqrt(np.pi / 2))
        return self

    def heads(self, X):
        return dict(eta=self.h_eta.predict(X), tau=np.full(len(X), self.tau),
                    mu=self.h_mu.predict(X), sigma=np.clip(self.h_sig.predict(X), 0.05, None))

    def p_pdn(self, X=None, H=None):
        H = self.heads(X) if H is None else H
        return flip_prob(H["eta"], H["tau"], H["mu"], H["sigma"])

    def _set_reference(self, H):
        self.ref = dict(eta=float(np.median(H["eta"])), tau=self.tau, mu=0.0,
                        sigma=float(np.median(H["sigma"])))

    def surrogate_components(self, X=None, H=None):
        """Exact Shapley of logit p_pdn over players P=(eta,tau), D=(mu), N=(sigma)."""
        H = self.heads(X) if H is None else H; n = len(H["eta"])
        members = {"P": ("eta", "tau"), "D": ("mu",), "N": ("sigma",)}
        cache = {}
        def v(S):
            key = frozenset(S)
            if key not in cache:
                h = {k: (H[k] if any(k in members[p] for p in key) else np.full(n, self.ref[k]))
                     for k in ("eta", "tau", "mu", "sigma")}
                cache[key] = logit(self.p_pdn(H=h))
            return cache[key]
        from math import factorial
        players = ["P", "D", "N"]; s = {p: np.zeros(n) for p in players}
        for p in players:
            others = [q for q in players if q != p]
            for k in range(3):
                for S in itertools.combinations(others, k):
                    w = factorial(k) * factorial(3 - k - 1) / factorial(3)
                    s[p] += w * (v(set(S) | {p}) - v(set(S)))
        return s, v(set())

    def calibrate_fidelity(self, X, g3_prob):
        """logit g3 ~ alpha + b_P s_P + b_D s_D + b_N s_N  (least squares on training rows)."""
        H = self.heads(X); self._set_reference(H)
        s, _ = self.surrogate_components(H=H)
        A = np.c_[np.ones(len(X)), s["P"], s["D"], s["N"]]
        coef = np.linalg.lstsq(A, logit(g3_prob), rcond=None)[0]
        self.alpha, self.b = float(coef[0]), dict(P=float(coef[1]), D=float(coef[2]), N=float(coef[3]))
        return self

    def predict_logit(self, X=None, H=None):
        s, _ = self.surrogate_components(X, H)
        return self.alpha + sum(self.b[c] * s[c] for c in ("P", "D", "N"))

    def explain(self, X):
        H = self.heads(X)
        s, _ = self.surrogate_components(H=H)
        phi = {c: self.b[c] * s[c] for c in ("P", "D", "N")}
        full = self.alpha + sum(phi.values())
        return dict(phi_P=phi["P"], phi_D=phi["D"], phi_N=phi["N"], base=np.full(len(X), self.alpha),
                    full=full, weights=dict(self.b), **H)

    @staticmethod
    def narrative(e, i, p_model):
        parts = {"proximity to a subtype cutoff": e["phi_P"][i], "expected real drift": e["phi_D"][i],
                 "visit-to-visit measurement noise": e["phi_N"][i]}
        detail = "; ".join(f"{k} {v:+.2f}" for k, v in parts.items())
        dist = F.distance_to_cutoff(e["eta"][i])
        up = {k: v for k, v in parts.items() if v > 0}
        if not up:
            return (f"Transition risk {p_model:.0%}. Every component is below the cohort reference "
                    f"({detail}); the underlying ratio is {dist:.2f} log-units from the nearest cutoff. "
                    f"Suggested action: routine follow-up.")
        main = max(up, key=up.get)
        advice = ("follow up: the change looks progressive" if main == "expected real drift"
                  else "re-examine (ideally OFF medication) before re-subtyping")
        return (f"Transition risk {p_model:.0%}. Main factor raising it: {main}. Contributions (logit, vs cohort "
                f"reference): {detail}. Underlying ratio is {dist:.2f} log-units from the nearest cutoff. "
                f"Suggested action: {advice}.")
