"""AR(1)-SHAP, part 2 -- Shapley values with a pluggable "how do we fill hidden features".

One estimator, three fillers, so any difference between them comes from the filler
and not from the estimator:

  InterventionalFiller  standard SHAP: hidden features copied from a random real visit
                        of another patient (the briefing's Eq. 1)
  PopulationFiller      ablation: hidden features drawn from the POPULATION conditional
                        P(X_hidden | X_shown) -- a Gaussian copula over all 26 features,
                        no patient history (the Aas et al. 2021 style of conditional SHAP)
  AR1Filler             the method: hidden features drawn from
                        P(X_hidden | X_shown, X_{t-1}, dt)  (the briefing's Eq. 3)

Value function v(S) = E[f(x_S, X_hidden)] is estimated by n_mc draws, re-drawn at every
step of every permutation because the conditioning set changes at every step. Within one
permutation the same random numbers are reused at every step (common random numbers), so
the step-to-step differences that become Shapley values carry less sampling noise. Antithetic
permutation sampling; efficiency holds exactly:  sum_j phi_j = f(x) - v_hat(empty).

Features missing at the visit being explained are not players: they stay missing in
every filled row and get zero credit, under all three fillers.

What v(empty) means differs:
  interventional / population   v(empty) = average prediction over the cohort
  AR1                           v(empty) = E[f(X_t) | X_{t-1}] = what this patient's last
                                visit predicts for today
so AR(1)-SHAP explains "why is today's prediction different from what we expected for
THIS patient", not "why is it different from the average patient".
"""
import numpy as np

from .ar1 import CopulaTransform, nearest_psd, pairwise_cov, shrink


def gauss_conditional_draw(mu, Sig, known, known_vals, hidden, n, rng, noise=None):
    """n draws of the hidden block of N(mu, Sig) given the known block.

    noise: optional (n, len(mu)) standard-normal matrix shared across the coalitions of
    one permutation (common random numbers); column j is always feature j's noise, so
    successive coalitions are positively correlated and their difference is less noisy.
    Each single draw still has exactly the right distribution."""
    h = np.asarray(hidden, int)
    if len(h) == 0:
        return np.zeros((n, 0))
    k = np.asarray(known, int)
    if len(k) == 0:
        m, C = mu[h], Sig[np.ix_(h, h)]
    else:
        Skk, Shk = Sig[np.ix_(k, k)], Sig[np.ix_(h, k)]
        K = np.linalg.solve(Skk, Shk.T).T
        m = mu[h] + K @ (known_vals - mu[k])
        C = Sig[np.ix_(h, h)] - K @ Shk.T
    C = (C + C.T) / 2 + 1e-9 * np.eye(len(h))
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(nearest_psd(C))
    E = rng.standard_normal((n, len(h))) if noise is None else noise[:, h]
    return m + E @ L.T


# ------------------------------------------------------------------------- fillers

class InterventionalFiller:
    """Hidden features copied from one random background visit (standard SHAP)."""
    name = "interventional"

    def __init__(self, background):
        self.B = np.asarray(background, float)

    def for_instance(self, x, ctx=None):
        return self

    def chain_state(self, n, rng):
        return rng.integers(0, len(self.B), size=n)        # same background rows all chain

    def fill(self, x, shown, n, rng, state=None):
        idx = rng.integers(0, len(self.B), size=n) if state is None else state
        rows = self.B[idx].copy()
        rows[:, shown] = x[shown]
        return rows


class PopulationFiller:
    """Hidden features from the population conditional (Gaussian copula, no history)."""
    name = "population"

    def __init__(self, visits_X, lam=0.1):
        X = np.asarray(visits_X, float)
        self.cop = CopulaTransform().fit(X)
        U = self.cop.to_normal(X)
        self.mu = np.nanmean(U, 0)
        self.Sig = nearest_psd(shrink(pairwise_cov(U - self.mu), lam))

    def for_instance(self, x, ctx=None):
        self.u = self.cop.to_normal(x[None, :])[0]
        return self

    def chain_state(self, n, rng):
        return rng.standard_normal((n, len(self.mu)))

    def fill(self, x, shown, n, rng, state=None):
        rows = np.tile(x, (n, 1))
        hidden = np.flatnonzero(~shown)
        known = np.flatnonzero(shown & ~np.isnan(self.u))
        Ud = gauss_conditional_draw(self.mu, self.Sig, known, self.u[known], hidden, n, rng, state)
        for c, j in enumerate(hidden):
            rows[:, j] = self.cop.from_normal_col(Ud[:, c], j)
        return rows


class AR1Filler:
    """Hidden features from this patient's own next-visit model.

    Static and clock features are known exactly given the previous visit and the visit
    date, so hiding them changes nothing (they receive exactly zero credit). Dynamic
    hidden features are drawn from N(mu_i, Sigma(dt_i)) conditioned on the dynamic
    features that are shown, where mu_i comes from the previous visit.
    """
    name = "ar1"

    def __init__(self, tm, features):
        self.tm = tm
        self.features = list(features)
        self.dyn_cols = np.array([self.features.index(f) for f in tm.dynamic])
        self.inert_cols = [self.features.index(f) for f in tm.static + tm.clock]

    def for_instance(self, x, ctx):
        """ctx: one-row pair frame for this visit (PREV_*, STALE_*, DT, current values)."""
        self.mu = self.tm.predict_mean(ctx)[0]
        self.Sig = self.tm.sigma_for(float(ctx.DT.iloc[0]))
        self.u = self.tm.cop_dyn.to_normal(x[self.dyn_cols][None, :])[0]
        return self

    def chain_state(self, n, rng):
        return rng.standard_normal((n, len(self.mu)))

    def fill(self, x, shown, n, rng, state=None):
        rows = np.tile(x, (n, 1))
        shown_d = shown[self.dyn_cols]
        hidden = np.flatnonzero(~shown_d)
        known = np.flatnonzero(shown_d & ~np.isnan(self.u))
        Ud = gauss_conditional_draw(self.mu, self.Sig, known, self.u[known], hidden, n, rng, state)
        for c, j in enumerate(hidden):
            rows[:, self.dyn_cols[j]] = self.tm.cop_dyn.from_normal_col(Ud[:, c], j)
        return rows


# ----------------------------------------------------------------------- estimator

def shapley_instance(f, x, filler, n_perm=32, n_mc=24, rng=None):
    """Permutation Shapley for ONE instance x (1-D, NaN = not recorded).

    Returns (phi over all columns, v_empty). Non-recorded columns get phi = 0.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, float)
    d = len(x)
    players = np.flatnonzero(~np.isnan(x))
    p = len(players)
    phi = np.zeros(d)
    if p == 0:
        return phi, float(f(x[None, :])[0])
    fx = float(f(x[None, :])[0])

    # Columns whose reveal cannot change the filled rows (AR1: static and clock) reuse
    # the previous coalition's block, so their credit is exactly 0, not 0 +/- MC noise.
    inert = set(getattr(filler, "inert_cols", ()))
    blocks, chains = [], []
    for k in range(n_perm):
        perm = rng.permutation(p) if k % 2 == 0 else perm[::-1]
        shown = np.isnan(x).copy()                  # non-recorded columns are fixed (missing)
        st = filler.chain_state(n_mc, rng) if hasattr(filler, "chain_state") else None
        blocks.append(filler.fill(x, shown, n_mc, rng, st))
        chain = [len(blocks) - 1]                   # block index of v(S_0), v(S_1), ...
        for pl in perm[:-1]:
            shown[players[pl]] = True
            if players[pl] in inert:
                chain.append(chain[-1])
            else:
                blocks.append(filler.fill(x, shown, n_mc, rng, st))
                chain.append(len(blocks) - 1)
        chains.append((perm, chain))
    vals = f(np.vstack(blocks)).reshape(-1, n_mc).mean(1)     # one value per block

    v_empty = 0.0
    for perm, chain in chains:
        v = np.append(vals[chain], fx)                         # v(S_0) ... v(S_{p-1}), v(all)
        phi[players[perm]] += np.diff(v)
        v_empty += v[0]
    return phi / n_perm, v_empty / n_perm


def explain_many(f, X, filler, contexts=None, n_perm=32, n_mc=24, seed=0, progress=None):
    """Explain every row of X. contexts[i] is passed to filler.for_instance."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    Phi = np.zeros(X.shape); base = np.zeros(len(X))
    for i in range(len(X)):
        filler.for_instance(X[i], None if contexts is None else contexts[i])
        Phi[i], base[i] = shapley_instance(f, X[i], filler, n_perm, n_mc, rng)
        if progress and (i + 1) % progress == 0:
            print(f"    {filler.name}: {i + 1}/{len(X)}", flush=True)
    return Phi, base
