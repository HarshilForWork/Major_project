"""Interventional Shapley values -- own implementation, no SHAP dependency.

For a value function f(z) and a single instance x with background rows B:
    phi_j = E_perm E_b [ f(z_{S u j}) - f(z_S) ],  z_S = x on S, b elsewhere.
Estimated with antithetic permutation sampling; exact for small player sets.
Efficiency holds by construction:  sum_j phi_j = f(x) - E_b f(b).
"""
import itertools
import numpy as np


def shapley_mc(f, X, B, n_perm=32, groups=None, rng=None):
    """Monte-Carlo interventional Shapley for many instances at once.

    f       vectorised function: (n, d) array -> (n,) array
    X       (n, d) instances to explain
    B       (m, d) background rows
    groups  optional list of column-index lists; players are groups instead of columns
    returns (n, p) attributions and an (n,) base value = mean of f over the background
    rows actually drawn for that instance, so efficiency is exact per instance:
        phi[i].sum() == f(X[i]) - base[i]
    """
    rng = np.random.default_rng(0) if rng is None else rng
    X, B = np.asarray(X, float), np.asarray(B, float)
    n, d = X.shape
    players = [[j] for j in range(d)] if groups is None else [list(g) for g in groups]
    p = len(players)
    phi = np.zeros((n, p)); base = np.zeros(n)
    for k in range(n_perm):
        perm = rng.permutation(p) if k % 2 == 0 else perm[::-1]          # antithetic pairs
        b = B[rng.integers(0, len(B), size=n)]                            # one background row per instance
        z = b.copy()
        prev = f(z); base += prev
        for pl in perm:
            z[:, players[pl]] = X[:, players[pl]]
            cur = f(z)
            phi[:, pl] += cur - prev
            prev = cur
    phi /= n_perm; base /= n_perm
    return phi, base


def shapley_exact(v, players):
    """Exact Shapley for a handful of players. v(frozenset) -> value (array or float)."""
    players = list(players); p = len(players)
    from math import factorial
    out = {q: 0.0 for q in players}
    for q in players:
        others = [r for r in players if r != q]
        for k in range(p):
            for S in itertools.combinations(others, k):
                w = factorial(k) * factorial(p - k - 1) / factorial(p)
                S = frozenset(S)
                out[q] = out[q] + w * (v(S | {q}) - v(S))
    return out
