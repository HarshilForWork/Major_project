"""Per-visit trajectory quantities for C3, computed on the FULL scheduled-visit sequence.

Every shift is taken over ALL scheduled visits of a patient, and only then are
unlabelled visits dropped. So "previous" / "next" always mean the ADJACENT scheduled
visit; if that visit has no label, the quantity is NaN rather than silently spanning a
gap (an earlier version shifted over labelled rows only, spanning gaps of up to 48 months
on 381 rows).
"""
import numpy as np
import pandas as pd
from . import formula as FM

DELTA_COLS = ["NP2RISE", "NP2TURN", "LEDD_TOTAL_MG", "MCATOT", "NP1RTOT", "NP1PTOT",
              "GDS_TOTAL", "SCOPA_AUT_TOTAL"]


def build(df):
    d = df.sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True).copy()
    d["logT"], d["logP"] = FM.log_scores(d.TREMOR_SCORE, d.PIGD_SCORE)
    d["ell"] = (d.logT - d.logP).where(d.LABEL.notna())
    g = d.groupby("PATNO")
    lp, ln = g.ell.shift(1), g.ell.shift(-1)
    mp, mn = g.VISIT_MONTH.shift(1), g.VISIT_MONTH.shift(-1)
    lam = (d.VISIT_MONTH - mp) / (mn - mp)                    # time-weighted interpolation
    a, b = 1 - lam, lam
    d["eta"] = a * lp + b * ln
    d["NOISE_FACTOR"] = 1 + a ** 2 + b ** 2                   # var(l - eta) / sigma^2
    d["ETA_NOISE_FACTOR"] = a ** 2 + b ** 2                   # var(noise inside eta) / sigma^2
    d["noise_abs"] = (d.ell - d.eta).abs()
    d["eta_next"] = d.groupby("PATNO").eta.shift(-1)
    d["mu"] = d.eta_next - d.eta
    for c in DELTA_COLS:
        d["D_" + c] = d[c] - g[c].shift(1)
    d["PRIOR_VISITS"] = g.cumcount()
    d["DIST_CUT"] = FM.distance_to_cutoff(d.ell)
    d["STATE_ON"] = (d.PDSTATE_USED == "ON").astype(float).where(d.PDSTATE_USED.notna())
    # --- history of the PREVIOUS scheduled visit (used by E5/E6) ------------------
    # Shifted here, before unlabelled rows are dropped, so "previous" is the adjacent
    # scheduled visit and is NaN when that visit carries no label.
    d["ell_prev"] = lp
    d["D_ell"] = d.ell - lp
    d["DIST_CUT_prev"] = FM.distance_to_cutoff(lp)
    # fresh groupby: STATE_ON was added after `g` was built
    d["STATE_ON_prev"] = d.groupby("PATNO").STATE_ON.shift(1)
    d["MONTHS_SINCE_PREV"] = d.VISIT_MONTH - mp

    d["L2"] = g.LABEL.shift(-2)                               # scheduled visit t+2
    d["SUSTAINED"] = np.where(d.LABEL_FLIPPED_NEXT.isna() | d.L2.isna(), np.nan,
                              ((d.LABEL_FLIPPED_NEXT == 1) & (d.L2 == d.NEXT_LABEL)).astype(float))
    d["REVERTED"] = np.where(d.L2.notna() & (d.LABEL_FLIPPED_NEXT == 1), (d.L2 == d.LABEL).astype(float), np.nan)
    return d[d.LABEL.notna()].reset_index(drop=True)
