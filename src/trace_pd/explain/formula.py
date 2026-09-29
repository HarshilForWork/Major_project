"""The Stebbins (2013) TD/PIGD rule, written in log-ratio coordinates.

T = mean of 11 tremor items, P = mean of 5 gait items (0-4 each)
l = log(T + eps) - log(P + eps)
TD            if  l >= log(1.15)
PIGD          if  l <= log(0.90)
INDETERMINATE otherwise

eps keeps the log finite when a score is 0. With eps = 0.5/16 (half an item point
spread over 16 items) the zone rule reproduces the stored labels on the data --
checked in tests/test_fcx.py.
"""
import numpy as np

EPS = 0.5 / 16
CUT_PIGD = float(np.log(0.90))      # l <= this  -> PIGD
CUT_TD = float(np.log(1.15))        # l >= this  -> TD
TREMOR_ITEMS = ["NP2TRMR", "NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML",
                "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"]
GAIT_ITEMS = ["NP2WALK", "NP2FREZ", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"]


def log_scores(tremor_mean, gait_mean):
    return np.log(np.asarray(tremor_mean, float) + EPS), np.log(np.asarray(gait_mean, float) + EPS)


def zone(l):
    """0 = PIGD, 1 = INDETERMINATE, 2 = TD (ordered along l)."""
    l = np.asarray(l, float)
    return np.where(l <= CUT_PIGD, 0, np.where(l >= CUT_TD, 2, 1))


ZONE_NAMES = np.array(["PIGD", "INDETERMINATE", "TD"])


def distance_to_cutoff(l):
    """Signed distance (log units) from l to the nearest cutoff. Positive = inside its zone
    by that margin; the explanation reports how far the patient is from switching subtype."""
    l = np.asarray(l, float)
    return np.minimum(np.abs(l - CUT_PIGD), np.abs(l - CUT_TD))
