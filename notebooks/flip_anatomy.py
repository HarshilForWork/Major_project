"""Flip anatomy: explain each observed subtype transition (visit t -> next visit).

Every flip is assigned ONE primary cause, checked in this order:
  1. EXAM-STATE ARTEFACT  the Part III exam's medication state differs between the two visits
                          (OFF <-> ON), and -- where PPMI recorded a same-day exam in the
                          other state -- recomputing the next label in the SAME state as
                          visit t removes the flip (observed counterfactual, "twin exam").
  2. THRESHOLD WOBBLE     not a state artefact, and the flip needs <= K item points of change
                          (points-to-flip distance at visit t), i.e. within rater noise.
  3. GENUINE CHANGE       everything else; decomposed exactly into a tremor path and a gait
                          path:  d log r = d log T - d log P.

Validation (not circular): artefacts and wobble should REVERT at the following visit far more
often than genuine flips.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ppmi_csv"
T3 = ["NP3PTRMR","NP3PTRML","NP3KTRMR","NP3KTRML","NP3RTARU","NP3RTALU","NP3RTARL","NP3RTALL","NP3RTALJ","NP3RTCON"]
P3 = ["NP3GAIT","NP3FRZGT","NP3PSTBL"]
T2, P2 = ["NP2TRMR"], ["NP2WALK","NP2FREZ"]
TREM, GAIT = T2 + T3, P2 + P3                     # 11 tremor, 5 gait items
ITEMS = TREM + GAIT
K_WOBBLE = 1                                      # points-to-flip threshold for "within noise"

def label_from(t_mean, p_mean):
    t_mean, p_mean = np.asarray(t_mean, float), np.asarray(p_mean, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p_mean > 0, t_mean / p_mean, np.where(t_mean > 0, np.inf, np.nan))
    out = np.where(r >= 1.15, "TD", np.where(r <= 0.90, "PIGD", "INDETERMINATE"))
    return np.where(np.isnan(r) & (t_mean == 0) & (p_mean == 0), "INDETERMINATE", out)

def lab(H):  return label_from(H[..., :11].mean(-1), H[..., 11:].mean(-1))

# --------------------------------------------------------------------------- data
d = pd.read_csv(ROOT / "data/processed/ppmi_tdpigd_long.csv", low_memory=False)
d = d.sort_values(["PATNO", "VISIT_MONTH"]).reset_index(drop=True)
for c in ["NEXT_LABEL", "PDSTATE_USED", "EVENT_ID"] + ITEMS:
    d["N_" + c] = d.groupby("PATNO")[c].shift(-1)
d["NN_LABEL"] = d.groupby("PATNO")["LABEL"].shift(-2)

# same-day twin exams (both OFF and ON recorded) from the raw Part III, missing codes cleaned
p3 = pd.read_csv(RAW / "MDS-UPDRS_Part_III_16Aug2026.csv", low_memory=False)
p3[T3 + P3] = p3[T3 + P3].apply(pd.to_numeric, errors="coerce").mask(lambda x: x > 4)
p2 = (pd.read_csv(RAW / "MDS_UPDRS_Part_II__Patient_Questionnaire_16Aug2026.csv", low_memory=False)
        [["PATNO", "EVENT_ID"] + T2 + P2].drop_duplicates(["PATNO", "EVENT_ID"]))
ex = p3[p3.PDSTATE.isin(["OFF", "ON"])].merge(p2, on=["PATNO", "EVENT_ID"])
ex = ex.dropna(subset=ITEMS).drop_duplicates(["PATNO", "EVENT_ID", "PDSTATE"])
ex["L"] = lab(ex[ITEMS].to_numpy(float))
twin = ex.pivot_table(index=["PATNO", "EVENT_ID"], columns="PDSTATE", values="L", aggfunc="first")

# --------------------------------------------------------------------------- flips
f = d[d.LABEL.notna() & d.N_NEXT_LABEL.notna() if False else d.LABEL.notna() & d.NEXT_LABEL.notna()].copy()
f["FLIP"] = (f.LABEL != f.NEXT_LABEL)
H0 = f[ITEMS].to_numpy(float); H1 = f[["N_" + c for c in ITEMS]].to_numpy(float)

# points-to-flip at visit t: fewest single-point item moves that reach the NEXT label
def points_to(Hrow, target, kmax=4):
    frontier = {tuple(Hrow)}
    for k in range(1, kmax + 1):
        nxt = set()
        for h in frontier:
            for j in range(16):
                for s in (-1, 1):
                    v = list(h); v[j] = min(4, max(0, v[j] + s)); nxt.add(tuple(v))
        arr = np.array(list(nxt))
        if (lab(arr) == target).any(): return k
        frontier = nxt if len(nxt) < 20000 else set(map(tuple, arr[np.random.default_rng(k).choice(len(arr), 20000, replace=False)]))
    return kmax + 1
fl = f[f.FLIP].copy(); Hf0 = fl[ITEMS].to_numpy(float)
fl["PTS_TO_FLIP"] = [points_to(h, t) for h, t in zip(Hf0, fl.NEXT_LABEL)]
fl["OBS_POINTS"] = np.abs(fl[["N_" + c for c in ITEMS]].to_numpy(float) - Hf0).sum(1)

# exam-state artefact, confirmed with the twin exam where one exists
st0, st1 = fl.PDSTATE_USED, fl.N_PDSTATE_USED
fl["STATE_CHANGED"] = st0.notna() & st1.notna() & (st0 != st1)
def matched_next(row):
    key = (row.PATNO, row.N_EVENT_ID)
    if not row.STATE_CHANGED or key not in twin.index: return np.nan
    v = twin.loc[key].get(row.PDSTATE_USED, np.nan)
    return v if isinstance(v, str) else np.nan
fl["NEXT_LABEL_MATCHED"] = fl.apply(matched_next, axis=1)
fl["TWIN_AVAILABLE"] = fl.NEXT_LABEL_MATCHED.notna()
fl["STATE_CONFIRMED"] = fl.TWIN_AVAILABLE & (fl.NEXT_LABEL_MATCHED == fl.LABEL)

# genuine-change decomposition: d log r = d log T - d log P   (+0.5 point smoothing for zeros)
eps = 0.5 / 16
Tm0, Pm0 = Hf0[:, :11].mean(1), Hf0[:, 11:].mean(1)
Hf1 = fl[["N_" + c for c in ITEMS]].to_numpy(float)
Tm1, Pm1 = Hf1[:, :11].mean(1), Hf1[:, 11:].mean(1)
fl["dlogT"] = np.log(Tm1 + eps) - np.log(Tm0 + eps)
fl["dlogP"] = np.log(Pm1 + eps) - np.log(Pm0 + eps)
fl["PATH"] = np.where(np.abs(fl.dlogT) >= np.abs(fl.dlogP), "tremor path", "gait path")

# primary cause, in priority order
cause = np.where(fl.STATE_CONFIRMED, "exam-state artefact (twin-confirmed)",
         np.where(fl.STATE_CHANGED & ~fl.TWIN_AVAILABLE, "exam-state change (unconfirmed)",
         np.where(fl.PTS_TO_FLIP <= K_WOBBLE, "threshold wobble",
                  "genuine: " + fl.PATH)))
fl["CAUSE"] = cause

# --------------------------------------------------------------------------- report
n_pairs, n_flip = len(f), len(fl)
print(f"visit pairs {n_pairs:,} | flips {n_flip:,} ({n_flip/n_pairs*100:.1f}%)")
print(f"state changed between visits: {fl.STATE_CHANGED.mean()*100:.1f}% of flips | twin exam available for "
      f"{fl.TWIN_AVAILABLE.sum()} of those; flip disappears in matched state: "
      f"{fl.STATE_CONFIRMED.sum()} ({fl.STATE_CONFIRMED.sum()/max(fl.TWIN_AVAILABLE.sum(),1)*100:.0f}%)")
print(f"points-to-flip at visit t: " + ", ".join(f"{k}:{(fl.PTS_TO_FLIP==k).mean()*100:.0f}%" for k in range(1,6)))
print(f"observed item-point change on flips: median {fl.OBS_POINTS.median():.0f} | "
      f"non-flip pairs median {np.abs(H1[~f.FLIP.to_numpy()]-H0[~f.FLIP.to_numpy()]).sum(1).__array__().__getitem__(slice(None)).tolist() and np.nanmedian(np.abs(H1[~f.FLIP.to_numpy()]-H0[~f.FLIP.to_numpy()]).sum(1)):.0f}")

rev = fl.NN_LABEL.notna()
fl["REVERTED"] = np.where(rev, fl.NN_LABEL == fl.LABEL, np.nan)
tab = (fl.groupby("CAUSE").agg(n=("CAUSE", "size"),
                               share=("CAUSE", lambda x: len(x) / n_flip * 100),
                               reverts_next_visit=("REVERTED", lambda x: np.nanmean(x) * 100),
                               n_with_followup=("REVERTED", lambda x: int(np.isfinite(x.astype(float)).sum())))
         .sort_values("share", ascending=False).round(1))
print("\nPRIMARY CAUSE OF EACH FLIP")
print(tab.to_string())
art = fl.CAUSE.str.startswith(("exam-state", "threshold"))
print(f"\nartefact-type (state + wobble): {art.mean()*100:.1f}% of flips | reversion {np.nanmean(fl.loc[art,'REVERTED'].astype(float))*100:.1f}%"
      f"  vs genuine {np.nanmean(fl.loc[~art,'REVERTED'].astype(float))*100:.1f}%")
print("\ndirection x cause (share of flips):")
fl["DIR"] = fl.LABEL.str[:4] + "→" + fl.NEXT_LABEL.str[:4]
print(pd.crosstab(fl.DIR, fl.CAUSE, normalize=False).to_string())
out = ROOT / "data/interim/flip_anatomy.csv"
fl[["PATNO","EVENT_ID","N_EVENT_ID","VISIT_MONTH","LABEL","NEXT_LABEL","PDSTATE_USED","N_PDSTATE_USED",
    "TWIN_AVAILABLE","NEXT_LABEL_MATCHED","PTS_TO_FLIP","OBS_POINTS","dlogT","dlogP","PATH","CAUSE","REVERTED"]].to_csv(out, index=False)
print(f"\nper-flip table -> {out.relative_to(ROOT)}")
