"""FCX summary figure: one panel per component. Every number is read from the
evaluation outputs (reports/metrics/fcx_results.txt, data/interim/fcx_*.csv)."""
from pathlib import Path as _Path
import re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_ROOT = _Path(__file__).resolve().parents[3]
INT, MET, FIG = _ROOT / "data/interim", _ROOT / "reports/metrics", _ROOT / "reports/figures"
import sys
_x = MET / "fcx_results_xgb.txt"
SRC = _x if ("--backend" not in sys.argv and _x.exists()) or ("xgb" in sys.argv) else MET / "fcx_results.txt"
txt = SRC.read_text(encoding="utf-8")
print(f"reading {SRC.name}")
SFX = "_xgb" if SRC.name.endswith("_xgb.txt") else ""
def num(pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else float("nan")
imp = pd.read_csv(INT / f"fcx_c1_importance{SFX}.csv", index_col=0).head(10)[::-1]
BLUE, GREEN, AMBER, GREY, NAVY = "#2a6ea8", "#2e7d4f", "#c98b2b", "#8a8a8a", "#14375e"

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.45))
# ---- C1
a = ax[0]
a.barh(imp.index, imp.tremor, color=BLUE, label="tremor channel")
a.barh(imp.index, imp.gait, left=imp.tremor, color=GREEN, label="gait channel")
a.set_xlabel("mean |attribution| to the implied log-ratio")
a.set_title("C1 · Implied exam\nwhat moves the classifier,\nsplit by formula channel", fontweight="bold", color=NAVY)
a.legend(frameon=False, loc="lower right")
fid = num(r"agrees with the classifier: (-?[0-9.]+)%"); alT = num(r"log T ([0-9.]+) \|"); alP = num(r"log P ([0-9.]+) \|")
a.text(0.98, 0.30, f"fidelity to classifier {fid:.1f}%\nimplied score vs TRUE score:\n  gait r = {alP:.2f}\n  tremor r = {alT:.2f}\n(the cheap features carry\nlittle tremor information)",
       transform=a.transAxes, ha="right", va="bottom", fontsize=8.8,
       bbox=dict(fc="#f7f7f7", ec=GREY))
for s in ("top", "right"): a.spines[s].set_visible(False)
# ---- C2
a = ax[1]
m = re.search(r"REAL vs BASELINE.*?FCX blame collapses the set ([0-9]+)% \| grouped Shapley blame ([0-9]+)% \| random ([0-9]+)%", txt)
vals = [float(m.group(i)) for i in (1, 2, 3)] if m else [np.nan] * 3
vals.append(num(r"'always examine tremor' on the same sets: ([0-9]+)%"))
bars = a.bar(["FCX\nper-patient\nblame", "grouped\nShapley", "random\nchannel", "always\ntremor"], vals,
             color=[AMBER, GREY, "#cccccc", BLUE])
for bb, v in zip(bars, vals):
    if np.isfinite(v): a.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
a.set_ylim(0, 100); a.set_ylabel("% of ambiguous conformal sets that COLLAPSE when\nthat channel's true exam score is given")
kap = num(r"kappa ([0-9.]+) \| ambiguous"); agr = num(r"agreement ([0-9.]+)% \| kappa")
a.set_title(f"C2 · Why is the set {{TD, PIGD}}?\nstraddle reproduces the set {agr:.0f}% (κ = {kap:.2f});\nbut 'always tremor' beats per-patient blame", fontweight="bold", color=NAVY)
for s in ("top", "right"): a.spines[s].set_visible(False)
# ---- C3
a = ax[2]
m = re.search(r"proximity ([0-9]+)% \| drift ([0-9]+)% \| noise ([0-9]+)%", txt)
sh = [float(m.group(i)) for i in (1, 2, 3)] if m else [np.nan] * 3
r2c3 = num(r"surrogate R\^2 (-?[0-9.]+)\)")
a.bar(["proximity\nto cutoff", "real drift", "exam noise"], sh, color=[BLUE, GREEN, AMBER])
for i, v in enumerate(sh): a.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
auc = num(r"black-box transition model: AUROC ([0-9.]+)")
a.set_ylim(0, 100); a.set_ylabel("share of predicted transition risk")
a.set_title(f"C3 · What transition risk is made of\ncheap-feature model: AUROC {auc:.2f},\nsurrogate R² {r2c3:.2f}", fontweight="bold", color=NAVY)
for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FCX — Formula-Coordinate Explanations of the TRACE-PD models", fontsize=14, fontweight="bold", color=NAVY, y=1.10)
fig.savefig(FIG / "fig_fcx_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / "fig_fcx_summary.pdf", bbox_inches="tight", facecolor="white")
print("wrote fig_fcx_summary.png / .pdf")
