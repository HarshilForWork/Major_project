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
num = lambda pat: float(re.search(pat, txt).group(1))
imp = pd.read_csv(INT / "fcx_c1_importance.csv", index_col=0).head(10)[::-1]
BLUE, GREEN, AMBER, GREY, NAVY = "#2a6ea8", "#2e7d4f", "#c98b2b", "#8a8a8a", "#14375e"

fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.45))
# ---- C1
a = ax[0]
a.barh(imp.index, imp.tremor, color=BLUE, label="tremor channel")
a.barh(imp.index, imp.gait, left=imp.tremor, color=GREEN, label="gait channel")
a.set_xlabel("mean |attribution| to the implied log-ratio")
a.set_title("C1 · Implied exam\nwhat moves the classifier,\nsplit by formula channel", fontweight="bold", color=NAVY)
a.legend(frameon=False, loc="lower right")
fid = num(r"agrees with the classifier: ([0-9.]+)%"); alT = num(r"log T ([0-9.]+) \|"); alP = num(r"log P ([0-9.]+) \|")
a.text(0.98, 0.30, f"fidelity to classifier {fid:.1f}%\nalignment with TRUE scores:\n  gait r = {alP:.2f}\n  tremor r = {alT:.2f}  ← model is\n  nearly blind to tremor",
       transform=a.transAxes, ha="right", va="bottom", fontsize=8.8,
       bbox=dict(fc="#f7f7f7", ec=GREY))
for s in ("top", "right"): a.spines[s].set_visible(False)
# ---- C2
a = ax[1]
m = re.search(r"FCX ([0-9]+)% \| grouped Shapley on classifier ([0-9]+)% \| random channel ([0-9]+)%", txt)
vals = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
bars = a.bar(["FCX\nstraddle blame", "grouped Shapley\non classifier", "random\nchannel"], vals, color=[AMBER, GREY, "#cccccc"])
for bb, v in zip(bars, vals): a.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
a.set_ylim(0, 100); a.set_ylabel("% of ambiguous sets resolved when the\nblamed channel's TRUE score is revealed")
kap = num(r"kappa ([0-9.]+) \| ambiguous"); agr = num(r"agreement ([0-9.]+)% \| kappa")
a.set_title(f"C2 · Why is the set {{TD, PIGD}}?\nstraddle matches the conformal set\n{agr:.0f}% of the time (κ = {kap:.2f})", fontweight="bold", color=NAVY)
for s in ("top", "right"): a.spines[s].set_visible(False)
# ---- C3
a = ax[2]
m = re.search(r"proximity ([0-9]+)% \| drift ([0-9]+)% \| noise ([0-9]+)%", txt)
sh = [float(m.group(i)) for i in (1, 2, 3)]
a.bar(["proximity\nto cutoff", "real drift", "exam noise"], sh, color=[BLUE, GREEN, AMBER])
for i, v in enumerate(sh): a.text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
auc = num(r"black-box transition model: AUROC ([0-9.]+)")
a.set_ylim(0, 100); a.set_ylabel("share of predicted transition risk")
a.set_title(f"C3 · What transition risk is made of\nblack-box AUROC {auc:.2f}:\nalmost no real-drift signal", fontweight="bold", color=NAVY)
for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("FCX — Formula-Coordinate Explanations of the TRACE-PD models", fontsize=14, fontweight="bold", color=NAVY, y=1.10)
fig.savefig(FIG / "fig_fcx_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(FIG / "fig_fcx_summary.pdf", bbox_inches="tight", facecolor="white")
print("wrote fig_fcx_summary.png / .pdf")
