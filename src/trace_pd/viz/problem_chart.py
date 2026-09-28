import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
_RAW       = _ROOT / "data" / "raw" / "ppmi_csv"
_PROCESSED = _ROOT / "data" / "processed"
_METRICS   = _ROOT / "reports" / "metrics"
_FIGURES   = _ROOT / "reports" / "figures"
for _d in (_PROCESSED, _METRICS, _FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


NAVY  = "#14375e"
BLUE  = "#2a6ea8"
TEAL  = "#1d7373"
AMBER = "#b8791c"
RED   = "#c0272d"
GREY  = "#6a6a6a"
BAND  = "#fdf3e3"

fig, ax = plt.subplots(figsize=(12.6, 6.2))
fig.patch.set_facecolor("white")
ax.set_position([0.095, 0.315, 0.875, 0.545])

fig.text(0.045, 0.945, "The Problem", fontsize=21, fontweight="bold", color=NAVY)
fig.text(0.045, 0.885,
         "Diagnostic certainty accumulates one visit at a time — the subtype stays provisional for over a year",
         fontsize=11, color=GREY)

# ---- axes ----
ax.set_xlim(-0.7, 16.4); ax.set_ylim(0, 108)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(GREY); ax.spines[sp].set_linewidth(1.2)

xs     = np.array([0, 2, 4, 6, 9, 12, 15])
vlabel = ["Visit 1", "Visit 2", "Visit 3", "Visit 4", "Visit 5", "Visit 6", "Visit 7+"]
mlabel = ["Month 0", "Month 2", "Month 4", "Month 6", "Month 9", "Month 12", "> 12 months"]
ax.set_xticks(xs)
ax.set_xticklabels([f"{v}\n{m}" for v, m in zip(vlabel, mlabel)], fontsize=9.6, color="#333333")
ax.set_yticks([]); ax.set_xlabel("Time since first clinic visit", fontsize=11,
                                 color="#333333", labelpad=9)
for lab in ax.get_xticklabels():
    lab.set_linespacing(1.5)

ax.text(-2.0, 92, "Confident\nenough to act", fontsize=9.5, color=TEAL,
        ha="center", va="center")
ax.text(-2.0, 22, "Still\nprovisional", fontsize=9.5, color=AMBER,
        ha="center", va="center")
ax.text(-4.4, 58, "Diagnostic\ncertainty", fontsize=12, fontweight="bold",
        color="#222222", ha="center", va="center")
ax.text(-4.4, 45, "(is this TD\nor PIGD?)", fontsize=8.8, color=GREY,
        ha="center", va="center")

# ---- provisional band + threshold ----
ax.add_patch(Rectangle((-0.7, 0), 17.1, 80, fc=BAND, ec="none", zorder=0))
ax.axhline(80, color=TEAL, lw=1.6, ls=(0, (6, 4)), zorder=2)
ax.text(-0.4, 83.5, "confident-subtype threshold", fontsize=9.5, color=TEAL,
        ha="left", va="bottom", fontweight="bold")

# ---- specialist-led trajectory ----
ys = np.array([18, 30, 42, 53, 64, 76, 90])
ax.plot(xs, ys, color=BLUE, lw=2.6, solid_capstyle="round", zorder=3)
ax.scatter(xs, ys, s=62, color=BLUE, zorder=4, edgecolor="white", linewidth=1.4)
ax.scatter([15], [90], s=150, color=TEAL, zorder=5, edgecolor="white", linewidth=1.8)
ax.text(14.6, 97, "subtype finally\nconfirmed", fontsize=9.6, color=TEAL,
        fontweight="bold", ha="right", va="center")

# ---- India trajectory ----
xi = np.array([0, 4, 9, 15])
yi = np.array([14, 26, 40, 56])
ax.plot(xi, yi, color=RED, lw=2.2, ls=(0, (5, 3)), zorder=3)
ax.scatter(xi, yi, s=46, color=RED, zorder=4, edgecolor="white", linewidth=1.2)
ax.text(15.4, 30, "In India: symptom-based diagnosis,\nfew movement-disorder specialists,\n"
                  "longer gaps between visits",
        fontsize=9.4, color=RED, ha="right", va="top",
        bbox=dict(fc=BAND, ec="none", pad=2))

# ---- span annotation ----
ax.annotate("", xy=(0, 8), xytext=(15, 8),
            arrowprops=dict(arrowstyle="<|-|>", color=AMBER, lw=1.4,
                            mutation_scale=11, shrinkA=0, shrinkB=0))
ax.text(7.5, 3.5, "6-7+ visits over more than a year spent managing the patient as a milder, generic case",
        fontsize=10.5, color=AMBER, fontweight="bold", ha="center", va="center",
        bbox=dict(fc=BAND, ec="none", pad=2))

# ---- takeaway strip ----
fig.patches.append(Rectangle((0.045, 0.035), 0.91, 0.125, transform=fig.transFigure,
                             fc="#eef4fb", ec=BLUE, lw=1.2, zorder=1))
fig.text(0.072, 0.098,
         "A third to half of patients are reclassified within 1-2 years, so even the confirmed label is unstable.",
         fontsize=12, fontweight="bold", color="#1a1a1a", va="center")
fig.text(0.072, 0.062,
         "Through all of it, a fast-declining PIGD patient looks like a milder TD case.",
         fontsize=12, color="#333333", va="center")

fig.savefig(_FIGURES / "fig_problem_chart.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(_FIGURES / "fig_problem_chart.pdf", bbox_inches="tight", facecolor="white")
print("done")
