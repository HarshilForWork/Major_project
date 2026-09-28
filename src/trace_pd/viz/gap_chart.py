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


RED   = "#c0272d"
BAND  = "#f7dcdc"
NAVY  = "#14375e"
GREY  = "#6a6a6a"

fig, ax = plt.subplots(figsize=(12.6, 6.2))
fig.patch.set_facecolor("white")

# ---------- title ----------
ax.set_position([0.085, 0.315, 0.885, 0.545])
fig.text(0.045, 0.945, "The Gap This Creates", fontsize=21, fontweight="bold", color=RED)
fig.text(0.045, 0.885, "Function lost while the subtype is still unconfirmed",
         fontsize=11, color=GREY)

# ---------- axes frame ----------
ax.set_xlim(-0.6, 16.2)
ax.set_ylim(0, 10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color(GREY);   ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_color(GREY); ax.spines["bottom"].set_linewidth(1.2)

ticks  = [0, 3, 6, 9, 12, 15]
labels = ["0", "3 months", "6 months", "9 months", "12 months", "> 12 months"]
ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10, color="#333333")
ax.set_yticks([])
ax.set_xlabel("Time since symptom onset", fontsize=11, color="#333333", labelpad=9)

ax.text(-1.25, 9.2, "Higher\nfunction", fontsize=9.5, color="#333333", ha="center", va="center")
ax.text(-1.25, 1.4, "Lower\nfunction",  fontsize=9.5, color="#333333", ha="center", va="center")
ax.text(-2.9, 5.4, "Patient\nfunction", fontsize=12, fontweight="bold",
        color="#222222", ha="center", va="center")
ax.text(-2.9, 4.1, "(e.g. mobility,\nindependence)", fontsize=8.8,
        color=GREY, ha="center", va="center")

# ---------- delay band ----------
ax.add_patch(Rectangle((3, 0), 9, 10, fc=BAND, ec="none", zorder=0))
ax.annotate("", xy=(3.15, 9.3), xytext=(11.85, 9.3),
            arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.4,
                            mutation_scale=11, shrinkA=0, shrinkB=0))
ax.text(7.5, 9.72, "Delay before confident diagnosis", fontsize=11.5,
        fontweight="bold", color=RED, ha="center", va="center")

# ---------- decline curve ----------
xs = np.array([0, 3, 6, 9, 12])
ys = np.array([9.0, 7.9, 6.9, 5.9, 4.9])
ax.plot(xs, ys, color=RED, lw=2.6, solid_capstyle="round", zorder=3)
ax.scatter(xs[1:], ys[1:], s=62, color=RED, zorder=4,
           edgecolor="white", linewidth=1.4)

ax.add_patch(FancyArrowPatch((12, 4.9), (15.8, 2.55), arrowstyle="-|>",
                             mutation_scale=17, color=RED, lw=2.4,
                             linestyle=(0, (5, 3)), zorder=3))
ax.text(15.9, 4.55, "Greater decline\nthe longer the delay",
        fontsize=10, fontweight="bold", color=RED, ha="right", va="center")

# visit markers along the delay
for x, lab in zip([3, 6, 9, 12], ["Visit 2", "Visit 3", "Visit 4", "Visit 6"]):
    ax.text(x, 0.5, lab, fontsize=8.6, color=GREY, ha="center")

# ---------- takeaway strip ----------
fig.patches.append(Rectangle((0.045, 0.035), 0.91, 0.125, transform=fig.transFigure,
                             fc="#fdf1f1", ec=RED, lw=1.2, zorder=1))
fig.text(0.072, 0.098,
         "Every extra month without a confident subtype means delayed, less-targeted care,",
         fontsize=12, fontweight="bold", color="#1a1a1a", va="center")
fig.text(0.072, 0.062,
         "especially costly for patients who are progressing fast.",
         fontsize=12, color="#333333", va="center")

fig.savefig(_FIGURES / "fig_gap_chart.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(_FIGURES / "fig_gap_chart.pdf", bbox_inches="tight", facecolor="white")
print("done")
