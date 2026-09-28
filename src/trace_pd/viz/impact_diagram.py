import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

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
GREEN = "#2e7d4f"
GREY  = "#6a6a6a"

W, H = 100, 108
fig, ax = plt.subplots(figsize=(5.3, 5.75))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor("white")


CARDS = [
    dict(y=90, accent=BLUE,  fill="#eef4fb",
         title="Earlier Confidence",
         body=["Confident subtype classification reached",
               "in fewer visits, supporting earlier and",
               "tailored monitoring — especially for",
               "faster-declining PIGD patients."]),
    dict(y=57, accent=GREEN, fill="#eef7f0",
         title="Transparent Predictions",
         body=["Every prediction carries a calibrated",
               "trust score and a plain-language",
               "explanation, not a black-box output."]),
    dict(y=24, accent=TEAL,  fill="#e9f6f6",
         title="Closing a Research Gap",
         body=["No existing system combines subtype",
               "prediction, calibrated confidence and",
               "visit-to-visit explainability.",
               "This framework is the first to do so."]),
]

BX, BW, BH = 6, 88, 27
for c in CARDS:
    y = c["y"]
    ax.add_patch(FancyBboxPatch((BX, y - BH/2), BW, BH,
                 boxstyle="round,pad=0,rounding_size=1.4",
                 fc=c["fill"], ec=c["accent"], lw=1.7, zorder=3))
    ax.add_patch(Rectangle((BX, y - BH/2 + 0.9), 1.6, BH - 1.8,
                 fc=c["accent"], ec="none", zorder=4))
    ax.text(BX + 5, y + 9.0, c["title"], fontsize=12.5, fontweight="bold",
            color=c["accent"], va="center", zorder=5)
    for i, line in enumerate(c["body"]):
        ax.text(BX + 5, y + 3.2 - i*4.1, line, fontsize=8.6,
                color="#222222", va="center", zorder=5)

for y0, y1 in ((90 - BH/2, 57 + BH/2), (57 - BH/2, 24 + BH/2)):
    ax.add_patch(FancyArrowPatch((BX + BW/2, y0), (BX + BW/2, y1 + 0.4),
                 arrowstyle="-|>", mutation_scale=13, color=NAVY, lw=1.6, zorder=2))

ax.text(W/2, 4.5, "A confident, explained subtype call — sooner.",
        fontsize=10, fontweight="bold", color=NAVY, ha="center", va="center")

plt.tight_layout()
fig.savefig(_FIGURES / "fig_impact_diagram.png", dpi=230, bbox_inches="tight", facecolor="white")
fig.savefig(_FIGURES / "fig_impact_diagram.pdf", bbox_inches="tight", facecolor="white")
print("done")
