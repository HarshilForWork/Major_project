# -*- coding: utf-8 -*-
"""Generate the preprocessing figures for the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
_RAW       = _ROOT / "data" / "raw" / "ppmi_csv"
_PROCESSED = _ROOT / "data" / "processed"
_METRICS   = _ROOT / "reports" / "metrics"
_FIGURES   = _ROOT / "reports" / "figures"
for _d in (_PROCESSED, _METRICS, _FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


NAVY = "#1a3a5c"; BLUE = "#2a6ea8"; GREEN = "#2e7d4f"; RED = "#b3352e"
AMBER = "#c98a2a"; GREY = "#5a5a5a"
LBLUE = "#e8f1f8"; LGREEN = "#e9f5ec"; LRED = "#fdecea"; LAMBER = "#fdf3e7"; LGREY = "#f2f2f2"


def box(ax, x, y, w, h, text, fc, ec, fs=9.5, bold=False, tc="#000000"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                 fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3, linespacing=1.45)


def arrow(ax, p1, p2, color=GREY, label=None, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2, zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my, label, fontsize=8, color=color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=4)


# ===========================================================================
# FIGURE A -- preprocessing pipeline with the leakage fork
# ===========================================================================
fig, ax = plt.subplots(figsize=(11.2, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(50, 97, "Preprocessing Pipeline and Leakage Control",
        ha="center", fontsize=14, fontweight="bold", color=NAVY)

# Stage 1: raw
box(ax, 3, 84, 94, 8.5,
    "RAW PPMI DATA  ·  17 visit-level tables + 4 static tables  ·  935 enrolled subjects",
    LGREY, GREY, 10, True, NAVY)

# Stage 2: join
box(ax, 3, 72, 94, 8.5,
    "JOIN   outer join on PATNO + EVENT_ID (visit-level)   |   left join on PATNO (static)\n"
    "medication joined by DATE INTERVAL — logs are not visit-keyed",
    LBLUE, BLUE, 9, False)
arrow(ax, (50, 84), (50, 80.5))

# Stage 3: filters
box(ax, 3, 60, 94, 8.5,
    "FILTER   PD cohort (COHORT==1) → 439 patients   ·   drop 3,703 non-scheduled visit rows\n"
    "no minimum follow-up filter   ·   missing values retained as NaN (no imputation)",
    LBLUE, BLUE, 9, False)
arrow(ax, (50, 72), (50, 68.5))

box(ax, 22, 50.5, 56, 6.5,
    "ANALYSIS TABLE   6,922 patient-visits  ×  75 columns",
    LGREY, NAVY, 10, True, NAVY)
arrow(ax, (50, 60), (50, 57))

# --- the fork ---
ax.text(50, 46.3, "COLUMNS SPLIT — the two branches must never rejoin before the model",
        ha="center", fontsize=9.5, fontweight="bold", color=RED,
        bbox=dict(fc="white", ec="none", pad=2))

arrow(ax, (42, 50.5), (24, 41.5), color=RED)
arrow(ax, (58, 50.5), (76, 41.5), color=GREEN)

# LEFT branch: label
box(ax, 2, 30, 44, 11,
    "BRANCH A — LABEL ONLY  (16 items)\n"
    "11 tremor items  ·  5 gait/balance items\n"
    "(13 from Part III exam, 3 from Part II)",
    LRED, RED, 9, False)

box(ax, 2, 17.5, 44, 10,
    "APPLY STEBBINS FORMULA\n"
    "ratio = mean(tremor) / mean(PIGD)\n"
    "≥1.15 → TD    ≤0.90 → PIGD    else → Indeterminate",
    LRED, RED, 8.8, False)
arrow(ax, (24, 30), (24, 27.5), color=RED)

box(ax, 7, 8, 34, 6.5, "Y  =  TD / PIGD / Indeterminate\n5,742 labelled visits (83.0%)",
    LRED, RED, 9, True)
arrow(ax, (24, 17.5), (24, 14.5), color=RED)

# RIGHT branch: features
box(ax, 54, 30, 44, 11,
    "BRANCH B — PERMITTED FEATURES  (27)\n"
    "10 non-formula ADL items  ·  6 non-motor assessment totals\n"
    "medication  ·  demographics  ·  disease duration",
    LGREEN, GREEN, 9, False)

box(ax, 54, 17.5, 44, 10,
    "LEAKAGE GUARD  (asserted in code)\n"
    "19 columns banned:  16 formula items\n"
    "+ NP3TOT, NP2PTOT, NHY (encode Y indirectly)",
    LAMBER, AMBER, 8.8, False)
arrow(ax, (76, 30), (76, 27.5), color=GREEN)

box(ax, 59, 8, 34, 6.5, "X  =  27 permitted features",
    LGREEN, GREEN, 9, True)
arrow(ax, (76, 17.5), (76, 14.5), color=GREEN)

# converge only at model
box(ax, 30, 1, 40, 5, "MODEL TRAINING   (X → Y)", LGREY, NAVY, 10, True, NAVY)
arrow(ax, (24, 8), (40, 6), color=RED)
arrow(ax, (76, 8), (60, 6), color=GREEN)

ax.text(76, 16.0, "verified: including the banned items → accuracy 0.906 (meaningless)",
        fontsize=7.8, color=AMBER, ha="center", va="center", style="italic")

plt.tight_layout()
plt.savefig(_FIGURES / "fig_preprocessing_flow.png", dpi=190, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote fig_preprocessing_flow.png")

# ===========================================================================
# FIGURE B -- cohort funnel + label/flip summary
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5),
                         gridspec_kw={"width_ratios": [1.35, 1, 1]})

# --- panel 1: funnel ---
ax = axes[0]
stages = [
    ("Enrolled subjects\nin PPMI extract", 935, LGREY, GREY),
    ("PD cohort\n(COHORT == 1)", 439, LBLUE, BLUE),
    ("Patients with a\ncomputable label", 439, LBLUE, BLUE),
    ("Patients usable at\nall 4 visit rounds", 275, LGREEN, GREEN),
]
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.set_title("Cohort funnel (patients)", fontsize=11, fontweight="bold", color=NAVY)
y = 82
for label, n, fc, ec in stages:
    w = 18 + 72 * (n / 935)
    x = (100 - w) / 2
    ax.add_patch(FancyBboxPatch((x, y), w, 13, boxstyle="round,pad=0.4,rounding_size=1.5",
                                 fc=fc, ec=ec, lw=1.6))
    ax.text(50, y + 8.4, f"{n}", ha="center", va="center", fontsize=13,
            fontweight="bold", color=ec)
    ax.text(50, y + 3.4, label, ha="center", va="center", fontsize=7.4, color="#222222")
    if y > 20:
        ax.annotate("", xy=(50, y - 3.2), xytext=(50, y - 0.4),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
    y -= 21
ax.text(50, 1, "6,922 visit rows → 5,742 labelled (83.0%)",
        ha="center", fontsize=8.5, color=NAVY, fontweight="bold")

# --- panel 2: label distribution ---
ax = axes[1]
labels = ["TD", "PIGD", "Indeter-\nminate"]
vals = [52.6, 36.6, 10.8]
cols = [BLUE, GREEN, AMBER]
bars = ax.bar(labels, vals, color=cols, edgecolor="white", width=0.62)
for b, v, n in zip(bars, vals, [3020, 2102, 620]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v}%\n(n={n})",
            ha="center", fontsize=8.6, fontweight="bold", color="#222222")
ax.set_ylim(0, 66); ax.set_ylabel("% of labelled visits", fontsize=9)
ax.set_title("Label distribution\n(class imbalance)", fontsize=11, fontweight="bold", color=NAVY)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)

# --- panel 3: flip rates ---
ax = axes[2]
labels2 = ["TD", "PIGD", "Indeter-\nminate", "Overall"]
vals2 = [20.9, 27.1, 76.3, 29.1]
cols2 = [BLUE, GREEN, AMBER, NAVY]
bars = ax.barh(range(4), vals2, color=cols2, edgecolor="white", height=0.6)
ax.set_yticks(range(4)); ax.set_yticklabels(labels2, fontsize=8.5)
ax.invert_yaxis()
for i, v in enumerate(vals2):
    ax.text(v + 1.6, i, f"{v}%", va="center", fontsize=8.8, fontweight="bold", color="#222222")
ax.set_xlim(0, 92); ax.set_xlabel("% flipping at next visit", fontsize=9)
ax.set_title("Label instability\n(4,918 visit pairs)", fontsize=11, fontweight="bold", color=NAVY)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8.5)

plt.tight_layout()
plt.savefig(_FIGURES / "fig_cohort_labels.png", dpi=190, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote fig_cohort_labels.png")
