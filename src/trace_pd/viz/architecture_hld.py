# -*- coding: utf-8 -*-
"""Three-column system architecture:  TRAINING (left) | DATA LAYER (centre) | INFERENCE (right).

Graphviz kept placing the data layer on the right because cluster contiguity
outweighs the ordering hints, so the columns are positioned explicitly here.
Every connector is a short horizontal hop between ADJACENT columns, which keeps
routing trivial: arrows leave the right edge of one box and enter the left edge
of another, with the label sitting on the gap between the columns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
_RAW       = _ROOT / "data" / "raw" / "ppmi_csv"
_PROCESSED = _ROOT / "data" / "processed"
_METRICS   = _ROOT / "reports" / "metrics"
_FIGURES   = _ROOT / "reports" / "figures"
for _d in (_PROCESSED, _METRICS, _FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


BLUE = "#2a6ea8"; GREEN = "#2e7d4f"; RED = "#b3352e"; AMBER = "#b8791c"
PURPLE = "#63479a"; TEAL = "#1d7373"; GREY = "#6a6a6a"; NAVY = "#16334f"
LBLUE = "#eef4fb"; LGREEN = "#eef7f0"; LTEAL = "#e9f6f6"
LRED = "#fdeeec"; LAMBER = "#fdf5e8"; LPURPLE = "#f1ecf9"; LGREY = "#f0f0f0"

# ---------------------------------------------------------------- geometry
W, H = 150.0, 116.0
COL_W = 40.0
X_TRAIN, X_DATA, X_INFER = 5.0, 55.0, 105.0          # column left edges
BW = 32.0                                            # box width
CX_T, CX_D, CX_I = X_TRAIN + COL_W / 2, X_DATA + COL_W / 2, X_INFER + COL_W / 2

fig, ax = plt.subplots(figsize=(15.0, 11.6))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def column(x, y, w, h, title, sub, fc, ec, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=1))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=ec, zorder=6)
    ax.text(x + w / 2, y + h - 6.6, sub, ha="center", va="center",
            fontsize=7.2, color="#555555", zorder=6)


def box(cx, cy, text, fc, ec, h=7.6, w=BW, fs=8.6, shape="box"):
    x, y = cx - w / 2, cy - h / 2
    if shape == "cyl":
        ax.add_patch(Rectangle((x, y + 1.3), w, h - 2.6, fc=fc, ec=ec, lw=1.5, zorder=4))
        ax.add_patch(Ellipse((cx, y + h - 1.3), w, 2.6, fc=fc, ec=ec, lw=1.5, zorder=5))
        ax.add_patch(Ellipse((cx, y + 1.3), w, 2.6, fc=fc, ec=ec, lw=1.5, zorder=3))
    elif shape == "ext":
        ax.add_patch(Rectangle((x + 0.7, y - 0.7), w, h, fc="#ffffff", ec=ec, lw=1.2, zorder=3))
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5, zorder=4))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.7",
                                    fc=fc, ec=ec, lw=1.5, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=6, linespacing=1.5)
    return (x, y, w, h)


def vdown(cx, y_from, y_to, label, color=GREY, ls="-"):
    """vertical connector inside a column"""
    ax.add_patch(FancyArrowPatch((cx, y_from), (cx, y_to), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=1.25,
                                 linestyle=ls, zorder=2, shrinkA=1, shrinkB=1))
    if label:
        ax.text(cx + 0.9, (y_from + y_to) / 2, label, fontsize=7.0, color=color,
                ha="left", va="center", zorder=7,
                bbox=dict(fc="white", ec="none", pad=0.6))


def hop(x_from, y_from, x_to, y_to, label, color=GREY, ls="-", above=True):
    """short horizontal connector between adjacent columns"""
    ax.add_patch(FancyArrowPatch((x_from, y_from), (x_to, y_to), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=1.25,
                                 linestyle=ls, zorder=2, shrinkA=1, shrinkB=1,
                                 connectionstyle="arc3,rad=0"))
    ax.text((x_from + x_to) / 2, (y_from + y_to) / 2 + (1.5 if above else -1.6),
            label, fontsize=7.0, color=color, ha="center", va="center", zorder=7,
            bbox=dict(fc="white", ec="none", pad=0.6))


def diag(p1, p2, label, color, ls="-", rad=0.14, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.25, linestyle=ls, zorder=2,
                                 shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=7.0,
            color=color, ha="center", va="center", zorder=7,
            bbox=dict(fc="white", ec="none", pad=0.6))


# ---------------------------------------------------------------- title
ax.text(W / 2, H - 2.0, "High-Level Design — System Architecture",
        ha="center", fontsize=15.0, fontweight="bold", color=NAVY)
ax.text(W / 2, H - 5.4, "PD motor subtype classification (TD / PIGD / Indeterminate)  ·  training (left) and inference (right) are independent subsystems, meeting only at the shared data layer (centre)",
        ha="center", fontsize=8.8, color="#555555")

# ---------------------------------------------------------------- columns
column(X_TRAIN, 6, COL_W, 100, "TRAINING", "offline · batch · per data extract", LBLUE, BLUE)
column(X_DATA,  6, COL_W, 100, "DATA LAYER", "written by training · read by inference", LTEAL, TEAL, 2.4)
column(X_INFER, 6, COL_W, 100, "INFERENCE", "online · per patient visit", LGREEN, GREEN)

# ---------------------------------------------------------------- TRAINING column
yT = [93.5, 85.5, 77, 68, 58.5, 47, 34]
LXT, RXT = CX_T - 8.6, CX_T + 8.6
box(CX_T, yT[0], "PPMI @ LONI IDA\n19 source tables", LGREY, GREY, shape="ext", fs=8.2)
box(CX_T, yT[1], "Ingestion & Join", "#ffffff", BLUE, h=7.0)
box(CX_T, yT[2], "Preprocessing &\nFeature Governance", "#ffffff", BLUE, h=7.0)
box(LXT, yT[3], "Label Engine\nStebbins ratio", LRED, RED, w=14.6, h=7.0, fs=7.6)
box(RXT, yT[3], "Leakage Guard\npolicy gate", LAMBER, AMBER, w=14.6, h=7.0, fs=7.6)
box(CX_T, yT[4], "Training Matrix\none row per patient-visit  ·  permitted X + subtype Y",
    "#ffffff", BLUE, h=7.6, fs=7.4)
box(LXT, yT[5], "Visit-Pair Builder\npairs t → t+1\nflip target · 4,918", LRED, RED,
    w=14.6, h=9.2, fs=7.2)
box(RXT, yT[5], "Subtype Trainer\n+ Calibrator", "#ffffff", BLUE, w=14.6, h=9.2, fs=7.8)
box(LXT, yT[6], "Transition Trainer\n+ Calibrator", "#ffffff", BLUE, w=14.6, h=8.4, fs=7.8)

vdown(CX_T, yT[0] - 3.8, yT[1] + 3.5, "extract", GREY, ls=(0, (4, 2)))
vdown(CX_T, yT[1] - 3.5, yT[2] + 3.5, "joined tables", BLUE)
diag((CX_T - 3, yT[2] - 3.5), (LXT, yT[3] + 3.5), "label items", GREY, rad=0.0, lx=-3.4, ly=0)
diag((CX_T + 3, yT[2] - 3.5), (RXT, yT[3] + 3.5), "feature split", GREY, rad=0.0, lx=3.6, ly=0)
diag((LXT, yT[3] - 3.5), (CX_T - 5, yT[4] + 3.8), "subtype Y", RED, rad=0.0, lx=-3.2, ly=0)
diag((RXT, yT[3] - 3.5), (CX_T + 5, yT[4] + 3.8), "permitted X", AMBER, rad=0.0, lx=3.4, ly=0)

# ---- the fork: single-visit rows train the subtype head,
#                the same rows are paired up to train the transition head
diag((CX_T - 5, yT[4] - 3.8), (LXT, yT[5] + 4.6), "same rows,\npaired t → t+1", BLUE,
     rad=0.0, lx=-4.6, ly=0.4)
diag((CX_T + 5, yT[4] - 3.8), (RXT, yT[5] + 4.6), "single-visit\nrows", BLUE,
     rad=0.0, lx=4.4, ly=0.4)
vdown(LXT, yT[5] - 4.6, yT[6] + 4.2, "flip target", RED)

# ---------------------------------------------------------------- DATA column
yD = [83, 73.5, 64, 54.5, 28]
box(CX_D, yD[0], "Raw Data Store", "#ffffff", TEAL, shape="cyl", h=8.4, fs=8.2)
box(CX_D, yD[1], "Curated Dataset\n6,922 x 75", "#ffffff", TEAL, shape="cyl", h=8.4, fs=8.2)
box(CX_D, yD[2], "Feature Registry\nX / Y / banned", LAMBER, AMBER, shape="cyl", h=8.4, fs=8.2)
box(CX_D, yD[3], "Model Registry\nsubtype + transition models\n+ calibrators", LPURPLE, PURPLE, shape="cyl", h=9.6, fs=7.6)
box(CX_D, yD[4], "Patient History Store\nper-patient visit record", "#ffffff", TEAL, shape="cyl", h=8.4, fs=8.0)

# ---------------------------------------------------------------- INFERENCE column
yI = [92.5, 83, 73.5, 64, 54.5, 45, 35.5, 26, 16.5, 9.5]
box(CX_I, yI[0], "Clinician Workstation", "#ffffff", GREEN)
box(CX_I, yI[1], "Inference API gateway", "#ffffff", BLUE)
box(CX_I, yI[2], "Feature Assembly", "#ffffff", GREEN)
box(CX_I, yI[3], "Prediction  ·  XGBoost", "#ffffff", GREEN)
box(CX_I, yI[4], "Conformal Confidence", LPURPLE, PURPLE)
box(CX_I, yI[5], "Longitudinal Tracker", "#ffffff", GREEN)
box(CX_I, yI[6], "Transition Risk", LAMBER, AMBER)
box(CX_I, yI[7], "Explanation Orchestrator", "#fffdf7", AMBER)
box(CX_I, yI[8], "Report Renderer", "#ffffff", GREEN)
box(CX_I, yI[9], "LLM Provider API", LGREY, GREY, shape="ext", h=5.6, fs=8.0)

lbl = ["requests", "routes visit", "feature X", "probabilities", "prediction set",
       "curve, stability", "flip risk", "grounded text"]
col = [GREY, BLUE, GREEN, GREEN, PURPLE, GREEN, AMBER, AMBER]
for i in range(8):
    vdown(CX_I, yI[i] - 3.8, yI[i + 1] + 3.8, lbl[i], col[i])
diag((CX_I + 13.5, yI[7] - 2.0), (CX_I + 13.5, yI[9] + 2.2), "prompt", GREY,
     ls=(0, (4, 2)), rad=-0.55, lx=4.2, ly=0)

# ---------------------------------------------------------------- TRAINING -> DATA
RT, LD = CX_T + BW / 2, CX_D - BW / 2
hop(RT, yT[1], LD, yD[0], "writes", BLUE)
hop(RT, yT[2], LD, yD[1], "writes", BLUE)
hop(RXT + 7.3, yT[3], LD, yD[2], "writes contract", AMBER)
hop(RXT + 7.3, yT[5], LD, yD[3] + 1.4, "publishes\nsubtype model", PURPLE)
# orthogonal route for the transition model, kept clear of the subtype trainer
_yb = yT[6]
ax.plot([LXT + 7.3, 49.5], [_yb, _yb], color=PURPLE, lw=1.25, zorder=2)
ax.plot([49.5, 49.5], [_yb, yD[3] - 3.4], color=PURPLE, lw=1.25, zorder=2)
ax.add_patch(FancyArrowPatch((49.5, yD[3] - 3.4), (LD, yD[3] - 3.4), arrowstyle="-|>",
                             mutation_scale=11, color=PURPLE, lw=1.25, zorder=2,
                             shrinkA=0, shrinkB=1))
ax.text(34.5, _yb + 2.4, "publishes\ntransition model", fontsize=7.0, color=PURPLE,
        ha="center", va="center", zorder=7, bbox=dict(fc=LBLUE, ec="none", pad=0.6))
diag((LD, yD[1] - 3.2), (RT - 3, yT[4] + 3.9), "reads", GREY, ls=(0, (4, 2)), rad=0.28, lx=2.0, ly=-2.4)

# ---------------------------------------------------------------- DATA -> INFERENCE
RD, LI = CX_D + BW / 2, CX_I - BW / 2
hop(RD, yD[2], LI, yI[2], "loads contract", AMBER)
hop(RD, yD[3] + 1.4, LI, yI[3], "loads subtype model", PURPLE)
diag((RD, yD[3] - 2.6), (LI, yI[6] + 1.2), "loads transition model", PURPLE, rad=-0.18, ly=-1.6)
diag((RD, yD[4] + 2.0), (LI, yI[5] - 1.4), "reads history", TEAL, rad=-0.16, ly=1.8)
diag((LI, yI[8] + 1.4), (RD, yD[4] - 2.0), "writes visit", TEAL, rad=-0.16, ly=-1.8)

plt.tight_layout()
plt.savefig(_FIGURES / "fig_HLD_system_architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.savefig(_FIGURES / "fig_HLD_system_architecture.pdf", bbox_inches="tight", facecolor="white")
print("wrote fig_arch_3col.png / .pdf")
