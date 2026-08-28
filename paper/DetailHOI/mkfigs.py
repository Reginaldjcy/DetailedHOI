import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = "/mnt/fast/DetailHOI/paper/figures"
C = json.load(open(os.path.join(OUT, "curves.json")))

# --- categorical slots 1-4 (validated: all checks pass, light mode) ---
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#e3e2dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": .6,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)

SERIES = [
    ("PViC baseline",      "baseline_filtered", BLUE,   "-",  "o"),
    (r"DetailHOI $K$=17",  "k17_r3",            ORANGE, "-",  "s"),
    (r"DetailHOI $K$=23",  "k23_r1",            AQUA,   "--", "^"),
    (r"DetailHOI $K$=31",  "k31_r1",            YELLOW, "-.", "D"),
]

# ============================ Figure: convergence ============================
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.28))

for label, key, col, ls, mk in SERIES:
    d = C[key]
    ep = sorted(int(k) for k in d)
    y = [d[str(e)] for e in ep]
    axL.plot(ep, y, color=col, lw=1.6, ls=ls, zorder=3,
             solid_capstyle="round", dash_capstyle="round")
    m = [i for i in range(len(ep)) if ep[i] in (20, 30)]
    axL.plot([ep[i] for i in m], [y[i] for i in m], mk, color=col,
             ms=3.4, mec="white", mew=.7, zorder=4)
    sub = [(e, v) for e, v in zip(ep, y) if e >= 20]
    axR.plot([e for e, _ in sub], [v for _, v in sub], color=col, lw=1.8, ls=ls,
             marker=mk, ms=4.2, mec="white", mew=.8, zorder=3,
             solid_capstyle="round", dash_capstyle="round")
    LABEL_DY = {"baseline_filtered": 3.5, "k17_r3": 0.0,
                "k23_r1": -4.5, "k31_r1": 4.0}
    axR.annotate(label, xy=(30, sub[-1][1]), xytext=(5, LABEL_DY[key]),
                 textcoords="offset points", va="center", ha="left",
                 fontsize=7.2, color=INK)

axL.axvline(20, color=MUTED, lw=.6, ls=(0, (2, 2)), zorder=1)
axL.annotate("LR drop", xy=(20, .455), xytext=(2, 0), textcoords="offset points",
             fontsize=6.8, color=MUTED, va="bottom")
axL.set_xlabel("epoch"); axL.set_ylabel("mAP")
axL.set_xlim(0, 31); axL.set_ylim(.44, .66)
axL.xaxis.set_major_locator(MultipleLocator(5))
axL.set_title("(a) Full training run", loc="left", color=INK)
style(axL)

axR.set_xlabel("epoch"); axR.set_ylabel("mAP")
axR.set_ylim(.6275, .6415)
axR.set_xticks([20, 22, 24, 26, 28, 30])
axR.set_title("(b) After the learning-rate drop", loc="left", color=INK)
style(axR)
axR.set_xlim(19.4, 36.0)
axR.spines["bottom"].set_bounds(20, 30)

fig.savefig(f"{OUT}/convergence.pdf")
fig.savefig(f"{OUT}/convergence.png", dpi=300)
plt.close(fig)

# ============================ Figure: ablation dot plot ============================
rows = [
    ("PViC baseline (no keypoints)",           None,  0.6395, BLUE),
    (r"$K$=31 interp. skeleton, 256/256",      "k31_r1", 0.6365, YELLOW),
    (r"$K$=23 interp. skeleton, 256/256",      "k23_r1", 0.6363, AQUA),
    (r"$K$=17 COCO joints, 384/128",           "k17_r3", 0.6396, ORANGE),
]
fig, ax = plt.subplots(figsize=(3.4, 1.85))
ys = list(range(len(rows)))
base = rows[0][2]
ax.axvline(base, color=MUTED, lw=.7, ls=(0, (2, 2)), zorder=1)
for y, (lab, _, v, col) in zip(ys, rows):
    ax.hlines(y, base, v, color=col, lw=1.6, zorder=2)
    ax.plot(v, y, "o", color=col, ms=6, mec="white", mew=.9, zorder=3)
    ax.annotate(f"{v:.4f}", xy=(v, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7, color=INK)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.2)
ax.set_xlim(.6345, .6412)
ax.set_ylim(-.6, len(rows) - .35)
ax.set_xlabel("best mAP, human-clarity V-COCO subset")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=GRID, lw=.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=2.5, width=.6)
fig.savefig(f"{OUT}/ablation.pdf")
fig.savefig(f"{OUT}/ablation.png", dpi=300)
plt.close(fig)
print("figures written")
