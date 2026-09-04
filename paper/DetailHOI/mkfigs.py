import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
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

def curve(key):
    d = C[key]
    ep = sorted(int(k) for k in d)
    return ep, [d[str(e)] for e in ep]

def last5(key):
    _, y = curve(key)
    return sum(y[-5:]) / 5

# The duplicated K=23 configuration fixes the resolution of every comparison.
FLOOR = max(abs(f("k23_r1") - f("k23_r1b")) for f in
            (lambda k: max(curve(k)[1]), lambda k: curve(k)[1][-1], last5))

SERIES = [
    ("PViC baseline",      "baseline_filtered", BLUE,   "-",  "o"),
    (r"DetailHOI $K$=17",  "k17_r3",            ORANGE, "-",  "s"),
    (r"DetailHOI $K$=23",  "k23_r1",            AQUA,   "--", "^"),
    (r"DetailHOI $K$=31",  "k31_r1",            YELLOW, "-.", "D"),
]

# ============================ Figure: convergence ============================
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.28))

# the two independent K=23 runs, drawn as a band = run-to-run spread
epb, yb1 = curve("k23_r1")
_,   yb2 = curve("k23_r1b")
for ax, lo in ((axL, 1), (axR, 20)):
    m = [i for i, e in enumerate(epb) if e >= lo]
    ax.fill_between([epb[i] for i in m], [yb1[i] for i in m], [yb2[i] for i in m],
                    color=AQUA, alpha=.20, lw=0, zorder=2)
axL.plot(epb, yb2, color=AQUA, lw=.9, ls="--", alpha=.55, zorder=3)
axR.plot([e for e in epb if e >= 20], [v for e, v in zip(epb, yb2) if e >= 20],
         color=AQUA, lw=1.0, ls="--", alpha=.55, zorder=3)

for label, key, col, ls, mk in SERIES:
    ep, y = curve(key)
    axL.plot(ep, y, color=col, lw=1.6, ls=ls, zorder=4,
             solid_capstyle="round", dash_capstyle="round")
    m = [i for i in range(len(ep)) if ep[i] in (20, 30)]
    axL.plot([ep[i] for i in m], [y[i] for i in m], mk, color=col,
             ms=3.4, mec="white", mew=.7, zorder=5)
    sub = [(e, v) for e, v in zip(ep, y) if e >= 20]
    axR.plot([e for e, _ in sub], [v for _, v in sub], color=col, lw=1.8, ls=ls,
             marker=mk, ms=4.2, mec="white", mew=.8, zorder=4,
             solid_capstyle="round", dash_capstyle="round")
    LABEL_DY = {"baseline_filtered": 6.5, "k17_r3": 0.0,
                "k23_r1": -8.0, "k31_r1": -1.0}
    tag = label + r" ($\times$2)" if key == "k23_r1" else label
    axR.annotate(tag, xy=(30, sub[-1][1]), xytext=(5, LABEL_DY[key]),
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
axR.set_xlim(19.4, 37.5)
axR.spines["bottom"].set_bounds(20, 30)

fig.savefig(f"{OUT}/convergence.pdf")
fig.savefig(f"{OUT}/convergence.png", dpi=300)
plt.close(fig)

# ====================== Figure: ablation dot plot (Last-5) ======================
# Values are means over epochs 26-30, matching Table 4's Last-5 column.
rows = [
    ("PViC baseline (no keypoints)",       [last5("baseline_filtered")],      BLUE),
    (r"$K$=31 interp. skeleton, 256/256",  [last5("k31_r1")],                 YELLOW),
    (r"$K$=23 interp. skeleton, 256/256",  [last5("k23_r1"), last5("k23_r1b")], AQUA),
    (r"$K$=17 COCO joints, 384/128",       [last5("k17_r3")],                 ORANGE),
]
fig, ax = plt.subplots(figsize=(3.4, 1.95))
ys = list(range(len(rows)))
base = rows[0][1][0]

ax.axvspan(base - FLOOR, base + FLOOR, color=MUTED, alpha=.13, lw=0, zorder=1)
ax.axvline(base, color=MUTED, lw=.7, ls=(0, (2, 2)), zorder=2)

for y, (lab, vals, col) in zip(ys, rows):
    for v in vals:
        ax.hlines(y, base, v, color=col, lw=1.6, zorder=3)
    if len(vals) > 1:
        ax.hlines(y, min(vals), max(vals), color=col, lw=3.2, alpha=.45, zorder=3)
    for v in vals:
        ax.plot(v, y, "o", color=col, ms=6, mec="white", mew=.9, zorder=4)
    tx = sum(vals) / len(vals)
    txt = f"{min(vals):.4f}/{max(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
    ax.annotate(txt, xy=(tx, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7, color=INK)

ax.annotate("run-to-run spread", xy=(base, -.52), xytext=(0, 0),
            textcoords="offset points", ha="center", va="center",
            fontsize=6.6, color=MUTED)
ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.2)
ax.set_xlim(.6335, .6408)
ax.set_ylim(-.85, len(rows) - .35)
ax.set_xlabel("Last-5 mAP (epochs 26–30), human-clarity V-COCO subset")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=GRID, lw=.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=2.5, width=.6)
fig.savefig(f"{OUT}/ablation.pdf")
fig.savefig(f"{OUT}/ablation.png", dpi=300)
plt.close(fig)
print(f"figures written (noise floor = {FLOOR*100:.2f} mAP)")
