import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/mnt/fast/DetailHOI/paper/figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
FROZEN, PANEL = "#f1f0ed", "#ffffff"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

fig, ax = plt.subplots(figsize=(7.1, 3.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, label, ec=MUTED, fc=PANEL, fs=7.2, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.1",
                                linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.4)
    return (x, y, w, h)

def poly(pts, color=MUTED, head=True, ls="-"):
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", linewidth=.9,
                                     color=color, linestyle=ls,
                                     shrinkA=0, shrinkB=0, zorder=1))
    a, b = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if head else "-",
                                 mutation_scale=7, linewidth=.9, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))

# ---------------- frozen perception ----------------
img   = box(1,    28.0,  9.5, 8.0, "input\nimage")
detr  = box(14.5, 28.0, 16.5, 8.0, "DETR  (frozen)\nR50 + enc/dec", ec=MUTED, fc=FROZEN, fs=6.9)
pose  = box(14.5, 15.5, 14.0, 8.0, "RTMPose  (frozen)\ntop-down pose", ec=ORANGE, fc=FROZEN, fs=6.7)

poly([(10.5, 32.0), (14.5, 32.0)])
poly([(5.75, 28.0), (5.75, 19.5), (14.5, 19.5)], color=MUTED)

# ---------------- proposals & keypoints ----------------
prop = box(36, 28.0, 16, 8.0, "region proposals\n$\\mathbf{b}_i,\\ \\mathbf{e}_i,\\ s_i,\\ c_i$", ec=BLUE)
kpts = box(36, 15.5, 16, 8.0, "keypoints\n$\\mathbf{p}_{i,k},\\ \\sigma_{i,k}$", ec=ORANGE)
poly([(31.0, 32.0), (36.0, 32.0)])
poly([(28.5, 19.5), (36.0, 19.5)], color=ORANGE)
poly([(44.0, 28.0), (44.0, 23.5)], color=BLUE, ls=(0, (2.5, 2)))
ax.text(43.0, 25.6, "human boxes", fontsize=6.3, color=INK2, ha="right", va="center")

# ---------------- positional embeddings ----------------
opes = box(56, 29.0, 16, 6.0, "object centre PE\n$W_o\\,\\mathrm{PE}(\\mathbf{c}_j)\\in\\mathbb{R}^{D_o}$", ec=BLUE, fs=6.9)
kpe  = box(56, 20.5, 16, 6.5, "skeleton PE $\\in\\mathbb{R}^{D_h}$\n$W_h\\,\\mathrm{vec}\\,[\\mathrm{PE}(\\mathbf{p}_{i,k})]_{k\\leq K}$",
           ec=ORANGE, fs=6.5)
hoq  = box(56, 10.0, 16, 7.0, "HO query $\\mathbf{q}_{ij}$\nvisual $\\oplus$ 36-d spatial", ec=MUTED, fs=6.9)

poly([(52, 32.0), (56, 32.0)], color=BLUE)
poly([(52, 19.5), (56, 23.75)], color=ORANGE)
poly([(52, 29.6), (54, 29.6), (54, 13.5), (56, 13.5)], color=BLUE)

# ---------------- concatenation ----------------
ax.plot([73.4, 74.6, 74.6, 73.4], [32.0, 32.0, 23.75, 23.75], color=INK2, lw=.9, zorder=1)
poly([(74.6, 27.9), (81.5, 27.0)], color=INK2)
ax.text(75.0, 30.3, "$\\mathbf{z}_{ij}$", fontsize=7.4, color=INK, ha="left", va="center")
ax.text(75.0, 24.6, "$D_h{+}D_o$\n$=512$", fontsize=6.0, color=INK2,
        ha="left", va="center", linespacing=1.3)

# ---------------- memory branch ----------------
mem = box(36, 1.5, 16, 6.0, "FPN + Swin block\nmemory $K/V$", ec=AQUA, fs=6.9)
poly([(30.0, 28.0), (30.0, 4.5), (36.0, 4.5)], color=AQUA)
ax.text(30.9, 11.0, "C5", fontsize=6.4, color=INK2, ha="left", va="center")
poly([(52, 4.5), (77.6, 4.5), (77.6, 16.0), (81.5, 16.0)], color=AQUA)

# ---------------- decoder & classifier ----------------
dec  = box(81.5, 12.0, 17.5, 19.0,
           "triplet decoder\n(2 layers)\n\nself-attn $q_{pos}$: box PE\ncross-attn $q_{pos}$: $\\mathbf{z}_{ij}$\n$K/V$: image memory",
           ec=AQUA, fs=6.4)
head = box(81.5, 2.5, 17.5, 6.0, "verb logits $\\times$ prior", ec=MUTED, fs=6.9)
poly([(72, 13.5), (74.0, 13.5), (74.0, 21.0), (81.5, 21.0)])
poly([(90.25, 12.0), (90.25, 8.5)])

# ---------------- legend ----------------
ax.add_patch(FancyBboxPatch((1, 10.4), 4.2, 2.4,
                            boxstyle="round,pad=0,rounding_size=.7",
                            linewidth=1.0, edgecolor=MUTED, facecolor=FROZEN, zorder=2))
ax.text(6.1, 11.6, "frozen", fontsize=6.3, color=INK2, ha="left", va="center")
ax.plot([1, 5.2], [7.4, 7.4], color=ORANGE, lw=1.7, solid_capstyle="round")
ax.text(6.1, 7.4, "added by DetailHOI", fontsize=6.3, color=INK2,
        ha="left", va="center")

fig.savefig(f"{OUT}/architecture.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT}/architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("ok")
