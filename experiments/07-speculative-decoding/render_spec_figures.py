#!/usr/bin/env python3
"""
Render Part 7's data figures in the series' chart style (palette and typography
matched to draft1/figures.html): warm off-white bg #fcfcfb, blue/green/red series,
grey grid, italic annotations, shaded past-the-knee region.

  fig7a  the headline: throughput vs offered rate, three arms (off / k3 / k5)
  fig7b  the seat collapse: max concurrency + flooded running batch, off vs k5
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GRID, AXIS = "#fcfcfb", "#e6e5e2", "#b8b7b3"
TXT, SOFT, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#e34948"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TXT, "axes.labelcolor": SOFT,
})

def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

# ---------------- Figure A: the inversion ----------------
rates = ["1", "2", "4", "6", "8", "16", "flood"]
x = range(len(rates))
off = [126, 249, 480, 693, 809, 905, 1096]
k5  = [126, 247, 443, 472, 494, 487, 473]
k3  = [126, 247, 466, 500, 504, 528, 542]

fig, ax = plt.subplots(figsize=(8.2, 3.1), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
ax.axvspan(2.5, 6.4, color="#0b0b0b", alpha=0.035, zorder=0)
ax.plot(x, off, "-o", color=BLUE, lw=2.2, ms=5, zorder=3)
ax.plot(x, k3, "-o", color=GREEN, lw=2.0, ms=4.5, zorder=3)
ax.plot(x, k5, "-o", color=RED, lw=2.0, ms=4.5, zorder=3)
ax.annotate("spec off ≈ 1,096", (6, 1096), xytext=(-12, 6),
            textcoords="offset points", ha="right", fontsize=11.5,
            fontweight=700, color=BLUE)
ax.annotate("k3 ≈ 542", (6, 542), xytext=(-4, 12),
            textcoords="offset points", ha="right", fontsize=11, fontweight=700,
            color=GREEN)
ax.annotate("k5 ≈ 473", (6, 473), xytext=(-4, -20),
            textcoords="offset points", ha="right", fontsize=11, fontweight=700,
            color=RED)
ax.text(1.5, 50, "indistinguishable below the knee", fontsize=10.5,
        style="italic", color=SOFT)
ax.text(-0.15, 1210, "speculation on: ceiling halves, knee moves to rate 3 to 4",
        fontsize=10.5, style="italic", color=SOFT)
ax.set_xticks(list(x), rates)
ax.set_yticks([0, 300, 600, 900, 1200])
ax.set_ylim(0, 1310)
ax.set_xlim(-0.3, 6.4)
ax.set_xlabel("offered rate (requests / second)", fontsize=12, fontweight="bold")
ax.set_ylabel("output tok/s", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("fig7a-sweep-inversion.png", facecolor=BG)
print("wrote fig7a-sweep-inversion.png")

# ---------------- Figure B: the seat collapse ----------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.6), dpi=200)
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.30, wspace=0.32)

panels = [
    ("max concurrency at startup",
     [("spec off", 83.71, BLUE, "83.7x"), ("k5", 24.64, RED, "24.6x")], 112),
    ("running batch under flood",
     [("spec off", 146, BLUE, "146 reqs"), ("k5", 28, RED, "28 reqs")], 196),
]
for ax, (title, bars, xmax) in zip(axes, panels):
    style_ax(ax)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.grid(axis="y", visible=False)
    names = [b[0] for b in bars][::-1]
    vals = [b[1] for b in bars][::-1]
    cols = [b[2] for b in bars][::-1]
    lbls = [b[3] for b in bars][::-1]
    ax.barh(names, vals, color=cols, height=0.55, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=11.5, fontweight=700, color=TXT, loc="left")
    for name, v, c, lbl in zip(names, vals, cols, lbls):
        ax.text(v + xmax * 0.025, name, lbl, va="center", fontsize=11,
                fontweight=700, color=c)
    ax.tick_params(labelsize=10.5)
fig.text(0.10, 0.05,
         "k5 keeps 6 state slots per request, so each request costs 6 seats "
         "from the same memory budget",
         fontsize=10.5, style="italic", color=SOFT)
fig.savefig("fig7b-seat-collapse.png", facecolor=BG)
print("wrote fig7b-seat-collapse.png")
