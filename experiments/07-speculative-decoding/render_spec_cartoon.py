#!/usr/bin/env python3
"""
Flat editorial-style cartoon for Part 6, in the series' illustration palette
(paper #faf7f2, blue #2a78d6, navy ink, amber #e07a3f, green #1baf7a).

Scene: the big model L is a press whose single wide arm stamps a whole row of
token tiles in one pass. Ahead of it, a small amber scout S has sketched faint
dashed tiles (the guesses). Under the arm: the base tile blue, three guesses
verified solid green, and two rejected tiles tumbling off the track, crossed out.
One pass, several tokens; wrong guesses discarded.

A vector stand-in for the D9 illustration (see DIAGRAM-PROMPTS-P6.md for the
image-gen prompt matching d1-d8's style). No text in the image, per the series'
illustration rules.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

PAPER = "#faf7f2"
BLUE = "#2a78d6"
NAVY = "#1b2a45"
AMBER = "#e07a3f"
GREEN = "#1baf7a"
GRAY = "#b9b4ab"
SHADOW = "#d8d2c6"

fig, ax = plt.subplots(figsize=(14, 7.9), dpi=100)
fig.patch.set_facecolor(PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.9)
ax.axis("off")


def tile(x, y, w=1.0, h=1.0, color=BLUE, angle=0.0, dashed=False, alpha=1.0,
         face=True, lw=3.0, shadow=True):
    t = matplotlib.transforms.Affine2D().rotate_deg_around(x + w / 2, y + h / 2,
                                                           angle) + ax.transData
    if shadow and face:
        s = FancyBboxPatch((x + 0.07, y - 0.09), w, h,
                           boxstyle="round,pad=0.06,rounding_size=0.16",
                           fc=SHADOW, ec="none", alpha=0.8 * alpha, transform=t,
                           zorder=2)
        ax.add_patch(s)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.16",
                       fc=color if face else "none", ec=NAVY if face else color,
                       lw=lw, alpha=alpha,
                       ls=(0, (4, 3)) if dashed else "-", transform=t, zorder=3)
    ax.add_patch(p)


# ---- the track ----
ax.plot([0.4, 13.6], [2.55, 2.55], color=NAVY, lw=4, solid_capstyle="round",
        zorder=1)
ax.plot([0.4, 13.6], [2.38, 2.38], color=SHADOW, lw=2, zorder=1)

# ---- L: the big press on the left ----
# body
body = FancyBboxPatch((0.7, 2.8), 2.6, 3.1,
                      boxstyle="round,pad=0.08,rounding_size=0.35",
                      fc=BLUE, ec=NAVY, lw=4, zorder=4)
ax.add_patch(body)
bodyhi = FancyBboxPatch((0.95, 5.05), 2.1, 0.55,
                        boxstyle="round,pad=0.05,rounding_size=0.22",
                        fc="#5c97e2", ec="none", zorder=5)
ax.add_patch(bodyhi)
# eye
ax.add_patch(Circle((2.75, 4.6), 0.30, fc=PAPER, ec=NAVY, lw=3.5, zorder=6))
ax.add_patch(Circle((2.86, 4.55), 0.115, fc=NAVY, ec="none", zorder=7))
# the single wide arm sweeping over the row of tiles (the ONE pass)
arm = FancyBboxPatch((3.35, 4.45), 7.1, 0.62,
                     boxstyle="round,pad=0.06,rounding_size=0.3",
                     fc=BLUE, ec=NAVY, lw=3.5, zorder=5)
ax.add_patch(arm)
# stamp pads pressing down from the arm onto each verified position
for i in range(5):
    px = 4.15 + i * 1.32
    ax.plot([px, px], [4.42, 3.95], color=NAVY, lw=3.5, zorder=4,
            solid_capstyle="round")
    pad = FancyBboxPatch((px - 0.28, 3.75), 0.56, 0.24,
                         boxstyle="round,pad=0.03,rounding_size=0.1",
                         fc=NAVY, ec="none", zorder=4)
    ax.add_patch(pad)
# soft "pass" glow under the arm
glow = FancyBboxPatch((3.6, 2.75), 6.6, 1.75,
                      boxstyle="round,pad=0.1,rounding_size=0.4",
                      fc=BLUE, ec="none", alpha=0.08, zorder=1)
ax.add_patch(glow)

# ---- the tile row on the track ----
# base token, already the model's own (blue)
tile(3.85, 2.72, color=BLUE, angle=-1.5)
# three accepted guesses (green)
for i, ang in enumerate((1.2, -0.8, 1.8)):
    tile(5.17 + i * 1.32, 2.72, color=GREEN, angle=ang)
# sparkle ticks over accepted tiles
for i in range(3):
    cx = 5.17 + i * 1.32 + 0.5
    ax.plot([cx - 0.1, cx, cx + 0.18], [3.98, 3.86, 4.12], color=GREEN, lw=3,
            solid_capstyle="round", zorder=6)

# ---- two rejected tiles tumbling off the track ----
for (tx, ty, ang) in ((9.35, 1.15, -24), (10.6, 0.55, 18)):
    tile(tx, ty, color=GRAY, angle=ang, alpha=0.9)
    # cross-out strokes
    cx, cy = tx + 0.5, ty + 0.5
    r = 0.34
    for a in (45, -45):
        dx = r * math.cos(math.radians(a))
        dy = r * math.sin(math.radians(a))
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=AMBER, lw=4.5,
                solid_capstyle="round", zorder=6)
# little motion arcs where they fell
for (mx, my) in ((9.2, 2.35), (10.5, 2.1)):
    arc = matplotlib.patches.Arc((mx, my), 0.9, 0.6, angle=0, theta1=200,
                                 theta2=330, color=GRAY, lw=2.5, zorder=2)
    ax.add_patch(arc)

# ---- S: the small amber scout ahead, sketching guesses ----
sx, sy = 11.9, 3.0
# body
ax.add_patch(Circle((sx, sy + 0.55), 0.42, fc=AMBER, ec=NAVY, lw=3.5, zorder=6))
# eye
ax.add_patch(Circle((sx + 0.14, sy + 0.62), 0.07, fc=NAVY, ec="none", zorder=7))
# legs
ax.plot([sx - 0.18, sx - 0.30], [sy + 0.16, sy - 0.28], color=NAVY, lw=3.5,
        solid_capstyle="round", zorder=6)
ax.plot([sx + 0.14, sx + 0.30], [sy + 0.14, sy - 0.28], color=NAVY, lw=3.5,
        solid_capstyle="round", zorder=6)
# pencil arm
ax.plot([sx - 0.36, sx - 0.95], [sy + 0.5, sy + 0.06], color=NAVY, lw=3.5,
        solid_capstyle="round", zorder=6)
pencil = Polygon([(sx - 1.22, sy - 0.12), (sx - 0.88, sy + 0.16),
                  (sx - 0.98, sy - 0.24)], closed=True, fc=AMBER, ec=NAVY,
                 lw=2.5, zorder=6)
ax.add_patch(pencil)
# faint dashed proposal tiles being sketched (not yet verified)
tile(11.0, 2.72, color=AMBER, dashed=True, face=False, alpha=0.85, lw=3.2,
     shadow=False)
tile(12.35, 2.72, color=AMBER, dashed=True, face=False, alpha=0.5, lw=3.0,
     shadow=False)
# sketch squiggles
ax.plot([10.7, 10.55, 10.7], [4.0, 4.15, 4.3], color=AMBER, lw=2.5, alpha=0.7,
        solid_capstyle="round", zorder=5)

# subtle paper grain: sparse dots
import random
random.seed(6)
for _ in range(240):
    gx, gy = random.uniform(0, 14), random.uniform(0, 7.9)
    ax.plot(gx, gy, ".", color=NAVY, alpha=0.03, ms=random.uniform(1, 2.4))

fig.savefig("d9-speculation-cartoon.png", facecolor=PAPER,
            bbox_inches="tight", pad_inches=0.15)
print("wrote d9-speculation-cartoon.png")
