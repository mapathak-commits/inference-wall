#!/usr/bin/env python3
"""
Render Part 2's decode-heartbeat figure straight from the captured trace: every
decode kernel in the measurement window as a tick on a timeline, so the stretched
gaps (the interleaved prefill chunks) are visible as real data, not a schematic.

Usage: gunzip the trace from benchmarks/02-prefill-freeze/, then
  python render_heartbeat.py chunked_prefill_trace.json
Writes fig2b-decode-heartbeat.png in the series chart style.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GRID, AXIS = "#fcfcfb", "#e6e5e2", "#b8b7b3"
TXT, SOFT, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, RED = "#2a78d6", "#e34948"

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cp_trace.json"
ev = json.load(open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
k.sort(key=lambda e: e["ts"])
dec = [e for e in k if e["name"].startswith("fused_recurrent_gated_delta_rule")]

t0 = dec[0]["ts"]
ticks_ms = [(e["ts"] - t0) / 1e3 for e in dec]
span = ticks_ms[-1]

# gaps (end of one decode kernel to start of the next), for annotation
gaps = []
for a, b in zip(dec, dec[1:]):
    g = (b["ts"] - (a["ts"] + a["dur"])) / 1e3
    mid = (a["ts"] + a["dur"] - t0) / 1e3
    gaps.append((mid, g))
big = sorted(gaps, key=lambda x: -x[1])
# keep at most 2 annotated gaps, at least 60 ms apart so labels don't collide
sel = []
for mid, g in big:
    if all(abs(mid - m2) > 60 for m2, _ in sel):
        sel.append((mid, g))
    if len(sel) == 2:
        break
big = sel

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": AXIS, "xtick.color": MUTED,
    "text.color": TXT, "axes.labelcolor": SOFT,
})
fig, ax = plt.subplots(figsize=(8.2, 2.3), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_yticks([])

# every decode kernel is one tick: the heartbeat
ax.vlines(ticks_ms, 0.25, 0.75, color=BLUE, lw=0.7, alpha=0.9)

# highlight the three widest gaps
for mid, g in big:
    ax.annotate("", xy=(mid + g, 0.5), xytext=(mid, 0.5),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(mid + g / 2, 0.86, f"{g:.1f} ms", ha="center", fontsize=10,
            fontweight=700, color=RED)

ax.text(0.01, 0.985,
        "each tick is one decode kernel, the victims' heartbeat; "
        "the wide gaps are prefill chunks taking their turn",
        transform=ax.transAxes, fontsize=10.5, style="italic", color=SOFT,
        va="top")
ax.set_xlim(-5, span + 5)
ax.set_ylim(0, 1.05)
ax.set_xlabel(f"time (ms), chunked prefill ON: {len(dec)} decode kernels over "
              f"{span:.0f} ms, never a full halt", fontsize=11.5)
fig.tight_layout()
fig.savefig("fig2b-decode-heartbeat.png", facecolor=BG)
print(f"wrote fig2b-decode-heartbeat.png ({len(dec)} ticks, span {span:.1f} ms)")
