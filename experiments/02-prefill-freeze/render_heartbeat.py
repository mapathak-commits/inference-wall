#!/usr/bin/env python3
"""
Render Part 2's step-heartbeat figure straight from the captured trace.

The engine runs one step at a time; the 8 full-attention layers fire their flash
kernels once per step in EVERY step type (prefill slice, mixed, pure decode), so
clustering flash kernels reconstructs the engine-step cadence across the whole
capture. The figure shows the last seconds of the injected prefill (steps ~39 ms
apart: each step is carrying a prefill slice alongside the victims' tokens, the
"elevated but bounded" regime) and then the prefill finishing, after which steps
snap back to the ~22 ms pure-decode cadence. Real data, no schematic: every tick
is one engine step at its true timestamp.

Usage: gunzip the trace from benchmarks/02-prefill-freeze/, then
  python render_heartbeat.py chunked_prefill_trace.json
Writes fig2b-step-heartbeat.png in the series chart style.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GRID, AXIS = "#fcfcfb", "#e6e5e2", "#b8b7b3"
TXT, SOFT, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#e34948"

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cp_trace.json"
ev = json.load(open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
k.sort(key=lambda e: e["ts"])

# engine steps: flash kernels fire once per step; cluster on >5 ms gaps
fa = [e for e in k if "flash" in e["name"] or "varlen" in e["name"]]
steps = [[fa[0]]]
for e in fa[1:]:
    if e["ts"] - steps[-1][-1]["ts"] > 5000:
        steps.append([e])
    else:
        steps[-1].append(e)

# prefill-path kernels mark which steps carry a prefill slice
pre = [e for e in k if e["name"].startswith(("chunk_gated_delta_rule", "chunk_fwd"))]
pre_end = pre[-1]["ts"] + pre[-1]["dur"]

t_end = k[-1]["ts"] + k[-1]["dur"]
WINDOW_S = 1.25  # the steady mixed-phase cadence + the decode tail
t_lo = t_end - WINDOW_S * 1e6
# snap the window start to the first step inside it, so there is no empty lead-in
starts_all = [s[0]["ts"] for s in steps if s[0]["ts"] >= t_lo]
t_lo = starts_all[0] - 15_000

marks = []
for s in steps:
    ts = s[0]["ts"]
    if ts < t_lo:
        continue
    kind = "mixed" if ts < pre_end else "decode"
    marks.append(((ts - t_lo) / 1e3, kind))

fig, ax = plt.subplots(figsize=(8.2, 2.5), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_yticks([])
plt.rcParams.update({"font.family": "sans-serif"})

# shade the two regimes
switch = (pre_end - t_lo) / 1e3
ax.axvspan(0, switch, color=RED, alpha=0.05, zorder=0)
ax.axvline(switch, color=SOFT, lw=1.2, ls=(0, (4, 3)), zorder=2)

# one tick per engine step
for x, kind in marks:
    ax.vlines(x, 0.22, 0.72, color=BLUE if kind == "decode" else RED,
              lw=2.2, zorder=3)

# cadence annotations under each regime
mixed_x = [x for x, kd in marks if kd == "mixed"]
dec_x = [x for x, kd in marks if kd == "decode"]
ax.text((mixed_x[0] + mixed_x[-1]) / 2, 0.06,
        "prefill still running: a step every ~39 ms\n"
        "(each carries a prefill slice + every victim's token)",
        ha="center", fontsize=10, color=RED, style="italic")
ax.text((dec_x[0] + dec_x[-1]) / 2, 0.06,
        "prefill done: back to a step\nevery ~22 ms",
        ha="center", fontsize=10, color=BLUE, style="italic")
ax.text(switch, 0.97, "the injected prompt finishes", ha="center", fontsize=10.5,
        fontweight=700, color=TXT, va="top")

# bracket one mixed-phase interval and one decode interval to make the cadence legible
def bracket(x1, x2, y, color, label):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.4))
    ax.text((x1 + x2) / 2, y + 0.055, label, ha="center", fontsize=9.5,
            fontweight=700, color=color)

if len(mixed_x) >= 3:
    ivals = sorted(range(len(mixed_x) - 1),
                   key=lambda i: mixed_x[i + 1] - mixed_x[i])
    mid = ivals[len(ivals) // 2]
    bracket(mixed_x[mid], mixed_x[mid + 1], 0.80, RED,
            f"{mixed_x[mid+1]-mixed_x[mid]:.0f} ms")
if len(dec_x) >= 6:
    bracket(dec_x[4], dec_x[5], 0.80, BLUE, f"{dec_x[5]-dec_x[4]:.0f} ms")

ax.set_xlim(0, WINDOW_S * 1e3)
ax.set_ylim(0, 1.02)
ax.set_xlabel("time (ms); each tick is one engine step, reconstructed from the "
              "trace's full-attention kernels", fontsize=11, color=SOFT)
fig.tight_layout()
fig.savefig("fig2b-step-heartbeat.png", facecolor=BG)
print(f"wrote fig2b-step-heartbeat.png ({len(marks)} steps shown, "
      f"{len(mixed_x)} mixed / {len(dec_x)} pure decode)")
