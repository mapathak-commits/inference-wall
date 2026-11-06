#!/usr/bin/env python3
"""Render Part 11 figures from the measured numbers in benchmarks/11-cuda-graphs.
  fig11a  ms per token against concurrent requests for three models, CUDA graphs on and off
  fig11b  share of each token the GPU spends working, single request, graphs on and off
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/mpathak/code/inference-wall/assets/figures"
BG, GRID, AXIS = "#fcfcfb", "#e6e5e2", "#b8b7b3"
TXT, SOFT, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, GREEN, RED, AMBER = "#2a78d6", "#1baf7a", "#e34948", "#e07a3f"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.edgecolor": AXIS,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TXT, "axes.labelcolor": SOFT})
def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)

# median ms per output token, vllm bench serve, 128 in / 256 out (study.log, study_nograph.log)
conc = [1, 4, 16, 64]
data = {
    "Qwen2.5-0.5B": {"on": [3.25, 4.08, 4.13, 5.64], "off": [14.24, 14.56, 14.34, 16.80]},
    "Qwen2.5-3B":   {"on": [14.89, 16.33, 16.72, 22.62], "off": [28.81, 28.94, 29.00, 30.79]},
    "Qwen3.5-4B":   {"on": [19.98, 20.84, 24.26, 41.06], "off": [29.57, 28.87, 29.59, 41.14]},
}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=200, sharey=False)
fig.patch.set_facecolor(BG)
for ax, (name, d) in zip(axes, data.items()):
    style_ax(ax)
    x = np.arange(len(conc))
    ax.plot(x, d["off"], "-o", color=AMBER, lw=2, ms=5, label="CUDA graphs off")
    ax.plot(x, d["on"], "-o", color=BLUE, lw=2, ms=5, label="CUDA graphs on")
    ax.set_xticks(x, [str(c) for c in conc]); ax.set_ylim(0, max(d["off"] + d["on"]) * 1.25)
    ax.set_title(name, fontsize=11.5, fontweight="bold", loc="left")
    ax.set_xlabel("concurrent requests", fontsize=10.5)
    ax.text(0, d["off"][0] * 1.07, f"{d['off'][0]/d['on'][0]:.1f}x", ha="center", fontsize=10, fontweight="bold", color=TXT)
axes[0].set_ylabel("ms per token", fontsize=11.5, fontweight="bold")
axes[0].legend(loc="lower right", frameon=False, fontsize=9.5)
fig.suptitle("Without CUDA graphs, time per token barely moves with load until the GPU's own work catches up",
             fontsize=11.5, fontweight="bold", x=0.01, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig11a-graphs-vs-concurrency.png", facecolor=BG); print("wrote fig11a")

# GPU working share = traced GPU kernel time per token / unprofiled ms per token, one request
models = ["Qwen2.5-0.5B", "Qwen3.5-4B"]
gpu_ms = [3.04, 19.58]; on_ms = [3.25, 19.98]; off_ms = [14.24, 29.57]
on_share = [g / t for g, t in zip(gpu_ms, on_ms)]; off_share = [g / t for g, t in zip([3.03, 19.62], off_ms)]
fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200); fig.patch.set_facecolor(BG); style_ax(ax)
x = np.arange(2); w = 0.32
b1 = ax.bar(x - w/2, [100*v for v in on_share], w, color=BLUE, label="CUDA graphs on", zorder=3)
b2 = ax.bar(x + w/2, [100*v for v in off_share], w, color=AMBER, label="CUDA graphs off", zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center",
                fontsize=10.5, fontweight="bold", color=b.get_facecolor())
ax.set_xticks(x, [f"{m}\n{g:.1f} ms of GPU work per token" for m, g in zip(models, gpu_ms)])
ax.set_ylim(0, 115); ax.set_ylabel("share of each token the GPU is working", fontsize=11, fontweight="bold")
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_title("Same GPU work either way; without graphs the GPU waits on the CPU", fontsize=12, fontweight="bold", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig11b-gpu-busy.png", facecolor=BG); print("wrote fig11b")
