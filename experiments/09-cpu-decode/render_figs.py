#!/usr/bin/env python3
"""Render Part 9 figures from the verified study numbers (see benchmarks/09-cpu-decode).

  fig9a  measured F16 decode speed, CPU and GPU, on both machines, with the ratio overlaid
  fig9b  M4 Pro roofline fit: ms/token vs weight GB, CPU and Metal, Stage 5 medians
  fig9c  byte-identical bf16 vs F16 decode on the EPYC slice and the M4 Pro CPU
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/mpathak/code/inference-wall/assets/figures"
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

# ---------------- fig9a: measured decode speed, F16, both machines ----------------
# F16 decode tok/s, same engine and weights. Server slice: closeout_ab.md (CPU mean of two
# rounds, A10G via CUDA). M4 Pro: stage5_summary.txt median of three interleaved rounds.
groups = ["Server slice\n(CPU and GPU on separate memory)", "MacBook M4 Pro\n(CPU and GPU share one memory)"]
cpu_tok = [4.035, 21.68]
gpu_tok = [54.58, 22.24]
ratios  = ["13.5x", "1.01x"]
x = np.arange(len(groups))
w = 0.32

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
bc = ax.bar(x - w / 2, cpu_tok, w, color=MUTED, label="CPU", zorder=3)
bg = ax.bar(x + w / 2, gpu_tok, w, color=BLUE,  label="GPU", zorder=3)
for bars, vals, col in ((bc, cpu_tok, MUTED), (bg, gpu_tok, BLUE)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center",
                fontsize=10.5, fontweight=700, color=col)
ax.text(0, 61, "GPU 13.5x faster", ha="center", fontsize=12, fontweight=700, color=TXT)
ax.text(1, 29, "GPU 1.01x faster", ha="center", fontsize=12, fontweight=700, color=TXT)
ax.set_xticks(x, groups)
ax.set_ylim(0, 68)
ax.set_ylabel("decode tok/s, F16 weights", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_title("Same model, same engine: the gap tracks the memory, not the math",
             fontsize=12, fontweight=700, loc="left", color=TXT)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9a-gap-vs-bandwidth.png", facecolor=BG)
print("wrote fig9a")

# ---------------- fig9b: M4 Pro roofline fit ----------------
gb = np.array([3.013, 3.563, 4.622, 8.666])
cpu_ms = np.array([19.19, 21.69, 26.49, 46.13])
mtl_ms = np.array([17.74, 19.87, 23.98, 44.97])
cpu_bw, cpu_t0 = 209.2, 4.63
mtl_bw, mtl_t0 = 204.9, 2.40
xx = np.linspace(0, 9.0, 50)

fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
ax.plot(xx, xx / cpu_bw * 1000 + cpu_t0, "-", color=BLUE, lw=1.6, zorder=2)
ax.plot(xx, xx / mtl_bw * 1000 + mtl_t0, "-", color=GREEN, lw=1.6, zorder=2)
ax.plot(gb, cpu_ms, "o", color=BLUE, ms=7, zorder=3, label="CPU (10 P-cores)")
ax.plot(gb, mtl_ms, "s", color=GREEN, ms=6.5, zorder=3, label="GPU (Metal)")
for g, m, lbl in zip(gb[:3], cpu_ms[:3], ["Q4_K_M", "Q6_K", "Q8_0"]):
    ax.text(g, m + 2.6, lbl, ha="center", fontsize=9.5, color=SOFT, style="italic")
ax.text(8.666 - 0.35, 45.5, "F16", ha="right", fontsize=9.5, color=SOFT, style="italic")
ax.text(9.2, 50.0, "1/slope = 209 GB/s", color=BLUE, fontsize=10.5,
        fontweight=700, ha="left", va="center")
ax.text(9.2, 43.6, "1/slope = 205 GB/s", color=GREEN, fontsize=10.5,
        fontweight=700, ha="left", va="center")
ax.text(0.15, 44, "same slope on both engines,\nthe intercepts differ by about 2 ms",
        fontsize=10.5, style="italic", color=SOFT)
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 55)
ax.set_xlabel("weight bytes streamed per token (GB)", fontsize=12, fontweight="bold")
ax.set_ylabel("ms per token", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.set_title("On unified memory both engines decode on the same line",
             fontsize=12, fontweight=700, loc="left", color=TXT)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9b-roofline-fit.png", facecolor=BG)
print("wrote fig9b")

# ---------------- fig9c: bf16 vs F16, byte-identical ----------------
machines = ["Server slice CPU\n(33 GB/s memory)", "MacBook M4 Pro CPU\n(238 GB/s memory)"]
bf16 = [4.005, 3.92]
f16  = [4.035, 20.45]
x = np.arange(2)
w = 0.32

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
bb = ax.bar(x - w / 2, bf16, w, color=MUTED, label="bf16 file", zorder=3)
bf = ax.bar(x + w / 2, f16,  w, color=BLUE,  label="F16 file", zorder=3)
for b, v in zip(bb, bf16):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center",
            fontsize=10.5, fontweight=700, color=MUTED)
for b, v in zip(bf, f16):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center",
            fontsize=10.5, fontweight=700, color=BLUE)
ax.text(0, 8.6, "1.01x", ha="center", fontsize=12, fontweight=700, color=TXT)
ax.text(1, 23.4, "5.2x", ha="center", fontsize=12, fontweight=700, color=TXT)
ax.text(0.5, 14.0, "same model, both files exactly the same size",
        ha="center", fontsize=10.5, style="italic", color=SOFT)
ax.set_xticks(x, machines)
ax.set_ylim(0, 27)
ax.set_ylabel("decode tok/s", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.set_title("Byte-identical files: a slow kernel only shows on fast memory",
             fontsize=12, fontweight=700, loc="left", color=TXT)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9c-bf16-vs-f16.png", facecolor=BG)
print("wrote fig9c")

# ---------------- Figure D: batched decode, same engine on both machines ----------------
# llama-batched-bench, npp=128, ntg=128, aggregate generation tok/s (S_TG) per cell.
# Sources: benchmarks/09-cpu-decode/batched_bench.md (EPYC slice, cores 0-7 pinned) and
# batched_bench_gpu.md (A10G via CUDA).
batch_cpu = [1, 2, 4, 8, 16, 32]
cpu_bf16  = [3.53, 6.70, 12.79, 20.06, 24.77, 26.28]
cpu_q4    = [9.17, 14.98, 19.93, 21.55, 25.24, 26.13]
batch_gpu = [1, 2, 4, 8, 16, 32, 64, 128]
gpu_f16   = [53.64, 93.19, 158.16, 251.89, 359.76, 442.60, 513.59, 552.46]
gpu_q4    = [117.21, 209.11, 302.78, 390.45, 622.24, 780.82, 856.06, 885.23]

fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
ax.plot(batch_gpu, gpu_q4,  "-o", color=GREEN, lw=2.0, ms=4.5, label="A10G, Q4")
ax.plot(batch_gpu, gpu_f16, "-o", color=BLUE,  lw=2.0, ms=4.5, label="A10G, F16")
ax.plot(batch_cpu, cpu_q4,  "--s", color=GREEN, lw=1.8, ms=4.5, label="server CPU, Q4")
ax.plot(batch_cpu, cpu_bf16, "--s", color=BLUE, lw=1.8, ms=4.5, label="server CPU, bf16")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(batch_gpu); ax.set_xticklabels([str(b) for b in batch_gpu])
ax.set_yticks([3, 10, 30, 100, 300, 1000]); ax.set_yticklabels(["3", "10", "30", "100", "300", "1,000"])
ax.set_xlim(0.8, 170); ax.set_ylim(2.5, 1400)
ax.set_xlabel("concurrent requests", fontsize=12, fontweight="bold")
ax.set_ylabel("aggregate decode tok/s", fontsize=12, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.annotate("both weight sizes meet at ~26 tok/s:\nthe CPU's math ceiling", xy=(32, 26.2),
            xytext=(44, 9), fontsize=10, style="italic", color=SOFT,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax.text(1.0, 1200, "batching lifts the CPU 7.4x and then stops; the GPU keeps climbing",
        fontsize=10.5, style="italic", color=SOFT)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9d-batched-decode.png", facecolor=BG)
print("wrote fig9d-batched-decode.png")

# ---------------- Figure E: decode gap vs model size, both machines ----------------
# Sources: benchmarks/09-cpu-decode/small_model_bench.md (server, same session) and
# benchmarks/09-cpu-decode/mac/small_model_bench.md (M4 Pro, run 1 and idle repeat).
# x = streamed GB (file bytes minus the untied embedding table), from the Mac file's table.
strm = [0.392, 0.525, 0.980, 0.988, 3.088, 8.655]
labels_e = ["0.5B Q4", "0.5B Q8", "1.5B Q4", "0.5B F16", "1.5B F16", "4B F16"]
srv_gap = [511.28/69.33, 471.46/56.17, 293.76/29.47, 339.59/31.42, 139.73/10.23, 54.34/3.80]
mac_r1  = [0.899, 0.863, 1.066, 1.034, 1.098, 0.987]
mac_r2  = [0.948, 0.901, 1.062, 0.992, 1.111, 1.036]
mac_gap = [(a+b)/2 for a,b in zip(mac_r1, mac_r2)]

fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
fig.patch.set_facecolor(BG)
style_ax(ax)
ax.axhline(503/33.2, color=BLUE, lw=1, ls=":", zorder=1)
ax.axhline(1.0, color=GREEN, lw=1, ls=":", zorder=1)
ax.plot(strm, srv_gap, "-o", color=BLUE, lw=2.0, ms=5, label="server slice: A10G over CPU", zorder=3)
ax.plot(strm, mac_gap, "-s", color=GREEN, lw=2.0, ms=5, label="M4 Pro: GPU over CPU", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.4, 1, 3, 9]); ax.set_xticklabels(["0.4", "1", "3", "9"])
ax.set_yticks([0.5, 1, 2, 5, 10, 20]); ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
ax.set_xlim(0.3, 12); ax.set_ylim(0.6, 25)
ax.set_xlabel("weight bytes streamed per token (GB)", fontsize=12, fontweight="bold")
ax.set_ylabel("GPU decode speed / CPU decode speed", fontsize=12, fontweight="bold")
ax.text(0.31, 503/33.2*1.12, "bandwidth ratio 15x", fontsize=9.5, style="italic", color=BLUE)
ax.text(0.31, 1.0*1.1, "bandwidth ratio 1x", fontsize=9.5, style="italic", color=GREEN)
ax.annotate("GPU's 1 ms fixed cost is over half\nof a 0.4 GB token: gap halves", xy=(0.392, srv_gap[0]),
            xytext=(0.9, 3.2), fontsize=9.5, style="italic", color=SOFT,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax.annotate("CPU wins on tiny quantized files", xy=(0.525, mac_gap[1]), xytext=(1.3, 0.68),
            fontsize=9.5, style="italic", color=SOFT, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
ax.legend(loc="center right", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9e-gap-vs-size.png", facecolor=BG)
print("wrote fig9e-gap-vs-size.png")
