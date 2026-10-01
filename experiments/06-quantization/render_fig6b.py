#!/usr/bin/env python3
"""Render fig6b: output tok/s vs offered rate, 4B fp16 vs 9B AWQ.
Values are the warm-sweep "Output token throughput (tok/s)" lines:
4B from Part 1 (benchmarks/01-hit-the-wall), 9B from benchmarks/06-quantization.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/mpathak/code/inference-wall/assets/figures"

rates = ["1", "2", "4", "6", "8", "16", "inf"]
x = list(range(len(rates)))
tok_4b = [126.26, 248.99, 482.34, 692.51, 815.10, 902.28, 1092.09]
tok_9b = [126.08, 248.22, 477.67, 619.26, 666.77, 775.09, 821.37]

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(x, tok_4b, "o-", color="#1f6fb2", lw=2, ms=6, label="4B fp16")
ax.plot(x, tok_9b, "o-", color="#2ca777", lw=2, ms=6, label="9B AWQ 4-bit")

ax.set_xticks(x)
ax.set_xticklabels(rates)
ax.set_xlabel("Offered rate (requests / second)")
ax.set_ylabel("Output tok/s")
ax.set_ylim(0, 1300)
# right headroom so the end labels sit beyond the last marker, not on the curves
ax.set_xlim(-0.2, len(rates) - 1 + 1.25)
ax.grid(True, alpha=0.3)

# end-of-curve labels, placed to the right of the final marker and nudged apart
ax.annotate("4B  1,092", xy=(x[-1], tok_4b[-1]), xytext=(x[-1] + 0.15, 1140),
            color="#1f6fb2", fontsize=10, va="center", ha="left")
ax.annotate("9B  821", xy=(x[-1], tok_9b[-1]), xytext=(x[-1] + 0.15, 760),
            color="#2ca777", fontsize=10, va="center", ha="left")
ax.annotate("nearly identical\nbelow the knee", xy=(1.5, 190), xytext=(2.7, 90),
            fontsize=9, color="#555", ha="center",
            arrowprops=dict(arrowstyle="->", color="#999"))

fig.tight_layout()
fig.savefig(f"{OUT}/fig6b-throughput-curves.png", dpi=130)
print("wrote fig6b")
