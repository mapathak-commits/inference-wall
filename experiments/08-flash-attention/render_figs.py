#!/usr/bin/env python3
"""Render Part 8 figures from S1 probe data and S3 census.
fig8a: prefill TTFT divergence (flash vs flex) across prompt length.
fig8b: kernel-family GPU-time split at 2k/8k/28k prefill, both backends.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/mpathak/code/inference-wall/assets/figures"

# ---- S1 prefill TTFT (seconds) ----
N       = [256, 1024, 2048, 4096, 8192, 16384, 24576, 30720]
flash   = [0.116, 0.280, 0.502, 0.992, 2.067, 4.591, 7.620, 10.181]
flex    = [0.119, 0.292, 0.544, 1.127, 2.540, 6.393, 11.704, 16.410]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N, flex,  "o-", color="#c1440e", lw=2, ms=6, label="FlexAttention (generic kernel)")
ax.plot(N, flash, "o-", color="#1f6fb2", lw=2, ms=6, label="FlashAttention")
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("Time to first token (seconds)")
ax.set_title("Prefill time diverges only once the prompt is long")
ax.set_ylim(-1.0, 18.5)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=False)
ax.annotate("3 ms apart\nat 256 tokens", xy=(500, 0.15), xytext=(4200, -0.4),
            fontsize=9, color="#555", va="center",
            arrowprops=dict(arrowstyle="->", color="#999"))
ax.annotate("6.2 s apart\nat 30,720 tokens", xy=(30720, 13.3), xytext=(15500, 10.0),
            fontsize=9, color="#555",
            arrowprops=dict(arrowstyle="->", color="#999"))
fig.tight_layout()
fig.savefig(f"{OUT}/fig8a-prefill-divergence.png", dpi=130)
print("wrote fig8a")

# ---- S3 kernel-family split (ms of GPU time) ----
# attention = the attention kernel only (flash_fwd_splitkv / triton_tem_fused_0)
lengths = ["2k", "8k", "28k"]
data = {
    "FlashAttention": {
        "attention": [22.4, 262.3, 3097.6],
        "matmul":    [438.6, 1664.1, 5756.3],
        "other":     [28.6, 108.0, 366.1],
    },
    "FlexAttention": {
        "attention": [51.0, 692.4, 8389.0],
        "matmul":    [438.4, 1663.4, 5754.8],
        "other":     [30.7, 116.9, 395.7],
    },
}
colors = {"attention": "#c1440e", "matmul": "#1f6fb2", "other": "#bbbbbb"}

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
for ax, backend in zip(axes, ["FlashAttention", "FlexAttention"]):
    d = data[backend]
    x = range(len(lengths))
    bottom = [0, 0, 0]
    for fam in ("matmul", "attention", "other"):
        vals = [v / 1000.0 for v in d[fam]]  # to seconds
        ax.bar(x, vals, bottom=bottom, color=colors[fam], label=fam,
               width=0.6, edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{l}\nprefill" for l in lengths])
    ax.set_title(backend)
    ax.set_ylim(0, 17.5)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("GPU time in profiler window (seconds)")
# label the attention share just above each 28k bar, inside the headroom
axes[0].annotate("attention 34%", xy=(2, 9.2), xytext=(2, 10.6), ha="center",
                 fontsize=9, color="#c1440e",
                 arrowprops=dict(arrowstyle="->", color="#c1440e", lw=0.8))
axes[1].annotate("attention 58%", xy=(2, 14.5), xytext=(2, 16.1), ha="center",
                 fontsize=9, color="#c1440e",
                 arrowprops=dict(arrowstyle="->", color="#c1440e", lw=0.8))
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index("attention"), labels.index("matmul"), labels.index("other")]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
fig.suptitle("Matmul time is identical between backends; only attention differs",
             y=0.91, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(f"{OUT}/fig8b-kernel-family-split.png", dpi=130)
print("wrote fig8b")
