"""Plot the extracted arrays. Runs in the perfetto venv (has matplotlib)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/home/mpathak/code/research/tools/tailfin/fn-observe-prototype"
attn = np.load(f"{D}/attn.npy")      # [L, H, S, S]
hidden = np.load(f"{D}/hidden.npy")  # [L+1, S, D]
meta = json.load(open(f"{D}/meta.json"))
tokens = meta["tokens"]
S = meta["seq_len"]

PAPER = "#faf7f2"

# --- Figure 1: attention heatmap, the local head vs. the sink head ---
# Two readability fixes over a plain 0-1 heatmap:
#   1. mask the upper triangle (causal: those cells are exactly 0 by construction,
#      not "no attention learned") so real zeros don't read as a checkerboard.
#   2. use a gamma<1 power norm + per-cell numbers so mid-range weights are legible
#      instead of everything snapping to pure black or yellow.
from matplotlib.colors import PowerNorm
il, ih = meta["interp_head"]
sl, sh = meta["sink_head"]
picks = [(il, ih, "an interpretable head\n(later words look back at 'cat')"),
         (sl, sh, "the sink head\n(every word dumps onto 'The')")]
labels = [t.strip() for t in tokens]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.patch.set_facecolor(PAPER)
norm = PowerNorm(gamma=0.45, vmin=0, vmax=1)
for ax, (L, H, sub) in zip(axes, picks):
    m = attn[L, H].copy()
    m[np.triu_indices(S, k=1)] = np.nan          # hide future positions
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(PAPER)                           # masked cells = paper color
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="equal")
    for i in range(S):                            # print the weight in each live cell
        for j in range(i + 1):
            v = attn[L, H, i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6.5, color="white" if v < 0.6 else "black")
    ax.set_title(f"layer {L}, head {H}\n{sub}", fontsize=10.5)
    ax.set_xticks(range(S)); ax.set_yticks(range(S))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("attends to", fontsize=9)
    ax.set_facecolor(PAPER)
axes[0].set_ylabel("token doing the attending", fontsize=9)
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03,
                    ticks=[0, 0.1, 0.25, 0.5, 1.0])
cbar.set_label("attention weight (nonlinear scale)")
fig.suptitle('GPT-2 attention on "%s"' % meta["prompt"], fontsize=12, y=1.02)
fig.savefig(f"{D}/fig_attention.png", dpi=130, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

# --- Figure 2: hidden-state norm growing across layers ---
norms = np.linalg.norm(hidden, axis=-1)  # [L+1, S]
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
for s in range(S):
    ax.plot(range(hidden.shape[0]), norms[:, s], marker="o", ms=3,
            label=tokens[s].strip())
ax.set_yscale("log")
ax.set_xlabel("layer (0 = embedding, %d = final)" % (hidden.shape[0] - 1))
ax.set_ylabel("hidden-state L2 norm (log scale)")
ax.set_title("One token's residual norm spikes far above the rest (per token)")
ax.legend(fontsize=7, ncol=2, frameon=False)
fig.savefig(f"{D}/fig_hidden_norm.png", dpi=130, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

# --- Figure 3: sink strength across every (layer, head) ---
# how much of the LAST token's attention each head puts on token 0
sink = attn[:, :, -1, 0]  # [L, H]
L, H = sink.shape
fig, ax = plt.subplots(figsize=(7.5, 5.2))
fig.patch.set_facecolor(PAPER)
im = ax.imshow(sink, cmap="magma", vmin=0, vmax=1, aspect="auto")
ax.set_xlabel("head")
ax.set_ylabel("layer (0 = first)")
ax.set_xticks(range(H)); ax.set_yticks(range(L))
ax.set_title("How much of the last token's attention each head\n"
             "parks on token 0 (%r)" % tokens[0].strip())
# mark the strongest sink head
sl, sh = meta["sink_head"]
ax.add_patch(plt.Rectangle((sh - .5, sl - .5), 1, 1, fill=False,
                           edgecolor="cyan", lw=2))
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label="fraction of last-token attention on token 0")
fig.savefig(f"{D}/fig_sink_grid.png", dpi=130, facecolor=PAPER, bbox_inches="tight")
plt.close(fig)

print("wrote fig_attention.png, fig_hidden_norm.png, fig_sink_grid.png")
print("attention row-sum check (should be ~1.0):",
      round(float(attn[0, 0, -1].sum()), 4))
print("last-layer norm range:", round(float(norms[-1].min()), 1),
      "to", round(float(norms[-1].max()), 1))
print("deep-half heads with >50%% on token 0:",
      round(float((sink[L // 2:] > 0.5).mean()) * 100), "%")
