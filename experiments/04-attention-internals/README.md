# Experiment 04 — attention internals

Scripts behind [Part 4](../../articles/part-4.md): open one prompt on a CPU in full
precision, keep every intermediate value, and read the attention weights and running
state the fast serving path throws away. No GPU needed; it runs in a few seconds.

| Script | What it does | Produces |
|---|---|---|
| `observe.py` | Runs GPT-2 (HuggingFace `transformers`, eager attention, fp32, CPU) on `"The cat sat on the keyboard again."` with `output_attentions` / `output_hidden_states`, and pulls out the per-head attention weights and per-layer running state | the raw arrays the two other scripts read |
| `plot.py` | Renders the three figures in the post | `assets/figures/fn2-attention-grids.png` (layer 4 head 3 vs. layer 5 head 1), `fn2-sink-grid.png` (the 12×12 sink map), `fn2-magnitude.png` (peak per-token state magnitude) |
| `extract2.py` | Cross-checks both effects on a handful of other models (OPT, Qwen) so the single clean GPT-2 example is not a one-off | the sink-score and magnitude-ratio numbers quoted in the post |

The claims each figure backs:

- **Attention sink** — a deep head (layer 5, head 1) sends its entire last-token pie to
  the first token; across the back half of the network, 92% of heads send more than half
  their attention there. (`fn2-attention-grids.png`, `fn2-sink-grid.png`.)
- **Massive activation** — one token's internal-state magnitude (L2 norm) peaks ~12x the
  median token across all layers. (`fn2-magnitude.png`.)

Both effects reproduce on OPT and Qwen via `extract2.py`.
