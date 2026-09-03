"""
FN2 rewrite: one simple prompt, one model (GPT-2), and instead of hand-picking
heads, SCAN all 144 (layer, head) pairs and report which ones actually show the
sink so the writeup can name them precisely.

For each head we score the "sink strength" = how much of the LAST token's
attention (the token doing the predicting) lands on token 0. We then surface:
  - the head with the strongest sink   (the one to feature)
  - a head with diffuse / local attention (a contrast)
Saves attn.npy, hidden.npy, meta.json for plotting in the other venv.
"""
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
PROMPT = "The cat sat on the mat."   # simpler: no coreference bait, 7 tokens
OUT = "/home/mpathak/code/research/tools/tailfin/fn-observe-prototype"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", torch_dtype=torch.float32
).eval()

enc = tok(PROMPT, return_tensors="pt")
tokens = [tok.decode([t]) for t in enc.input_ids[0].tolist()]
S = len(tokens)
with torch.no_grad():
    out = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=True)

attn = np.stack([a[0].numpy() for a in out.attentions])       # [L, H, S, S]
hidden = np.stack([h[0].numpy() for h in out.hidden_states])  # [L+1, S, D]
L, H, _, _ = attn.shape

# sink strength per head: last token's attention weight onto token 0
sink = attn[:, :, -1, 0]                        # [L, H]
li, hi = np.unravel_index(sink.argmax(), sink.shape)
# a diffuse/local contrast head: smallest sink among layer-0 heads
local_h = int(sink[0].argmin())

print(f"prompt: {PROMPT!r}   tokens: {tokens}   ({S} tokens)")
print(f"layers={L} heads={H}\n")
print(f"strongest sink head: layer {li}, head {hi}  "
      f"-> {sink[li, hi]:.0%} of last token's attention on token 0 ({tokens[0]!r})")
print(f"contrast (local) head: layer 0, head {local_h}  "
      f"-> only {sink[0, local_h]:.0%} on token 0")
print()
# how common is the sink? fraction of heads (deep half) with >50% on token 0
deep = attn[L // 2:, :, -1, 0]
print(f"heads in the deep half with >50% of last-token attention on token 0: "
      f"{(deep > 0.5).mean():.0%}")

np.save(f"{OUT}/attn.npy", attn)
np.save(f"{OUT}/hidden.npy", hidden)
meta = {
    "model": MODEL, "prompt": PROMPT, "tokens": tokens, "seq_len": S,
    "n_layers": L, "n_heads": H,
    "sink_head": [int(li), int(hi)], "sink_strength": float(sink[li, hi]),
    "local_head_layer0": local_h, "local_strength": float(sink[0, local_h]),
    "attn_row_sum_check": float(attn[li, hi, -1].sum()),
    "hidden_dim": int(hidden.shape[-1]),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
