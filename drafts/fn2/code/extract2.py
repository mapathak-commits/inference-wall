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
PROMPT = "The cat sat on the keyboard again."   # funny + a clear subject-tracking head
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

# the "interpretable" contrast head: the one with the strongest real word-pair
# links, i.e. big off-diagonal weight that is NOT self, the immediate neighbor,
# or the sink column (token 0). This is the head that shows meaning being wired
# up (e.g. several later words pointing back at the subject).
mask = np.ones((L, H))
best_link = -1.0
int_l, int_h = 0, 0
for a in range(L):
    for b in range(H):
        m = attn[a, b]
        score = 0.0
        for i in range(2, S):
            for j in range(1, i - 1):    # skip self, neighbor, token 0
                score = max(score, m[i, j])
        if score > best_link:
            best_link, int_l, int_h = score, a, b

print(f"prompt: {PROMPT!r}   tokens: {tokens}   ({S} tokens)")
print(f"layers={L} heads={H}\n")
print(f"strongest sink head: layer {li}, head {hi}  "
      f"-> {sink[li, hi]:.0%} of last token's attention on token 0 ({tokens[0]!r})")
print(f"interpretable head: layer {int_l}, head {int_h}  "
      f"-> strongest word-pair link {best_link:.2f}")
# report that head's links back to the subject
m = attn[int_l, int_h]
for qi in range(2, S):
    row = m[qi].copy(); row[qi] = 0; row[0] = 0
    kj = int(row.argmax())
    if row[kj] > 0.3:
        print(f"    {tokens[qi]!r:>10} -> {tokens[kj]!r:<10} {row[kj]:.2f}")
print()
deep = attn[L // 2:, :, -1, 0]
print(f"heads in the deep half with >50% of last-token attention on token 0: "
      f"{(deep > 0.5).mean():.0%}")

np.save(f"{OUT}/attn.npy", attn)
np.save(f"{OUT}/hidden.npy", hidden)
meta = {
    "model": MODEL, "prompt": PROMPT, "tokens": tokens, "seq_len": S,
    "n_layers": L, "n_heads": H,
    "sink_head": [int(li), int(hi)], "sink_strength": float(sink[li, hi]),
    "interp_head": [int(int_l), int(int_h)], "interp_link": float(best_link),
    "attn_row_sum_check": float(attn[li, hi, -1].sum()),
    "hidden_dim": int(hidden.shape[-1]),
}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
