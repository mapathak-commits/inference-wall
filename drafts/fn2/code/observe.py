"""
Watch what a model does inside, for one prompt.

Pull the attention weights and residual-stream activations out of GPT-2 in eager
mode, find the head that sinks hardest onto the first token, and measure the
activation spike. CPU, fp32, no GPU needed. This is the readable companion to
the FN2 field note; the plotting split (CPU venv extracts, matplotlib venv
draws) lives in extract2.py + plot.py.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    attn_implementation="eager",   # don't fuse attention; keep the matrix
    torch_dtype=torch.float32,
).eval()

prompt = "The cat sat on the mat."
enc = tok(prompt, return_tensors="pt")
tokens = [tok.decode([t]) for t in enc.input_ids[0].tolist()]

with torch.no_grad():
    out = model(
        **enc,
        output_attentions=True,       # keep the attention weights
        output_hidden_states=True,    # keep every layer's residual stream
        use_cache=True,               # and the KV cache
    )

# attentions: one tensor per layer -> [layers, heads, query, key]
attn = np.stack([a[0].numpy() for a in out.attentions])
# hidden states: one per layer + embedding -> [layers+1, tokens, dim]
hidden = np.stack([h[0].numpy() for h in out.hidden_states])

# Which head puts the most of the LAST token's attention on token 0?
sink = attn[:, :, -1, 0]                          # [layers, heads]
layer, head = np.unravel_index(sink.argmax(), sink.shape)
print(f"strongest sink: layer {layer}, head {head} -> "
      f"{sink[layer, head]:.0%} on {tokens[0]!r}")

# How common is it? deep-half heads with >half their attention on token 0.
deep = sink[sink.shape[0] // 2:]
print(f"deep-half heads sinking >50%: {(deep > 0.5).mean():.0%}")

# The activation spike: per-token residual norm, relative to the median.
norms = np.linalg.norm(hidden, axis=-1)           # [layers+1, tokens]
ratio = norms.max(axis=1) / np.median(norms, axis=1)
peak = int(ratio.argmax())
big = int(norms[peak].argmax())
print(f"activation spike: {ratio[peak]:.0f}x the median at layer {peak}, "
      f"on {tokens[big]!r}")
