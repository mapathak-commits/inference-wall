# I opened up one prompt to see what the model was thinking. It was thinking about the word "The."

*Part of The Inference Wall. A detour from the usual rig: instead of Qwen3.5-4B under load on an A10G, this one opens up a small model, GPT-2, on a CPU, keeping every intermediate value so the arithmetic is slow enough and small enough to read.*

---

Most of this series watches models from the outside. How many tokens per second, how big a batch, how the work looks in a profiler trace. That's the serving layer, the plumbing that turns your prompt into an answer fast, and it's where the money is.

But there's a layer underneath that the plumbing never shows you: the actual thinking. When a model reads your prompt, it does a huge pile of arithmetic. Which earlier tokens does it look at? How strongly? What is it holding onto as it goes? Those numbers exist for a fraction of a second and then they're gone.

I asked a simple question: for one prompt, can I just watch what the model is doing inside while it reads? It turns out you can, and the picture is stranger than I expected.

## What attention is doing

If you've read [the primer on what happens inside an LLM](https://mapathak-commits.github.io/inference-wall/articles/primer-2/), skip ahead. If not, take one sentence: *The cat sat on the keyboard again.* The model reads it one token at a time; for simplicity, let's say each word is a token. When it reaches "sat" it has a problem. On its own, "sat" means nothing; what sat is back at "cat." To resolve it, the model reaches back over the tokens it has already read and pulls "cat" toward "sat." That reaching back is **attention**, and it is the whole reason a model handles a sentence rather than an unordered pile of tokens.

It doesn't attend to just one earlier token. Each token spreads a fixed budget of weight across all the tokens before it, a probability distribution over the earlier tokens: non-negative weights that sum to 1. I'll call one token's distribution its pie, since attention divides it into slices, one per earlier token. When the model processes "sat," a well-behaved pie puts most of its mass on "cat" and a little on "the."

And the model does this many times over in parallel. Each pass is a **head**, and different heads look for different things: one might chase the subject of the verb, another just the token right before. Stack those heads into **layers** that refine the picture, and a small model already holds a lot of them. GPT-2, the one I'll use here, has 12 layers of 12 heads: 144 separate pies for every token. That was the thing I wanted to see.

## Getting the numbers out

The fast tools everyone serves models with, like vLLM or Ollama, *can't* show you this: their whole job is to reach the answer fast and skip the scratch work on the way. To see it you have to use a slower library, HuggingFace `transformers`, and ask it to hand back the numbers the fast tools throw away.

If you don't care about the code, skip the gray boxes. It comes down to three settings that mean keep the attention pie-slices, keep the running state, and keep the memory of earlier tokens:

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", attn_implementation="eager", torch_dtype=torch.float32,
).eval()

out = model(**enc,
            output_attentions=True,      # keep the attention weights
            output_hidden_states=True,   # keep the running state at every layer
            use_cache=True)              # and the memory of earlier tokens (the KV cache)
```

That gives back two things worth staring at. The **attention weights**: for my eight-token sentence, a stack of 8x8 grids, one per head, where each row is a token and each cell says how big a slice it gave to an earlier token. And the **running state**: the vector the model carries for each token, snapshotted after every layer. The full runnable version is [`observe.py`](code/observe.py).

## What one head is looking at

The clean way to read an attention grid: pick a row, which is one token, and the bright cells are the earlier tokens it leaned on. I picked `"The cat sat on the keyboard again."` because some heads do something genuinely readable with it.

Take layer 4, head 3. It's a subject-tracking head: several later tokens reach back and grab the subject of the sentence. "sat" looks at "cat" with a slice of 0.96, "on" looks at "cat" at 0.89, even the final period points back at "cat." You can watch the model tie the sentence together, exactly the intuition you'd hope for. Now score every head by a different number: how big a slice does the *last* token hand to the *first* token? One head wins outright. Layer 5, head 1 gives the first token a slice of 1.00, the whole pie. Here the two sit side by side:

![Two GPT-2 attention grids side by side. On the left, layer 4 head 3, several rows point back at the "cat" column with printed weights like 0.96 and 0.89. On the right, layer 5 head 1, one solid bright column on the first token, every cell reading 1.00.](fig_attention.png)

*Each row is a token doing the looking; each cell is the slice it gave an earlier token. Numbers are printed in, darker means smaller, and the blank upper triangle is just the future, which no token is allowed to see. Left, layer 4, head 3: a readable head, where later tokens reach back to the subject, "cat." Right, layer 5, head 1: the surprise. Every token, whatever it means, hands its entire slice to the first token, "The."*

The left panel is what I assumed all attention looked like: tokens wiring up to each other, meaning getting assembled. The right panel is the surprise. A whole head, deep in the network, has decided the single most useful place to look is a throwaway article at the front of the sentence.

And it isn't one odd head. If I score all 144 by that same first-token measure and lay them out as a grid, the back half of the network lights up almost entirely:

![A 12-by-12 grid of layer versus head, shaded by how much each head's last token looks at the first token. The top rows (early layers) are dark; the bottom rows (deep layers) are mostly bright.](fig_sink_grid.png)

*Each square is one head, shaded by how much of the last token's attention it dumps on the first token. The early layers up top still do real local work like the subject-tracking head above. Deeper in the network, most heads have gone bright. The boxed square is layer 5, head 1. Across the back half of the network, 92% of heads send more than half their attention to the first token.*

## Why it does that

This is a known effect, called an **attention sink**, and once you see the reason it stops being mysterious.

Remember the pie has to add up to 1. A head is forced to spend its whole budget on the earlier tokens, whether or not any of them are relevant to its job. But heads are specialists. A head that hunts for, say, the verb three tokens back has nothing to do in a sentence where that pattern doesn't appear. It still has to put its pie somewhere.

It dumps the budget instead on a token that is always there, always in the same spot, and carries no meaning worth disturbing: the first one. The sink is the model's junk drawer, a safe place to offload attention it doesn't want to spend. The first token gets the job because every later token can see it, and a fixed target is easy for the model to learn. The [StreamingLLM paper](https://arxiv.org/abs/2309.17453) by Xiao et al. in 2023 named this effect and showed that the model depends on it.

## The second surprise: one token's magnitude explodes

While I had the internals open, I looked at the other thing the model hands back: the running state it carries for each token. Each token's state is a vector, and I can summarize it with a single number: its **magnitude**, the plain length of that vector, its L2 norm: the square root of the sum of its squared components. Every token's vector has the same number of components, 768 of them in GPT-2, so this isn't about one token having a longer vector than another. It's about how big the numbers inside are. For each token I take the largest magnitude it reaches at any layer. Seven of the eight land within a narrow band. One does not.

![A horizontal bar chart of eight tokens, each bar the token's largest state magnitude across all layers. Seven bars are short and about equal; the bar for the first token, "The," is more than ten times longer than any other.](fig_hidden_norm.png)

*Each bar is one token's largest internal-state magnitude across all layers. Seven tokens sit in a tight band near a magnitude of 250. "The" reaches about 3,100, roughly 12 times the median token. That single value is large enough to dominate everything the network carries at that position.*

The outlier is the first token again. Its magnitude also climbs highest in the middle layers and eases back toward the pack by the final one, so if you only read the model's output, which is all you normally get, you'd never see it. You have to look inside the computation to catch it.

These spikes are called **massive activations**, named by [Sun et al. in 2024](https://arxiv.org/abs/2402.17762), and they're the flip side of the sink. The model parks a big, roughly constant scratch value on one token and then points its spare attention there. The junk drawer and the scratch pad are the same token.

I ran the same check across a handful of other models, including Meta's OPT and Alibaba's Qwen, and both effects showed up every time. The point here is the intuition and how to look, not a survey, so one clean example carries it.

## Why the fast tools can't show you this

I found all of this without touching vLLM or Ollama, the tools I use everywhere else, because they never build the picture I just showed you.

That 8x8 grid, one weight for every pair of tokens, is the expensive part of attention. For a real prompt of thousands of tokens it's a grid of millions of cells, and its size grows with the *square* of the length. The entire art of fast serving is to get the *result* of attention without ever writing that giant grid down. FlashAttention, the subject of a coming post, computes it in small tiles and never stores the full grid. PagedAttention, the trick vLLM is built on, streams the earlier tokens' memory through the chip as fast as it can and would never stop to hand you a labeled table. Speed comes precisely from throwing away the scratch work I wanted to read.

The numbers I plotted exist for only a few microseconds inside a fused chip operation and then they're gone. The serving layer stays perfectly observable, and watching it is most of what this series does. But this deeper math layer is deliberately optimized out of existence in the fast path. To see it, you run the slow version: one small model, full precision, on a CPU, with the flags that keep everything. It would never survive in production. It's also the only version that stops to write down what the model is thinking.

## Why it matters

Two throwaway observations about an eight-token sentence turn out to sit under two of the hardest problems in running these models cheaply.

**The sink is why you can't just forget the start of a long chat.** When a conversation runs past a model's window, the obvious fix is to drop the oldest tokens. StreamingLLM showed this wrecks the model's quality, and the sink is why: the deep layers are still pouring most of their attention onto those first few tokens. Delete them and every head's pie has to be re-sliced onto tokens that were only ever meant to be ignored, and the model falls apart. The fix is to always keep the first few tokens in the window, no matter how long the conversation grows, so the sink never disappears from under the deep layers.

**The high-magnitude token is why shrinking models is hard.** Part 5 of this series, still to come, runs models in 4 bits instead of 16, which saves enormous memory but means squeezing every number into a tiny range of values. That squeeze hates outliers: one value 30 or 100 times bigger than its neighbors stretches the range until everything else rounds to mush. The massive-activation token is exactly that outlier, and it shows up on nearly every pass. A big slice of the research on shrinking models is, underneath, elaborate machinery for handling these specific spikes.

Both of these were discovered the hard way, at scale, by teams running models in production. And both are sitting right there in forty lines of code on a single toy sentence, if you're willing to run the slow version that writes down what the fast one erases.

The two effects are one phenomenon seen from two sides. A large, roughly constant value sits on the first token, and the network's spare attention drains onto it. Neither has anything to do with the word "The" in particular; the first token is simply a convenient, always-present place to park what the model doesn't need. That bookkeeping is not a curiosity. It sets a hard limit on two of the nastiest problems in serving these models: what you can evict from a long context, and how far you can compress the weights.

---

*Method: GPT-2, HuggingFace `transformers` eager attention, full precision, CPU, prompt `"The cat sat on the keyboard again."` The sink score is the last token's attention weight on the first token, per head. The magnitude spike is the largest per-token state vector magnitude, its L2 norm, relative to the median, across all layers. Code: [`observe.py`](code/observe.py).*
