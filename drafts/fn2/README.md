# Field note: I opened up one prompt to see what the model was thinking. It was thinking about the word "The."

*Field notes are the short lane of The Inference Wall: one prompt, one measurement, one thing I didn't expect. No production stack, no benchmark suite. Just a look inside.*

---

Most of this series watches models from the outside. How many tokens per second, how big a batch, how the work looks in a profiler trace. That's the serving layer, the plumbing that turns your prompt into an answer fast, and it's where the money is.

But there's a layer underneath that the plumbing never shows you: the actual thinking. When a model reads your prompt, it does a huge pile of arithmetic. Which earlier words does it look at? How strongly? What is it holding onto as it goes? Those numbers exist for a fraction of a second and then they're gone.

So I asked a simple question. For one prompt, can I just watch what the model is doing inside while it reads? It turns out you can, and the picture is stranger than I expected.

## A one-minute refresher on attention

If you've read [the primer on what happens inside an LLM](https://mapathak-commits.github.io/inference-wall/articles/primer-2/), skip this. If not, here's the only idea you need.

A language model reads your prompt one word at a time (really *tokens*, which are word-pieces, but think "words"). The single trick that lets it understand a sentence rather than a bag of words is **attention**: as the model processes each word, it looks back over the earlier words and decides how much weight to give each one. When it reads "sat" in *the cat sat on the keyboard*, attention is what lets "sat" look back and connect to "cat," the thing doing the sitting.

Each word spreads a fixed budget of attention over the words before it. The budget always adds up to 1, like slicing a pie: give a bigger slice to one word and every other slice shrinks. So an attention pattern is just a set of pie-slices, one per earlier word, saying where this word is looking.

The model doesn't do this once. It has many **heads**, each looking for its own kind of relationship (one might track the subject of the sentence, another the previous word), stacked in **layers** that refine the picture. GPT-2, the small open model I'll use here, has 12 layers with 12 heads each: 144 little attention patterns per word. That's the thing I wanted to see.

## Getting the numbers out

Here's the catch, and it's the whole reason this is a field note. The fast tools everyone actually serves models with, like vLLM or Ollama, *can't* show you this (I'll come back to why). You have to use a slower, more honest library, HuggingFace `transformers`, and ask it to hand back the scratch work it normally throws away.

If you don't care about the code, skip the gray boxes. It's three settings that mean "keep the attention pie-slices, keep the running state, and keep the memory of earlier words":

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", attn_implementation="eager", torch_dtype=torch.float32,
).eval()

out = model(**enc,
            output_attentions=True,      # keep the attention weights
            output_hidden_states=True,   # keep the running state at every layer
            use_cache=True)              # and the memory of earlier tokens (the KV cache)
```

That gives back two things worth staring at. The **attention weights**: for my eight-token sentence, a stack of 8x8 grids, one per head, where each row is a word and each cell says how big a slice it gave to an earlier word. And the **running state**: the vector the model carries for each word, snapshotted after every layer, which I'll get to in the second half.

The full runnable version is [`observe.py`](code/observe.py). Everything below just reads off those two.

## What one head is looking at

The clean way to read an attention grid: pick a row (that's one word), and the bright cells are the earlier words it leaned on. My prompt for the rest of this note is `"The cat sat on the keyboard again."`, chosen because some heads do something genuinely readable with it.

Take layer 4, head 3. It's a "who did what" head: several later words reach back and grab the subject of the sentence. "sat" looks at "cat" with a slice of 0.96, "on" looks at "cat" at 0.89, even the final period points back at "cat." You can watch the model tie the sentence together, exactly the intuition you'd hope for. Now score every head by a different number: how big a slice does the *last* word hand to the *first* word? One head wins outright. Layer 5, head 1 gives the first word a slice of 1.00, the whole pie. Here the two sit side by side:

![Two GPT-2 attention grids side by side. On the left, layer 4 head 3, several rows point back at the "cat" column with printed weights like 0.96 and 0.89. On the right, layer 5 head 1, one solid bright column on the first word, every cell reading 1.00.](fig_attention.png)

*Each row is a word doing the looking; each cell is the slice it gave an earlier word (numbers printed in, darker means smaller; the blank upper triangle is just the future, which no word is allowed to see). Left, layer 4, head 3: a readable head, where later words reach back to the subject, "cat." Right, layer 5, head 1: the surprise. Every word, whatever it means, hands its entire slice to the first word, "The."*

The left panel is what I assumed all attention looked like: words wiring up to each other, meaning getting assembled. The right panel is the surprise. A whole head, deep in the network, has decided the single most useful place to look is a throwaway article at the front of the sentence.

And it isn't one odd head. If I score all 144 by that same first-word measure and lay them out as a grid, the back half of the network lights up almost entirely:

![A 12-by-12 grid of layer versus head, shaded by how much each head's last word looks at the first word. The top rows (early layers) are dark; the bottom rows (deep layers) are mostly bright.](fig_sink_grid.png)

*Each square is one head, shaded by how much of the last word's attention it dumps on the first word. Early layers (top) still do real local work like the "cat" head above. In the deep layers (bottom), most heads have gone bright. The boxed square is layer 5, head 1. Across the back half of the network, 92% of heads send more than half their attention to the first word.*

## Why it does that

This is a known effect, called an **attention sink**, and once you see the reason it stops being mysterious.

Remember the pie has to add up to 1. A head is forced to spend its whole budget on the earlier words, whether or not any of them are relevant to its job. But heads are specialists. A head that hunts for, say, the verb three words back has nothing to do in a sentence where that pattern doesn't appear. It still has to put its pie somewhere.

So it dumps the budget on a word that is always there, always in the same spot, and carries no meaning worth disturbing: the first one. The sink is the model's junk drawer, a safe place to offload attention it doesn't want to spend. The first word gets the job because every later word can see it, and a fixed target is easy for the model to learn. The [StreamingLLM paper](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023) named this effect and showed the model quietly depends on it, which matters in a minute.

## The second surprise: one word's magnitude explodes

While I had the internals open, I looked at the other thing the model hands back: the running state it carries for each word. Each word's state is a vector, and I can summarize it with a single number: its **magnitude**, how far the vector reaches from zero (the square root of its squared components). Every word's vector has the same number of components, 768 of them in GPT-2, so this isn't about one word having a longer vector than another. It's about how big the numbers inside are. Track that magnitude layer by layer and almost every word grows gently and stays in a tight pack. Except one.

![Per-word state magnitude across the layers, on a log scale. One line, the first word, shoots far above the pack in the middle layers, rides high, and drops back at the end.](fig_hidden_norm.png)

*The magnitude of each word's internal state, layer by layer (log scale, so each gridline is 10x). Every word grows gently except "The," which spikes to about 39 times larger than the rest through the middle of the network, then settles back into the pack right at the end. On a normal scale the spike would flatten every other line to the floor.*

One word blows up far past the others, the same word again, peaks in the middle of the network, and quietly returns to the pack by the final layer. If you only looked at the model's output, which is all you normally get, you'd never know it happened. You have to watch the middle of the computation to catch it.

These spikes are called **massive activations** ([Sun et al., 2024](https://arxiv.org/abs/2402.17762)), and they're the flip side of the sink. The model parks a big, roughly constant scratch value on one word and then points its spare attention there. The junk drawer and the scratch pad are the same word.

(I ran the same check across a handful of other models, from Meta's OPT to Alibaba's Qwen, and both effects showed up every time. But the point here is the intuition and how to look, not a survey, so one clean example carries it.)

## Why the fast tools can't show you this

Here's the tie back to the rest of the series. I found all of this without touching vLLM or Ollama, the tools I use everywhere else, because they never build the picture I just showed you.

That 8x8 grid, one weight for every pair of words, is the expensive part of attention. For a real prompt of thousands of words it's a grid of millions of cells, and its size grows with the *square* of the length. The entire art of fast serving is to get the *result* of attention without ever writing that giant grid down. FlashAttention, the subject of the next full post, computes it in small tiles and never stores the full grid. PagedAttention, the trick vLLM is built on, streams the earlier words' memory through the chip as fast as it can and would never stop to hand you a labeled table. Speed comes precisely from throwing away the scratch work I wanted to read.

So the numbers I plotted exist for a few microseconds inside a fused chip operation and then they're gone. The serving layer stays perfectly observable, and watching it is most of what this series does. But this deeper math layer is deliberately optimized out of existence in the fast path. To see it, you run the slow, honest version: one small model, full precision, on a CPU, with the flags that keep everything. It would never survive in production. It's also the only version that stops to write down what the model is thinking.

## Why it matters

Two throwaway observations about an eight-token sentence turn out to sit under two of the hardest problems in running these models cheaply.

**The sink is why you can't just forget the start of a long chat.** When a conversation runs past a model's window, the obvious fix is to drop the oldest words. StreamingLLM showed this wrecks the model's quality, and the sink is why: the deep layers are still pouring most of their attention onto those first few words. Delete them and every head's pie has to be re-sliced onto words that were only ever meant to be ignored, and the model falls apart. The fix that works is to *keep* a few opening words forever, however long the chat gets. The junk drawer turns out to be structural.

**The high-magnitude word is why shrinking models is hard.** The series finale is about running models in 4 bits instead of 16 or 32, which saves enormous memory but means squeezing every number into a tiny range of values. That squeeze hates outliers: one value 30 or 100 times bigger than its neighbors stretches the range until everything else rounds to mush. The massive-activation word is exactly that outlier, and it shows up on nearly every pass. A big slice of the research on shrinking models is, underneath, elaborate machinery for handling these specific spikes.

Both of these were discovered the hard way, at scale, by teams running models in production. And both are sitting right there in forty lines of code on a single toy sentence, if you're willing to run the slow version that writes down what the fast one erases.

I went looking to watch a model think. What I mostly found was housekeeping: a quiet place to dump attention it doesn't need, and a scratch value stuck on the nearest throwaway word. The fun part isn't that the model is doing something deep with the word "The." It's that this unglamorous bookkeeping matters enough that two of the nastiest problems in serving are, underneath, just fights with it.

---

*Method: GPT-2, HuggingFace `transformers` eager attention, full precision, CPU, prompt `"The cat sat on the keyboard again."` The sink score is the last token's attention weight on the first token, per head. The magnitude spike is the largest per-token state vector magnitude (L2 norm) relative to the median, across all layers. Code: [`observe.py`](code/observe.py).*
