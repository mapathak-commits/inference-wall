---
title: "FlashAttention: what it does, and why its payoff scales with prompt length"
permalink: /articles/part-8/
---

*Part 8 of "The Inference Wall". Same rig as the rest of the series: one NVIDIA A10G
(23 GB). Model is Qwen2.5-7B-Instruct, fp16, single stream. Draft.*

*Manas Pathak · draft, September 2026*

[The Inference Wall]({{ '/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }}) · **Part 8**

FlashAttention is the attention kernel underneath most modern serving stacks. It computes the
same attention every transformer does, but it does the arithmetic without ever writing the
large intermediate score matrix out to GPU memory, which is where a naive implementation spends
most of its time. This post measures what that buys, and the answer turns out to depend
entirely on one variable: how long the prompt is.

The study is a single-variable sweep. Same model, same GPU, same single-stream probe, and the
only thing that changes is prompt length, from 256 tokens up to 30,720. I run it twice: once
with FlashAttention, once with a competent general-purpose attention kernel that lacks
FlashAttention's memory tuning, so the difference between the two runs is attention and nothing
else. Then I open a profiler trace and confirm, kernel by kernel, that the wall-clock gap lives
exactly where the theory says it should.

The insight is that the benefit is not a fixed percentage. At 256 tokens the two kernels finish
within three milliseconds of each other; at 30,720 they are six seconds apart, on the identical
hardware. FlashAttention optimizes attention, attention cost grows with the square of the
prompt length, so the value of the optimization grows with prompt length too. It is negligible
at the short prompts most benchmarks use and large at the long contexts production actually
hits. The one fact you need going in comes from the
[primer]({{ '/articles/primer/' | relative_url }}): serving a token has two phases, a
**prefill** that reads the whole prompt in one dense pass and a **decode** that emits the answer
one token at a time. Attention behaves differently in each, and FlashAttention changes both.

## Why attention is the term that explodes

Start with the shape of the work, because the whole post follows from it. A transformer
layer does two large things per token: a stack of matrix multiplies (the projections and the
MLP) and an attention step. The matrix multiplies are the model's fixed cost. Their size
depends on the model's width, not on how long your prompt is, so processing a 30,000-token
prompt runs the same per-token matmul work as processing a 256-token one, just more times.
That cost grows **linearly** with prompt length.

Attention does not. In attention, every token looks at every token before it. At position
*n* there are *n* prior tokens to score against, so the total work of attending over a prompt
of length *N* grows with *N* positions each doing up to *N* comparisons: it scales as *N*
squared. Double the prompt and the matmul work doubles, but the attention work quadruples.
This is the single most important fact about long-context inference, and it means attention
is a term that starts negligible and, past some crossover, takes over.

You can watch the crossover happen. Take FlashAttention's own prefill time on this rig,
fit to a curve, with the quadratic piece pulled out as a share of the total:

| Prompt length | Prefill time | Share that is the quadratic attention term |
|---|---|---|
| 2,048 | 0.50 s | 3.4% |
| 8,192 | 2.07 s | 12.2% |
| 32,768 | (extrapolated) | 35.8% |

At 2,000 tokens the quadratic term is 3%, lost in the noise of the linear matmul cost. This
is the region where a casual benchmark lives, and it is exactly why attention optimizations
look pointless there: there is almost no attention cost to optimize. By 32,000 tokens the
quadratic term is more than a third of the entire prefill and climbing. Optimizing attention
only matters once attention is a large enough slice of the bill to be worth optimizing, and
whether it is depends entirely on how long your prompts are.

## What FlashAttention actually does

The name suggests speed; the mechanism is about memory, and it is worth being concrete
rather than waving at the word.

The naive way to compute attention is to build the full table of scores: for a prompt of
length *N*, an *N* by *N* grid where entry (i, j) is how much token *i* attends to token *j*.
At 30,000 tokens that grid is 900 million numbers **per attention head**, and there are many
heads per layer and many layers. Writing that grid out to the GPU's main memory (its HBM) and
reading it back to apply the softmax is a staggering amount of memory traffic, and on modern
GPUs memory traffic, not arithmetic, is the binding constraint. The grid is also why a naive
implementation can simply run out of memory on a long prompt: there is nowhere to put it.

FlashAttention's insight, from [Dao et al. 2022](https://arxiv.org/abs/2205.14135), is that
you never need the whole grid at once. It walks the prompt in tiles that fit in the GPU's
tiny on-chip scratchpad (its SRAM, far smaller and far faster than HBM), computes each tile's
scores there, folds them into a running softmax, and discards them before moving to the next
tile. The full *N* by *N* grid is never written to main memory at all. The arithmetic is the
same; the memory traffic collapses from quadratic to linear. That is the entire trick, and it
is why the kernel is "IO-aware": it is optimized for the memory system, not the math.

Decode gets a second, related trick. When the model emits token number 30,001, it must attend
back over all 30,000 cached tokens, but it is producing only one new token, so there is very
little arithmetic to hide the cost of reading that much cached state. The decode-side form of
FlashAttention, sometimes called FlashDecoding, splits that long backward read across many
parallel workers on the GPU and combines their partial results, so the one new token's
attention is computed by the whole chip at once instead of a single underused slice of it.
Hold onto this split-the-read idea; the trace at the end shows it is where the decode gap
comes from.

## The counterfactual: what "off" even means

Here is a wrinkle that shapes the whole experiment. On a modern serving stack there is no
"FlashAttention off" switch. The kernel is not a feature bolted onto a default; it *is* the
default. To measure what it buys, you have to serve the model with a **different** attention
backend and compare.

The comparison here is FlashAttention against **FlexAttention**, PyTorch's general-purpose,
compiled attention kernel. This matters for honesty about what the gap means. FlexAttention is
not a naive, materialize-the-whole-grid implementation; it is a real, tiled, compiled kernel.
So the numbers below are **not** flash-versus-nothing. They are flash versus a competent
general-purpose kernel that lacks FlashAttention's specific memory-traffic tuning and, more
importantly for decode, lacks its split-the-read trick. Read every gap in this post as "what
FlashAttention's specialization buys over a solid generic baseline," which is the honest
version of the question and a harder bar than beating a strawman. Everything else is held
fixed: same model, same GPU, same prompts, same single-stream probe, the attention backend
selected by an environment variable at server start and nothing else touched.

## Prefill: the quadratic gets steeper

The probe sends one prompt at a time, warm, and records time-to-first-token as prompt length
climbs from 256 to 30,720 tokens. Time-to-first-token is essentially prefill time: it is how
long the model takes to read the prompt before it can emit anything. Here are both backends:

| Prompt length | FlashAttention TTFT | FlexAttention TTFT |
|---|---|---|
| 256 | 0.116 s | 0.119 s |
| 1,024 | 0.280 s | 0.292 s |
| 2,048 | 0.502 s | 0.544 s |
| 4,096 | 0.992 s | 1.127 s |
| 8,192 | 2.067 s | 2.540 s |
| 16,384 | 4.591 s | 6.393 s |
| 24,576 | 7.620 s | 11.704 s |
| 30,720 | 10.181 s | 16.410 s |

Read it top to bottom and the whole thesis is in one column. At 256 tokens the two are three
milliseconds apart, indistinguishable. The gap widens slowly, then not slowly: by 8k it is
half a second, by 16k it is nearly two seconds, by 30k it is **6.2 seconds**. The two kernels
are running the identical model through the identical matmuls; the only thing that differs is
how each computes attention, and attention is the term going quadratic.

Fitting each column to a curve makes the mechanism explicit. Both fits have nearly the same
linear coefficient, about 220 microseconds per token, which is the matmul cost both kernels
share. They differ in the quadratic coefficient, the part that is pure attention: **3,689
picoseconds per token-squared for FlashAttention against 10,099 for FlexAttention.** The slow
kernel's attention term is 2.7 times steeper. That single number is the difference between the
two, and because it multiplies *N* squared, it is invisible at small *N* and merciless at
large *N*. It is the entire story of the table.

![FlashAttention versus FlexAttention prefill time as prompt length grows from 256 to 30,720 tokens: the two curves are indistinguishable below about 2,000 tokens and diverge sharply after, with FlexAttention bending upward far more steeply as the quadratic attention term takes over]({{ '/assets/figures/fig8a-prefill-divergence.png' | relative_url }})

## Decode: the gap that a flat line hides

Prefill is the dramatic half, but decode is where the split-the-read trick earns its keep,
and the numbers are more lopsided. The probe also measures **inter-token latency**, the gap
between successive output tokens during generation, as a function of how much context the model
is decoding on top of. Same two backends:

| Context length | FlashAttention decode ITL | FlexAttention decode ITL |
|---|---|---|
| 257 | 33.8 ms | 35.4 ms |
| 2,050 | 33.9 ms | 38.7 ms |
| 8,193 | 35.0 ms | 49.9 ms |
| 16,385 | 35.8 ms | 65.1 ms |
| 24,577 | 36.8 ms | 80.7 ms |
| 30,720 | 37.3 ms | 91.8 ms |

FlashAttention's decode is **essentially flat**: 33.8 ms at short context, 37.3 ms at 30k, an
11% rise across a 120-fold increase in context. From the outside the model barely notices how
much history it is carrying. FlexAttention starts at the same place and then climbs steadily
to 91.8 ms, **159%** slower at depth, nearly tripling its per-token latency purely because the
context got longer.

This is the split-the-read trick, felt from the client side. Each decode step reads the entire
cached context to attend over it. FlashAttention's decode kernel spreads that read across the
whole GPU, so a longer read is still done in about the same wall-clock time; the line stays
flat. FlexAttention reads that context in a way that does not parallelize the same way, so
every additional thousand tokens of history adds directly to every single token's latency.
For an interactive assistant deep in a long conversation, that is the difference between a
steady stream and a visible slowdown that gets worse the longer you talk to it.

Note which column would have lied to you. If you benchmarked decode at 2,000 tokens of
context, the number every quick test uses, you would see 33.9 against 38.7 ms and shrug. The
entire divergence lives past the context length almost nobody puts on the test bench.

## The trace: the six seconds have a name

The client-side numbers *say* attention is the difference. A profiler trace *shows* it, and
turns the claim into something you can count. I re-ran both backends with vLLM's torch profiler armed and
captured a bounded window at three prefill lengths and two decode depths, then bucketed every
GPU kernel in each window into three families: **attention**, **matmul** (the projections and
MLP), and everything else. The traces are downloadable and openable in a browser with no GPU;
the reproduce section says how.

The first thing the trace establishes is the cleanest control in the whole series, the matmul
time in the prefill windows for both backends:

| Prompt length | FlashAttention matmul time | FlexAttention matmul time |
|---|---|---|
| 2,048 | 439 ms | 438 ms |
| 8,192 | 1,664 ms | 1,663 ms |
| 28,672 | 5,756 ms | 5,755 ms |

The matmul work is **byte-for-byte identical** between the two backends, to within a
millisecond, because it is the same model doing the same projections and the same MLP. Every
difference between FlashAttention and FlexAttention is therefore in the attention family and
nowhere else. The trace has isolated the one variable perfectly: whatever the wall-clock gap
is, the trace can point to the exact kernel it lives in.

And it does. Here is the attention time in those same windows:

| Prompt length | FlashAttention attention | FlexAttention attention |
|---|---|---|
| 2,048 | 22 ms (4.6% of the window) | 51 ms (9.8%) |
| 8,192 | 262 ms (12.9%) | 693 ms (28.0%) |
| 28,672 | 3,098 ms (33.6%) | 8,389 ms (57.7%) |

At 28k prefill, FlashAttention computes attention in 3.1 seconds; FlexAttention needs 8.4
seconds for the identical math, a **2.7-fold** kernel gap. That 5.3-second attention gap is,
within a rounding error, the entire gap between the two window totals, 9.2 seconds against
14.5: the matmuls matched, so the whole difference landed in attention and nowhere else. And
notice the share column. On FlexAttention, attention has grown to 58% of all GPU time, the term
that started as a rounding error now the single largest thing the GPU does. That is the
quadratic, seen directly.

The decode trace closes the loop by naming the kernels. In FlashAttention's decode windows the
attention work is a kernel called `flash_fwd_splitkv_kernel`: literally "split the KV read,"
the split-the-read trick from earlier, appearing by name. It is the same tiled, parallel-read
kernel FlashAttention uses in prefill, so decode attention stays around 31% of the window even
at 28k context and the per-token latency stays flat. FlexAttention runs a single monolithic
kernel, `triton_tem_fused_0`, with no such split. At 28k decode context that one kernel swells
to **58% of the entire window**, and the window itself is 17.0 seconds against
FlashAttention's 10.3. The +11%-versus-+159% decode gap from the client-side table is, at the
kernel level, exactly this: one kernel that parallelizes the long read against one that does
not.

![Kernel-family breakdown of GPU time at 2k, 8k, and 28k prefill for both backends: the matmul family is identical between the two, while the attention family grows far faster for FlexAttention, reaching 58% of GPU time at 28k against FlashAttention's 34%]({{ '/assets/figures/fig8b-kernel-family-split.png' | relative_url }})

## What to take away

1. **The benefit scales with prompt length, it is not a fixed percentage.** FlashAttention
   was three milliseconds ahead at 256 tokens and six seconds ahead at 30,000, on the identical
   model and GPU. Because it optimizes attention, and attention cost grows with prompt length,
   the value of the optimization tracks how long your prompts are. Before you decide whether an
   attention optimization matters for you, ask what context length you actually serve.

2. **The reason is the quadratic.** Matmul work grows linearly with prompt length; attention
   grows with the square. So attention starts as a negligible slice of the bill and, past a
   crossover this rig puts around a few thousand tokens, becomes the largest slice. Everything
   FlashAttention does is aimed at the term that eventually takes over.

3. **What it actually does is cut memory traffic, not arithmetic.** It computes attention in
   on-chip tiles and never writes the full score grid to main memory, turning quadratic memory
   traffic into linear. For decode it additionally splits the long backward read over the whole
   GPU, which is why its per-token latency stays flat as context grows while a generic kernel's
   climbs 159%.

4. **The trace isolates the variable perfectly.** Because the matmul time was byte-for-byte
   identical between the two backends, every second of difference provably lived in the
   attention family, and the decode gap resolved to a single named kernel: `flash_fwd_splitkv`,
   the split-the-read trick, against a monolithic kernel that does not split. When two backends
   share everything but one kernel, the trace turns "attention is the difference" from a claim
   into a measurement.

5. **Beating a strawman would have been easy; this was not one.** The baseline here,
   FlexAttention, is itself a real tiled compiled kernel, not a naive implementation. The gaps
   in this post are what FlashAttention's specialization buys over a competent generalist, which
   is the number worth knowing.

Next in the series: this post kept everything on the A10G. I take the model off the rig
entirely and decode it on a CPU, where a subtlety about how two byte-identical weight files are
laid out makes one of them decode several times faster than the other on a laptop and at
exactly the same speed on a server.

---

*Reproduce: the prefill/decode probe (`s1_probe.py`), the trace-capture driver
(`s3_capture.py`), the kernel-family census (`tp_census.py`), and the server launch script are
in `experiments/08-flash-attention/`; the raw probe logs, the fitted coefficients, and the
captured profiler traces (gzipped chrome-trace JSON, openable in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev) with no GPU) are in `benchmarks/08-flash-attention/`.
Single A10G, Qwen2.5-7B-Instruct, fp16, single stream; your absolute numbers will differ, the
shape will not. The backend is selected at server start with the `VLLM_ATTENTION_BACKEND`
environment variable (`FLASH_ATTN` against `FLEX_ATTENTION`); everything else is held fixed.*

---

**Previous:** [Part 7: Speculative decoding]({{ '/articles/part-7/' | relative_url }}) · **Next:** [Part 9: Off the rig]({{ '/articles/part-9/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }})

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
