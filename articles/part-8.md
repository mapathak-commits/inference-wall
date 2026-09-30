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
hardware. FlashAttention optimizes attention, attention cost grows with the square of prompt
length, so the value of the optimization grows with it too: negligible at the short prompts most
benchmarks use, large at the long contexts production hits. The one fact you need going in comes
from the [primer]({{ '/articles/primer/' | relative_url }}): serving a token has two phases, a
**prefill** that reads the whole prompt in one dense pass and a **decode** that emits the answer
one token at a time. Attention behaves differently in each, and FlashAttention changes both.

## Why attention is the term that explodes

A transformer layer does two large things per token: a stack of matrix multiplies (the
projections and the MLP) and an attention step. The matmuls are the model's fixed cost; their
size depends on the model's width, not the prompt, so they grow **linearly** with prompt
length. Attention does not. Every token attends to every token before it, so over a prompt of
length *N* the work is *N* positions each scoring up to *N* others: it scales as *N* squared.
Double the prompt and the matmuls double, but the attention quadruples. That is why attention
starts negligible and, past some crossover, takes over.

You can watch the crossover in FlashAttention's own prefill time, fit to a curve with the
quadratic piece pulled out as a share of the total:

| Prompt length | Prefill time | Share that is the quadratic attention term |
|---|---|---|
| 2,048 | 0.50 s | 3.4% |
| 8,192 | 2.07 s | 12.2% |
| 32,768 | (extrapolated) | 35.8% |

At 2,000 tokens the quadratic term is 3%, lost in the linear matmul cost, which is exactly the
region a casual benchmark lives in and exactly why attention optimizations look pointless
there. By 32,000 tokens it is more than a third of the prefill and climbing. Optimizing
attention only matters once attention is a large enough slice of the bill to matter, and that
depends entirely on prompt length.

## What FlashAttention actually does

The name suggests speed; the mechanism is about memory. The naive way to compute attention is
to build the full table of scores: for a prompt of length *N*, an *N* by *N* grid where entry
(i, j) is how much token *i* attends to token *j*. At 30,000 tokens that grid is 900 million
numbers **per attention head**, across many heads and many layers. Writing it out to the GPU's
main memory (its HBM) and reading it back for the softmax is a staggering amount of memory
traffic, and on modern GPUs memory traffic, not arithmetic, is the binding constraint.

FlashAttention's insight, from [Dao et al. 2022](https://arxiv.org/abs/2205.14135), is that you
never need the whole grid at once. It walks the prompt in tiles that fit in the GPU's tiny
on-chip scratchpad (its SRAM, far faster than HBM), computes each tile's scores there, folds
them into a running softmax, and discards them before the next tile. The *N* by *N* grid is
never written to main memory. The arithmetic is the same; the memory traffic collapses from
quadratic to linear. That is why the kernel is "IO-aware": it is optimized for the memory
system, not the math.

Decode gets a second, related trick. To emit token 30,001 the model attends back over all
30,000 cached tokens, but produces only one new token, so there is little arithmetic to hide
the cost of that long read. The decode-side form, sometimes called FlashDecoding, splits the
read across many parallel workers on the GPU and combines their partial results, so the read is
done by the whole chip at once. Hold onto this split-the-read idea; the trace shows it is where
the decode gap comes from.

One note on the comparison, because it shapes what the gaps below mean. There is no
"FlashAttention off" switch; it *is* the default, so measuring it means swapping in a different
backend. The counterfactual here is **FlexAttention**, PyTorch's general-purpose compiled
kernel. It is not a naive materialize-the-grid implementation, it is a real tiled compiled
kernel that simply lacks FlashAttention's memory tuning and its split-the-read decode. So the
numbers are what FlashAttention's specialization buys over a competent generalist, a harder bar
than beating a strawman. Everything else is held fixed and the backend is selected by one
environment variable at server start.

## Prefill: the quadratic gets steeper

The probe sends one prompt at a time, warm, and records time-to-first-token, which is
essentially prefill time, as prompt length climbs from 256 to 30,720 tokens:

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

The whole thesis is in one column. At 256 tokens the two are three milliseconds apart,
indistinguishable; by 8k the gap is half a second, by 16k nearly two seconds, by 30k it is
**6.2 seconds**. Fit each column to a curve and the mechanism is explicit: both fits share the
same linear coefficient, about 220 microseconds per token, the matmul cost common to both, and
differ only in the quadratic coefficient, the pure-attention part: **3,689 picoseconds per
token-squared for FlashAttention against 10,099 for FlexAttention**, 2.7 times steeper. Because
that coefficient multiplies *N* squared, it is invisible at small *N* and merciless at large.

![FlashAttention versus FlexAttention prefill time as prompt length grows from 256 to 30,720 tokens: the two curves are indistinguishable below about 2,000 tokens and diverge sharply after, with FlexAttention bending upward far more steeply as the quadratic attention term takes over]({{ '/assets/figures/fig8a-prefill-divergence.png' | relative_url }})

## Decode: the gap that a flat line hides

Decode is where the split-the-read trick earns its keep, and the numbers are more lopsided. The
probe also measures **inter-token latency**, the gap between successive output tokens, as a
function of how much context the model is decoding on top of:

| Context length | FlashAttention decode ITL | FlexAttention decode ITL |
|---|---|---|
| 257 | 33.8 ms | 35.4 ms |
| 2,050 | 33.9 ms | 38.7 ms |
| 8,193 | 35.0 ms | 49.9 ms |
| 16,385 | 35.8 ms | 65.1 ms |
| 24,577 | 36.8 ms | 80.7 ms |
| 30,720 | 37.3 ms | 91.8 ms |

FlashAttention's decode is **essentially flat**: 33.8 ms at short context, 37.3 ms at 30k, an
11% rise across a 120-fold increase in context. FlexAttention starts at the same place and
climbs to 91.8 ms, **159%** slower at depth, nearly tripling per-token latency purely because
the context got longer. This is the split-the-read trick from the client side: FlashAttention's
decode kernel spreads the long cached-context read across the whole GPU, so it costs about the
same wall-clock time no matter how long it is, while FlexAttention's read does not parallelize
that way, so every extra thousand tokens of history adds directly to every token's latency. And
note that a decode benchmark at 2,000 tokens, the length most quick tests use, would show 33.9
against 38.7 ms and hide the entire divergence.

## The trace: the six seconds have a name

The client-side numbers *say* attention is the difference; a profiler trace *shows* it. I
re-ran both backends with vLLM's torch profiler armed, captured a bounded window at three
prefill lengths and two decode depths, and bucketed every GPU kernel into three families:
**attention**, **matmul** (projections and MLP), and everything else. The traces are openable
in a browser with no GPU; the reproduce section says how.

The first thing the trace establishes is the cleanest control in the whole series, the matmul
time in the prefill windows for both backends:

| Prompt length | FlashAttention matmul time | FlexAttention matmul time |
|---|---|---|
| 2,048 | 439 ms | 438 ms |
| 8,192 | 1,664 ms | 1,663 ms |
| 28,672 | 5,756 ms | 5,755 ms |

The matmul work is **byte-for-byte identical** between the two backends, to within a
millisecond, because it is the same model doing the same projections and MLP. Every difference
between the two is therefore in the attention family and nowhere else. Here is the attention
time in those same windows:

| Prompt length | FlashAttention attention | FlexAttention attention |
|---|---|---|
| 2,048 | 22 ms (4.6% of the window) | 51 ms (9.8%) |
| 8,192 | 262 ms (12.9%) | 693 ms (28.0%) |
| 28,672 | 3,098 ms (33.6%) | 8,389 ms (57.7%) |

At 28k prefill, FlashAttention computes attention in 3.1 seconds; FlexAttention needs 8.4 for
the identical math, a **2.7-fold** gap. Since the matmuls matched, that 5.3-second attention
gap is, within a rounding error, the entire gap between the two window totals (9.2 seconds
against 14.5). On FlexAttention attention has grown to 58% of all GPU time, the single largest
thing the GPU does: the quadratic, seen directly.

The decode trace names the kernels. FlashAttention's is `flash_fwd_splitkv_kernel`, literally
"split the KV read," the split-the-read trick appearing by name. It is the same tiled kernel it
uses in prefill, so decode attention stays around 31% of the window at 28k and per-token
latency stays flat. FlexAttention runs a single monolithic `triton_tem_fused_0` with no split;
at 28k it swells to **58% of the window**, and the window is 17.0 seconds against
FlashAttention's 10.3. That is the +11%-versus-+159% decode gap at the kernel level: one kernel
that parallelizes the long read against one that does not.

![Kernel-family breakdown of GPU time at 2k, 8k, and 28k prefill for both backends: the matmul family is identical between the two, while the attention family grows far faster for FlexAttention, reaching 58% of GPU time at 28k against FlashAttention's 34%]({{ '/assets/figures/fig8b-kernel-family-split.png' | relative_url }})

## What to take away

1. **The benefit scales with prompt length, it is not a fixed percentage.** FlashAttention was
   three milliseconds ahead at 256 tokens and six seconds ahead at 30,000, on the identical
   model and GPU. It optimizes attention, attention cost grows with the square of prompt length,
   so the value of the optimization tracks the context length you actually serve. Decide whether
   it matters for you by that, not by a short-prompt benchmark.

2. **What it does is cut memory traffic, not arithmetic.** It computes attention in on-chip
   tiles and never writes the full score grid to main memory, turning quadratic memory traffic
   into linear, and for decode it splits the long cached-context read across the whole GPU,
   which is why its per-token latency stays flat as context grows while the generic kernel's
   climbs 159%.

3. **The trace turns the claim into a measurement.** Because matmul time was byte-for-byte
   identical between the two backends, every second of difference provably lived in attention,
   and the decode gap resolved to a single named kernel, `flash_fwd_splitkv`, against a
   monolithic kernel that does not split the read. The baseline, FlexAttention, is a real
   compiled kernel and not a strawman, so these are the gains over a competent generalist.

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
