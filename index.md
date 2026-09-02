---
layout: home
title: The Inference Wall
---

*Understanding ML inference by breaking it apart.*

---

## What this series is

Suppose you want to run an open-source AI model yourself instead of paying to call one over
an API. You rent a GPU, load the model onto it, start a server, and point your app at it.
That last mile, actually serving the model's answers to real users, is called *inference*,
and it is where a surprising amount of money and latency quietly goes. It turns out to be
much harder to do well than "the model fits, so ship it," and this series is about why.

Most writing that tries to explain inference is one of three kinds. One is a code walkthrough
of how a serving engine works inside ([Gordić's "Inside vLLM"](https://www.aleksagordic.com/blog/vllm)
is the definitive one), great if you want to read the source, heavy going if you just want
intuition. Another is a profiling how-to
([Red Hat's](https://developers.redhat.com/articles/2025/10/16/profiling-vllm-inference-server-gpu-acceleration-rhel)
and [Spheron's](https://www.spheron.network/blog/gpu-profiling-ai-workloads-nsight-compute-pytorch-profiler-guide/)
guides): how to point Nsight or the torch profiler at a server, but not what the numbers turn
out to be. The third is theory: math about how fast a chip *could* go in principle (you will
meet "roofline" and "arithmetic intensity" if you go looking). All are useful and all are well covered. This series lives in the gap between them,
the empirical middle: **I take one real model on one ordinary GPU, turn a single knob until
something breaks, report the before-and-after number, and open a profiler trace, a recording
of exactly what the GPU did, to explain why.** A break, a number, and a trace, every time.

The point is not to benchmark hardware. It is to build the felt sense that lets you look
at a serving system and know *which resource is about to run out and what to do about it*,
the intuition that is hard to get from either code or theory alone. So this is written for
the engineer who can already rent a GPU box and get a model serving, but has never opened a
profiler and has no gut feel for where the latency actually goes. You do not need to know
CUDA, or how an attention kernel works inside; you need to be curious about why the server
is slower than you expected.

Underneath all five posts is one claim, worth stating up front so you can watch it recur:
**on this hardware, inference is a bytes-through-memory problem more than a
math problem.** The GPU spends its time moving the model's weights through memory to emit
each token, so nearly every limit you hit, and every win you get, comes down to how many
bytes move per token and how many requests share each move. Hold that frame; each post is a
different face of it.

## The rig, stated once

Everything in the series runs on the same setup, so the results compose into one story:

- **Model:** `Qwen/Qwen3.5-4B`, fp16, served text-only. A real hybrid-attention model,
  not a toy. The finale swaps in `Qwen3.5-9B-AWQ` to make a point about quantization.
- **GPU:** a single NVIDIA A10G, 23 GB. Mid-range, the kind of GPU a small team actually
  has, not an H100 cluster.
- **Server:** `vllm serve` driven by `vllm bench serve` under a request-rate sweep, so
  every latency number is a real percentile under load, not a single-stream average.

The A10G is a deliberate choice, not a budget compromise. It has modest memory (23 GB) and
modest bandwidth, so both walls this series is about, running out of memory *capacity* and
running out of memory *bandwidth* (how fast the weights can be streamed), sit close enough to
touch. That is exactly what makes the limits *visible*: you can hit them with one small model
and read what happens. On a data-center GPU like an H100
(80 GB) or a B200 (192 GB), with several times the memory and an order of magnitude more
compute and bandwidth, most of the specific breakages here simply do not occur at this
scale, the model fits many times over and the GPU is nowhere near saturated. That does not
make the lessons go away; it moves the walls. The bigger GPU hits the *same* walls, just
with a bigger model, a longer context, or more concurrent users, and the shape of what
happens when you get there is what this series is teaching you to recognize.

One boundary is worth drawing sharply, because it is the easiest place to over-extend these
lessons. Everything here is **single-GPU**: one GPU holding the whole model, where the wall
is always how fast that GPU moves weight bytes through its own memory. The moment you split
a model across several GPUs with tensor parallelism, which is how the 70B-class models are
actually served, a new wall appears that this series never touches: the chips must exchange
partial results every layer, and that cross-GPU traffic can bind before local memory
bandwidth does. The bottleneck moves from bytes-through-memory to bytes-through-the-network,
and the dials that matter change with it. So read this as a map of single-node inference. It
is the right map for one GPU, and the wrong one for a cluster.

So the absolute numbers are specific to this rig. What transfers is the *shape* of every
result and the *method* that produced it: pick one variable, push it until something gives,
and read why. That caveat holds for the whole series; I will not repeat it in every post.

## The arc

The posts are not a grab-bag. They tell one story about hitting the limits of a single
GPU and pushing them back. **New parts are published weekly** — the
[all-posts page]({{ '/articles/' | relative_url }}) is the quick table of contents;
the full arc:

1. **[Hit the wall]({{ '/articles/part-1/' | relative_url }})** *(published)*. Put the 4B
   model under load and find exactly where it saturates. The
   surprise: an 8.6 GB model on a 23 GB GPU, two-thirds empty, tops out at about seven
   requests a second. The wall is not memory, it is the decode loop itself, and which
   resource runs out first flips with model size. This post is the thesis; everything after
   pushes against the wall it establishes.

2. **[Read why, and push back]({{ '/articles/part-2/' | relative_url }})** *(published)*.
   Inject one fat prompt into a stream of short requests and
   watch it freeze everyone else's token stream. Then flip the chunked-prefill flag and
   watch the worst-case stutter drop by 2.4x. The twist: the same flag does *nothing* on a small
   model, which teaches you when an optimization is even measurable. The trace shows the
   freeze happening kernel by kernel.

3. **[The batching cliff]({{ '/articles/part-3/' | relative_url }})** *(published)*.
   Batching is the single lever that turns an idle GPU into a full
   one. Turning it off drops the 4B from ~1,091 to 49 tok/s, a ~22x cliff, and the win
   flattens once each sequence's own un-shareable work fills the step, which is why a bigger
   model gets *less* from batching, not more. What you give up is per-stream smoothness and, without it,
   fairness entirely.

4. **Cache starvation** *(coming)*. How a server slows down gracefully instead of crashing when memory
   gets tight, and the surprise underneath: on this hybrid model the KV cache is so hard to
   exhaust that forcing pressure took a 15x cache cut, and even then the scheduler throttled
   *admission* (a collapsed running batch) rather than firing the preemption the post set out
   to measure. The wrong prediction is the finding.

5. **Overcome the limit: quantization** *(coming)*. The finale. A 9B model that fp16 cannot serve
   *usefully* on this GPU is made to fit *and* serve at roughly 80% of the 4B's speed, using
   4-bit weights. Decode is bandwidth-bound, so what matters is bytes moved per token, and
   the 4-bit 9B's weights come out to only ~1.3x the 4B's fp16 weights (measured, not the
   2.25x its parameter count implies). The payoff of the whole arc: how you beat a hardware
   limit instead of just measuring it.

All five posts are backed by measurements on this exact model. Parts 3 and 4 were the
"earn-it" slots, strong ideas held back until re-measured on the 4B rather than back-filled
from the earlier small-model study, and that discipline paid off: Part 4's headline flipped
under re-measurement (the preemption it expected never fired), which is exactly the kind of
finding back-filling would have buried.

## How to read it

If the terms KV cache, prefill, decode, or batch are fuzzy, start with the
**[primer]({{ '/articles/primer/' | relative_url }})** ("How
an LLM actually serves a request"). It builds the one mechanical picture every post relies on,
a serving step as a single shared stream of the weights that advances a whole batch by one
token, and it deliberately stops before any of the findings. If those terms are already
second nature, skip it and go straight to [Part 1]({{ '/articles/part-1/' | relative_url }}).

Then start with Part 1; it sets up the wall the rest of the series pushes against. After that
the posts stand alone, but they gain from being read in order, since each one is a
different answer to the question Part 1 raises: *the GPU ran out of something under load,
so what do we do about it?*

Every post ends with a reproduce section: the exact scripts and raw logs, and where
available a downloadable `.json` trace you can open in `chrome://tracing` without owning a
GPU. (The version pin, vLLM 0.18.0, is stated once in Part 1's setup and holds throughout.)
If a number here matters to you, you should be able to argue with it by re-running
it. The scripts live in [`experiments/`](https://github.com/mapathak-commits/inference-wall/tree/main/experiments)
and the raw logs and traces in [`benchmarks/`](https://github.com/mapathak-commits/inference-wall/tree/main/benchmarks),
one folder per post.

## A standing note on honesty

Failures are reported as results, not hidden. A flag that does nothing on a small model,
a cold-start benchmark that lies by 200x, a path that is blocked by the hardware entirely,
these are in the posts on purpose. They are where the real learning is, and leaving them
out would make the series another too-clean explainer. When something did not work, you
will read why.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
