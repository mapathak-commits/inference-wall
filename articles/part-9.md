---
title: "Off the rig: decode on a CPU, and the machine that could see its own bottleneck"
permalink: /articles/part-9/
---

*Part 9 of "The Inference Wall", run off the series' usual rig: Qwen3.5-4B on two very
different CPUs, an AMD EPYC server and an Apple M4 Pro laptop, with each machine's own GPU
alongside for comparison. Draft.*

*Manas Pathak · draft, September 2026*

[The Inference Wall]({{ '/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }}) · **Part 9**

Here are two files. Both hold the same 4-billion-parameter language model. Both are exactly
8,665,620,192 bytes, down to the byte. The only difference is the numeric format the weights
are stored in: one is `bf16`, the other is `F16`. These are two 16-bit floating-point layouts
that carry the same information in the same number of bytes, so a model that must read every
weight once to emit a token should run at exactly the same speed from either file.

On a laptop, one of them decodes **5.2x faster than the other**. On a server, they run at
**identical speed**. Same two files, same model, same benchmark. The gap appears on the fast
machine and vanishes on the slow one, which is the opposite of what "the laptop is just
better" would predict. This post is about why, and about what that one inversion says about
how you should reason about inference speed on any machine.

The one fact you need is the claim this whole series is built on: **decode, the phase where a
model emits its answer one token at a time, is bandwidth-bound.** To produce each token the
machine streams the entire model's weights through memory once, so decode speed is close to
`memory_bandwidth ÷ bytes_per_token`, and it tracks how fast the memory bus can move bytes,
not how many arithmetic operations the chip can do. The [primer]({{ '/articles/primer/' | relative_url }})
builds that picture from scratch, and [Part 1]({{ '/articles/part-1/' | relative_url }})
measures it on a GPU. Everything below is that same claim, taken off the GPU and pushed onto
hardware it was never tested on, to see if it survives, and to see what it teaches when it
starts to fray.

## Why bother running an LLM on a CPU at all

The whole series so far serves the model on a GPU, because that is what everyone does and it
is the right answer. But "you need a GPU" is received wisdom, and received wisdom is worth
poking. CPUs have had fifty years of engineering poured into them: deep cache hierarchies,
aggressive prefetchers, wide vector units, and on Apple's chips a dedicated matrix
coprocessor. The interesting question is not "is a CPU slower" (it is), but *by how much, and
why exactly*. If decode really is bytes-through-memory, then a CPU's decode speed should be
set by its memory bandwidth and nothing else, and the CPU-vs-GPU decode gap should be the
*bandwidth* gap between them, not the much larger *compute* gap. That is a sharp, falsifiable
prediction, and CPUs are where you can test it cleanly, because you can pick two CPUs whose
memory systems are wildly different and watch the gap move with the bandwidth.

Two machines, chosen to be as different as possible on exactly that axis:

- **An AMD EPYC server** (Zen 2, 8 cores) with DDR4 memory. Its measured streaming bandwidth
  is about **33 GB/s**. This is the same box the rest of the series runs on, and it also holds
  the series' NVIDIA A10G GPU, whose memory bandwidth is roughly 600 GB/s. So on this machine
  the CPU and GPU sit on *separate* memory pools with a large bandwidth cliff between them.

- **An Apple M4 Pro laptop** (10 performance cores + 4 efficiency cores) with LPDDR5X memory.
  Its measured streaming bandwidth is about **238 GB/s**, roughly seven times the EPYC's. Its
  key feature is *unified memory*: the CPU and the on-chip GPU share the **same** physical
  memory and the **same** bandwidth. There is no cliff between them, by construction.

That last property is the whole reason the M4 is interesting. On the EPYC, moving decode from
the CPU to the GPU means moving to a memory system with ~18x more bandwidth. On the M4, moving
from the CPU to the GPU changes the compute engine but *not the memory bus*. If decode is
bandwidth-bound, the CPU-to-GPU decode gap should nearly disappear on the M4 and stay large on
the EPYC, and the difference between the two should be almost entirely the difference in their
memory topologies. That is the experiment.

The model is the series' `Qwen3.5-4B`, the same weights throughout, run under
[`llama.cpp`](https://github.com/ggml-org/llama.cpp), which is the one engine that runs the
identical build on a CPU, an Apple GPU via Metal, and an NVIDIA GPU via CUDA. That matters:
it means every CPU-vs-GPU number below compares the same code on the same weights, changing
only which silicon executes it. `pp512` in the tables is prefill throughput, `tg128` is
decode throughput, both in tokens per second.

## The gap collapses, exactly as far as the bandwidth says it should

Start with the headline. Here is single-stream decode, same model, held at a fixed weight
format so it is a fair fight, across all three engines:

| precision | A10G GPU | M4 Pro CPU | EPYC CPU |
|---|---|---|---|
| **F16** (16-bit) decode | 54.6 tok/s | 20.5 tok/s | 4.0 tok/s |
| **Q4** (4-bit) decode | 121.3 tok/s | 57.3 tok/s | 9.9 tok/s |

Read it as ratios against the GPU, which is the gap this series cares about:

| gap (same precision) | F16 | Q4 |
|---|---|---|
| A10G GPU → EPYC CPU | **13.5x** | **12.3x** |
| A10G GPU → M4 Pro CPU | **2.7x** | **2.1x** |

The EPYC CPU is an order of magnitude behind its own GPU. The M4 CPU is only about 2.5x
behind. That is the collapse: same kind of comparison, same model, and the gap shrinks by
5x when you move to a machine whose CPU has fast memory.

Now the part that makes it a *result* rather than an observation. The bandwidth-bound thesis
does not just say "the gap should be smaller on the M4." It says *how much* smaller, in
advance, from the memory bandwidths alone. Decode gap should equal bandwidth gap. So:

| box | GPU bandwidth ÷ CPU bandwidth | decode gap the thesis predicts | decode gap measured |
|---|---|---|---|
| EPYC + A10G | ~600 / 33 = **~18x** (nominal), ~13x achieved | ~13x | **12.5x** |
| M4 Pro (unified) | 238 / 238 = **1.0x** | ~1x | **1.09x** |

On the M4 the CPU-vs-Metal decode gap, measured carefully by interleaving the two engines so
they share a thermal state, is **1.089x** (median of twelve paired runs; in one round the CPU
actually won). Against a bandwidth ratio of 1.0x. On the EPYC the gap is 12.5x against a
bandwidth ratio of ~13x. **Bandwidth predicts the decode gap to within about 10% on two
machines whose bandwidth ratios differ by more than 13x.**

For contrast, the *compute* gap between these CPUs and their GPUs is 43x (EPYC) and 2.7x (M4).
If decode tracked compute, the M4 gap would be 2.7x, not 1.09x, and the EPYC gap would be 43x,
not 12.5x. Compute is wrong by a factor of 2.5 to 3.4 on both boxes. Bandwidth is right on
both. That is the thesis surviving a port to hardware it was never fit on, which is the only
kind of survival worth much.

One honest note on what each machine contributes. The EPYC tests the thesis with a *large*
prediction (13x), which is discriminating: lots of wrong models cannot fake a 13x gap. The M4
tests it with a *null* prediction (1x), which is intrinsically weaker, because plenty of wrong
models also predict "about the same." The M4's real evidentiary weight comes from a tighter
fit I will get to below, not from the 1.09x on its own.

## The trick that made the "CPU beats GPU" headline, and why it is a trick

You may have seen the claim that an Apple laptop can match or beat a datacenter GPU at LLM
inference. Look back at the first table and you can see where that comes from, and why it does
not mean what it sounds like. The M4 CPU does Q4 decode at 57 tok/s. The A10G does *F16*
decode at 55 tok/s. Put those two numbers side by side, and the laptop CPU "beats" the GPU.

But those are different precisions. Q4 is 4-bit weights, one quarter the bytes per token of
F16. The comparison quietly gave the laptop a model that streams a quarter of the bytes and
gave the GPU one that streams four times as many, then reported the laptop as faster. Hold
the precision fixed and the illusion dissolves: at Q4 the A10G does 121 tok/s to the M4 CPU's
57, so the GPU is 2.1x ahead; at F16 it is 55 to 20, 2.7x ahead. The GPU wins at *every*
precision. It has more memory bandwidth, and decode is bandwidth-bound, so it decodes faster,
full stop.

The genuinely surprising thing is not that the laptop CPU beats the GPU (it does not). It is
that a laptop CPU gets within 2.5x of a datacenter GPU *at all*, which no CPU could do a
hardware generation ago. That closeness is entirely the LPDDR5X memory: the M4 CPU pulls
about 172 GB/s of effective decode bandwidth, versus about 34 GB/s on the EPYC. The 5x
CPU-to-CPU gap between the two laptops-and-servers is the DDR4-to-LPDDR5X gap, nothing else.

## The two identical files, and the instrument that could see the difference

Now back to the puzzle from the top: two byte-identical files, `bf16` and `F16`, decoding at
5.2x different speeds on one machine and identical speed on the other.

Here are the numbers. Same model, both files exactly 8,665,620,192 bytes, decode tokens per
second:

| machine | bf16 | F16 | F16 / bf16 |
|---|---|---|---|
| M4 Pro CPU | 3.92 | 20.45 | **5.22x** |
| EPYC CPU | 4.02, 3.99 | 4.04, 4.03 | **1.005x** |

Since the two files are the same size, bandwidth cannot explain any of the difference. The
bytes-per-token term is identical. So the M4's 5.2x has to come from something other than
memory, and it does: `llama.cpp` has a fast, hand-tuned kernel for `F16` on Apple's NEON
vector units and only a slow, generic path for `bf16`. On the M4, decode from the `bf16` file
is bottlenecked on that slow kernel converting each weight, long before it reaches the memory
wall. Swap to the `F16` file and the same bytes now flow through a fast kernel, and decode
jumps to 20 tok/s, which puts it back at the memory wall (about 177 GB/s effective). The 5.2x
is a pure software gap, measured with the hardware and the byte count held perfectly constant.

Now the EPYC. Same two files, same slow `bf16` kernel (it is the same `llama.cpp`). And yet
`bf16` and `F16` decode at *the same speed*, 4.0 tok/s either way. How can the slow kernel not
cost anything here, when it cost 5.2x on the M4?

Because on the EPYC the memory bus is so slow that the slow kernel is not the bottleneck. At
33 GB/s of DRAM bandwidth, streaming 8.6 GB of weights per token already caps decode at about
4 tok/s, and that cap is *below* the speed the slow `bf16` kernel can sustain. The kernel is
slow, but the memory is slower, so the memory wall is what you hit first, and fixing the
kernel (moving to `F16`) buys nothing because you are still stuck behind the memory. On the
M4, with 7x the bandwidth, the memory wall sits far above the slow kernel's ceiling, so the
kernel becomes the thing you hit first and fixing it buys the full 5.2x.

Here is the sentence this whole experiment exists to earn. **A slow machine cannot tell the
difference between "at the memory wall" and "held back by a slow kernel that happens to run at
the same speed as the memory wall."** On the EPYC, `bf16` decode runs at 34 GB/s effective,
which is exactly its memory ceiling, and reads as a clean confirmation that decode is
bandwidth-bound. On the M4, the identical `bf16` decode also runs at 34 GB/s effective, but
here the memory ceiling is 238 GB/s, so that same 34 GB/s is provably *not* the memory wall,
it is the kernel. The two explanations are degenerate on the slow box and only separate on the
fast one.

This is worth internalizing as a general habit, not a fact about `bf16`. When you measure an
inference number and it lands right on a hardware ceiling, that agreement is only evidence if
the ceiling is high enough that a coincidence would be unlikely. A low ceiling can rubber-stamp
a wrong explanation, because too many different bottlenecks all read out at the same low speed.
**Fast hardware is a better measuring instrument, not just faster hardware**: raising the
memory ceiling by 7x separated two explanations that were indistinguishable underneath it. I
came to this because an earlier version of the CPU study, run only on the EPYC, cited the
`bf16` row as one of its clean confirmations of the thesis. It was a right conclusion resting
partly on a coincidence, and it took the faster machine to notice.

## Where the simple thesis starts to fray

"Decode speed = bandwidth ÷ bytes-per-token" is the version of the claim you can hold in your
head, and on a slow machine it is almost exactly right. On a fast machine it starts to leak,
and the leak is instructive.

The clean test of the simple claim is this: decode speed times bytes-per-token should be a
*constant*, equal to the memory bandwidth, no matter which quantization you run, because it is
just measuring the bandwidth from three different angles. On the EPYC it is beautifully
constant: bf16, Q8, and Q4 all give 30 to 34 GB/s effective, all pinned to the 33 GB/s wall.
On the M4 it is *not* constant: effective bandwidth climbs from 173 GB/s at Q4 to 224 GB/s at
F16. The thesis did not fail, but its simplest form ran out of resolution.

The reason is that each token also carries a small *fixed* cost that has nothing to do with
streaming weights, the per-token overhead of running the model's control logic and
synchronizing the threads once per token. Model it as one line:

```
time_per_token = weight_bytes / bandwidth + fixed_overhead
```

Fit that to the M4's decode times across four quantizations and it is almost perfect (the fit
explains 99.98% of the variance). It reports a bandwidth slope of about 209 GB/s, which is 88%
of the measured streaming ceiling, and a fixed overhead of a few milliseconds per token. The
striking part: the same fit on the M4's *GPU* gives an almost identical bandwidth slope, about
205 GB/s, while the two engines differ in peak compute by 2.7x. The entire residual difference
between the CPU and GPU lives in the fixed-overhead term (about 2.4 ms on the GPU versus 4.6 ms
on the CPU), not in the streaming. This is the tight fit that gives the M4's "1x" its real
evidentiary weight: it is not just that the gap is small, it is that decode time is a straight
line in weight-bytes whose *slope is the memory bandwidth*, and that slope is the same on two
engines with wildly different compute.

Why does the fixed cost matter more on the fast machine? Because it did not shrink when the
memory got faster. On the EPYC, streaming 3 GB of Q4 weights takes about 91 ms and the fixed
cost is around 10 ms, so overhead is under 10% of a token and you can ignore it. On the M4 the
same streaming takes about 14 ms while the fixed cost is a few ms, so overhead is now a
*quarter* of the token and you cannot. This is Amdahl's law pointed at the inference wall:
**the faster the memory, the larger a share the fixed per-token overhead becomes, and the
sooner it, rather than bandwidth, is the thing worth optimizing.** The constant-effective-
bandwidth test is a low-bandwidth diagnostic; it works precisely because the EPYC is slow, and
it is the first thing to break when you make the memory fast.

## The advantage that no throughput table shows: energy

On the M4 the GPU is only 1.09x faster at decode than the CPU, and the fit above says that for
a large enough model the gap goes to exactly 1x, since the only term that differs is a
constant. Which invites the conclusion that on a unified-memory machine the GPU is pointless
for decode: same bandwidth, same speed, why bother.

That conclusion is wrong, and the reason is a metric none of the throughput tables carry. I
measured power with Apple's `powermetrics` while decoding. For essentially the same tokens per
second, the CPU burns about 42 W while the GPU burns about 26 W, so the GPU does the work for
**about 1.75x less energy per token** (this reproduced across two independent captures at 1.78x
and 1.73x). The mechanism is visible in the clocks: CPU decode pins all the performance cores at
100% and runs them against the chip's power ceiling; the same work on the GPU leaves the CPU
nearly idle.

So on unified memory the honest guidance inverts depending on what is scarce. If you care about
tokens per second, the engine barely matters for decode and you should spend your effort on
quantization and on that fixed per-token overhead. If you care about battery or heat, which on a
laptop you usually do, the engine matters a great deal and you should run decode on the GPU. The
question "which engine should decode run on?" has no bandwidth-based answer on this machine. It
has a power-based one.

## A rule that inverts: do not split the model across engines

One more result specific to unified memory, because it flatly contradicts the discrete-GPU
habit. On a normal GPU box you offload as many of the model's layers to the GPU as its memory
will hold, because any layer on the GPU is a layer running on faster hardware, so more offload
is always better up to the memory limit.

On the M4 I swept that offload knob, moving layers from the CPU to the Metal GPU a few at a
time, and decode got *slower* across most of the range: any split from a few layers up to about
two-thirds of the model runs decode at 0.80 to 0.87x of running everything on the CPU. It only
breaks even near full offload. There is no bus to blame here, the memory is unified, so nothing
crosses anything when a layer moves. The cost is pure synchronization: splitting the model
between two engines that alternate on the same memory costs a flat ~7 ms per token just for the
handoff, and each layer you actually move to the GPU only buys back ~0.35 ms. You have to move
about twenty layers before the buyback covers the handoff. The same handoff is 28x cheaper per
token during prefill, because a 512-token prefill batch pays it once and amortizes it across all
512 tokens, while decode pays it fresh for every single token.

So on unified memory the rule is: **pick one engine and commit.** Partial offload exists on
discrete GPUs to cope with limited GPU memory; on a machine with one big shared memory pool
there is no capacity problem to cope with, so the feature has no upside and a real cost. It is a
clean example of an optimization whose sign flips when the hardware underneath it changes.

## What survives, and what to take away

The claim the series is built on, that decode is bytes-through-memory and tracks bandwidth
rather than compute, survives being carried onto two CPUs and a unified-memory laptop, and it
survives in the harder direction: on a machine engineered to erase the very bandwidth gap the
original GPU result leaned on, the CPU-to-GPU decode gap collapsed from 12.5x to 1.09x, exactly
as far as the bandwidth gap collapsed from 13x to 1x. The prefill gap, meanwhile, stayed put at
about 3x against a 2.7x compute ratio, because prefill *is* compute-bound and does not care
about the memory topology. Same machine, both phases, the compute-versus-bandwidth split laid
out in one pair of columns.

What the port adds is where the simple model frays, and each fray is a lesson:

- The tidy "effective bandwidth is constant across quantizations" test is a *slow-machine*
  diagnostic. Speed the memory up and per-token overhead stops being a rounding error, and the
  two-parameter fit, not the one-liner, is the right model.
- A number landing on a hardware ceiling is only evidence if the ceiling is high. A low ceiling
  can confirm a wrong explanation, because many different bottlenecks read out at the same low
  speed. The byte-identical `bf16`-versus-`F16` pair, at 1.005x on the slow box and 5.22x on the
  fast one, is the whole argument in two files.
- On unified memory, decode's throughput advantage from the GPU nearly vanishes but its *energy*
  advantage does not, and an optimization as basic as "offload layers to the GPU" inverts its
  sign.

The bottom line for anyone deciding whether to run a model on a CPU: you can, and on a
modern high-bandwidth CPU you land within ~2.5x of a mid-range GPU on single-stream decode, for
a quarter of the fuss. What you give up is not really per-token latency any more; it is
*batching*. The GPU's real headline in this series was never single-stream decode, it was
serving many requests at once, where it turns one weight-read into progress for a hundred
sequences. A bandwidth-saturated CPU has no spare bytes per second to amortize that way, so its
batched throughput is roughly its single-stream throughput, and the gap that closed to 2.5x on
one stream reopens to something like 20x under real serving load. Decode on a CPU is viable.
*Serving* on a CPU is a different wall, and the series will get to it.

## Reproduce

Everything here is `llama.cpp` at build `458681e` (2026-09-01), one build per machine, driven
by `llama-bench`. The model is `Qwen/Qwen3.5-4B` in bf16, F16, Q8_0, and Q4_K_M GGUFs; the F16
file was produced from the bf16 file with `llama-quantize ... F16` so the two are byte-identical
(8,665,620,192 bytes each), which is what makes the kernel-versus-memory A/B clean. CPU runs
use `-ngl 0`, Apple GPU runs use `-ngl 99` (Metal), NVIDIA runs use `-ngl 99` (CUDA). The
CPU-vs-GPU decode ratio on the M4 is measured by interleaving the two engines round-robin so
each pair shares a thermal state, because absolute tokens per second on a laptop drift up to 20%
over a long benchmarking session. Power is from `sudo powermetrics --samplers cpu_power,gpu_power`.
The full staged study, including the microbenchmarks, the roofline fit, the offload sweep, the
long-context KV-cache results, and an audit pass that reran every surprising number, lives in
the companion repo.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
