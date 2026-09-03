# Quantization as a fit-enabler: how a 9B model serves at 80% of a 4B's speed

*Draft 1. Part 5 (finale) of "The Inference Wall." Same rig
throughout: one NVIDIA A10G (23 GB). Qwen3.5-4B (fp16) vs Qwen3.5-9B (4-bit AWQ).*

---

This series opened by hitting a wall: a 4B model on one mid-range GPU, saturating at about
seven requests a second, bottlenecked not by memory but by how fast the GPU could push its
weights through the decode loop. Every post since has been about pushing that wall back.
This finale is about the bluntest lever of all, the one you reach for when the model you
*want* to run will not fit *usefully* on the GPU: **quantization.**

Here is the setup that makes the point. I wanted to serve Qwen3.5-9B, a model about twice
the size of the 4B, on the same 23 GB GPU. In its native format (fp16, meaning each of the
model's numbers is stored in 16 bits) its weights alone are about 18 GB (9 billion
parameters at 2 bytes each). On a 23 GB GPU at vLLM's default 0.9 memory fraction that
leaves under 3 GB for the KV cache, which is why I did not run fp16 here: even if it loads,
a KV budget that small serves only a trivial amount of concurrent context, which defeats the
point of standing the model up at all. I did not benchmark that degenerate config; the honest
framing for a 23 GB GPU is not "9B fp16 vs something faster," it is "**9B in a usable form,
or no 9B at all.**"

The surprising part is how small the penalty turns out to be. A 4-bit version of the 9B
not only fits, it **serves at roughly 80% of the 4B's throughput**, despite having more
than twice the parameters. This post is about why that lopsided trade exists, and it comes
straight back to the memory-bandwidth story the whole series has been building.

## What quantization actually is (the one-paragraph version)

A model's "weights" are just a giant pile of numbers. By default each is stored in 16 bits
(fp16, "half precision"). **Quantization** stores them in fewer bits, here 4 bits each,
using a scheme that picks the 4-bit levels carefully so the numbers stay close to their
originals. The immediate payoff is size: 4-bit weights take about a quarter of the bytes
of 16-bit weights. The scheme used here is **AWQ** (Activation-aware Weight Quantization),
which is designed to keep the quantized model's answers close to the original's, and vLLM
runs it with a fast GPU kernel called `awq_marlin`. You do not need the internals; you need
one fact, which the rest of the post leans on: **a 4-bit weight is ~4x fewer bytes to read
than the same weight in fp16.** (In practice a real 4-bit model keeps a few tensors, such as
the embeddings, in higher precision, so the whole-model shrink is less than a clean 4x, as
the memory table below shows.)

## First, does it fit? (this is the whole point)

Before any speed number, the memory table, because "it fits at all" is the result that
matters most. Here is how vLLM carves up the 23 GB GPU for each model, measured at load:

| | Qwen3.5-4B (fp16) | Qwen3.5-9B (AWQ 4-bit) |
|---|---|---|
| Weights | 8.61 GiB | **11.21 GiB** |
| Available KV cache | 9.45 GiB | 6.65 GiB |
| KV cache capacity | 77,088 tokens | **54,384 tokens** |
| Max concurrency @ 2,048 tok/req | 83.7x | **59.0x** |

![Stacked memory-budget bars for both models against the 23 GB GPU: the 4B in fp16 uses 8.61 GB of weights and 9.45 GB of KV cache, while the 9B in 4-bit AWQ uses 11.21 GB of weights and 6.65 GB of KV cache, so a model with 2.25x the parameters weighs only ~1.3x as much and still leaves room for a real cache](fig5a-memory-budget.png)

(Every cell is quoted from vLLM's own startup log for each model, the `gpu_worker.py`
"Available KV cache memory" line and the `kv_cache_utils.py` "GPU KV cache size" /
"Maximum concurrency" lines, not computed by hand. As Part 1 noted, the concurrency figure
is vLLM's own hybrid-aware count, not tokens divided by request length.)

Read the weights row. A **9B** model in 4-bit weighs **11.2 GiB**, only about 1.3x the
**4B**'s fp16 weights (8.6 GiB), even though it has 2.25x the parameters. That compression
is the entire reason it fits on the GPU at all: 11.2 GiB of weights leaves 6.65 GiB for
the KV cache (the server's per-request working memory), which is still room for **54,000
tokens, or 59 concurrent max-length requests**. The fp16 version's ~18 GiB of weights
would have left under 3 GiB for the KV cache, less than half of this, and a serving
budget that thin is not worth standing up. **Quantization did not make the 9B faster here;
it made it usable here.** That is the framing to hold onto.

## Now the surprise: it serves almost as fast as the smaller model

With the 9B-AWQ actually running, here is its warm serving curve under the same rate sweep
this series has used throughout (256-token in, 128-token out, warm server, percentiles).
The latency columns: TTFT is time-to-first-token (the wait before the answer starts), ITL
is inter-token latency (the gap between streamed tokens), E2E is the end-to-end total:

| Offered rate | Achieved req/s | Output tok/s | TTFT p50 | TTFT p99 | ITL p99 | E2E p99 |
|---|---|---|---|---|---|---|
| 1 /s  | 0.99 | 126 | 136 ms   | 214 ms    | 63 ms  | 3,799 ms |
| 2 /s  | 1.94 | 248 | 146 ms   | 267 ms    | 73 ms  | 4,456 ms |
| 4 /s  | 3.73 | 478 | 195 ms   | 401 ms    | 135 ms | 7,475 ms |
| 6 /s  | 4.84 | 619 | 581 ms   | 2,911 ms  | 342 ms | 20,405 ms |
| 8 /s  | 5.21 | 667 | 1,011 ms | 7,403 ms  | 457 ms | 20,697 ms |
| 16 /s | 6.06 | 775 | 2,550 ms | 12,998 ms | 612 ms | 25,363 ms |
| inf   | 6.42 | 821 | 7,649 ms | 23,275 ms | 612 ms | 31,142 ms |

Now put it next to the 4B fp16 from Part 1, at the same offered rates:

| | 4B fp16 | 9B AWQ |
|---|---|---|
| req/s at rate 4 | 3.77 | **3.73** |
| tok/s at rate 4 | 482 | **478** |
| Sustained req/s ceiling | ~7 to 8.5 | **~6.4** |
| Sustained tok/s ceiling | ~1,000 to 1,090 | **~820** |

Below the knee the two models are **nearly identical**: at rate 4 the 9B does 3.73 req/s
and 478 tok/s against the 4B's 3.77 and 482, a difference you would not notice. The ceiling
is only modestly lower: the 9B tops out around 6.4 req/s and 820 tok/s versus the 4B's
~7 to 8.5 and ~1,000. **A model with 2.25x the parameters serves at roughly 80% of the smaller
model's rate.** If you expected the bigger model to be roughly twice as slow, this should
be the surprise of the post.

![Two throughput curves against offered rate for the 4B fp16 and the 9B AWQ: they lie nearly on top of each other below the knee, then diverge at the ceiling to about 1,092 tokens a second for the 4B and 821 for the 9B, a model with more than twice the parameters running at roughly 80% of the smaller one's speed](fig5b-throughput-curves.png)

## Why the bigger model isn't much slower: it's the bytes, not the parameters

The explanation is the per-sequence cost from Parts 1 and 3. A decode step's cost per
sequence, the part that does not amortize across the batch and so sets the ceiling, is
dominated by *how many bytes of weights it moves*, not by the parameter count or the raw
arithmetic. That is the number to compare between two models.

And that reframes this comparison completely. Read the weights row of the table again: the
9B in 4-bit weighs **11.2 GiB** against the 4B-fp16's **8.6 GiB**, only about **1.3x the
bytes** even though it has 2.25x the parameters. Quantization took a would-be ~2.25x increase
in bytes-per-token and compressed it to ~1.3x.

Profiling both models settles which ratio governs. If you sweep the batch size and fit the
per-step decode time, it comes out as a fixed cost plus a per-sequence cost (the fit Part 1
introduced and Part 3 read off the batching sweep). The per-sequence term is where the model's size shows up, and measured from
traces it is **313 us/seq for the 4B and 395 us/seq for the 9B, a ratio of 1.26x.** That
lands on the *byte* ratio (1.30x) and nowhere near the *parameter* ratio (2.25x). If the
extra parameters were what cost you, the 9B would run at ~45% of the 4B's rate; because it is
the extra *bytes* that cost you, and quantization held those to 1.3x, it runs at ~1 / 1.3 ≈
**77%**, essentially the ~80% we measure. **That is the whole point: what a decode step
actually spends its time moving is bytes, so a model with more than twice the parameters, but
only 1.3x the bytes, serves at nearly the same speed.**

![Two rows comparing what a decode step moves: the 4B fp16 as a handful of large weight tiles, and the 9B 4-bit as 2.25x as many tiles each a quarter the size, so the two armloads of bytes come out nearly equal at about 1.3x rather than 2.25x](d8.jpg)

## The cost side (nothing is free)

The 4-bit path is not a pure win, and the curve shows where it pays:

- **Higher per-token latency at load.** At a matched offered rate the 9B's ITL p99 runs
  higher than the 4B's (342 ms vs ~130 ms at rate 6). Most of that is position on the curve,
  not the model: the 9B's ceiling is lower, so at the same offered rate it is nearer
  saturation and its queue inflates sooner. The `awq_marlin` kernel also has to *dequantize*
  the 4-bit weights each step, a small extra per-step cost the fp16 path skips, but that is
  the minor term: the measured per-sequence cost is only 1.26x the 4B's, nowhere near the
  2.6x the ITL gap at rate 6 might suggest.
- **A lower ceiling.** The 9B saturates around 6.4 req/s vs the 4B's ~7 to 8.5, roughly a
  fifth to a quarter lower sustained throughput.

So the trade is real: you spend some steady-state speed and some smoothness to buy the
ability to run a model that otherwise would not load. When the alternative is "the model
does not fit," that trade is overwhelmingly worth it. When you already have headroom, it is
a genuine judgment call, and the curve above is the kind of measurement that lets you make
it rather than guess.

(A note for the curious, kept out of the main thread because it is on different hardware: I
also compared 4-bit against fp16 head-to-head inside a single model on a separate rig, and
the shape held, the int4 win is mostly a property of *quantization itself*, not of any one
serving engine. The bytes-read mechanism is general.)

## What to take away

1. **Quantization's first job is to make a model usable, not to make it fast.** In fp16 the
   9B's ~18 GiB of weights would leave under 3 GiB for the KV cache on this GPU, too little
   to serve real context; 4-bit AWQ dropped the weights to 11.2 GiB and left room for 59
   concurrent requests. "Serves usefully or doesn't" is a bigger lever than any percentage
   speedup.
2. **Parameter count is the wrong unit for decode speed; bytes-read is the right one.**
   The 9B has 2.25x the parameters but, in 4-bit, only ~1.3x the weight bytes of the 4B in
   fp16, which is why it serves at ~80% of the speed (roughly 1 / 1.3) rather than the ~45%
   the parameter count would suggest.
3. **The 4-bit path costs you smoothness and ceiling, not correctness.** Expect modestly
   higher ITL under load and a throughput ceiling a fifth to a quarter lower. Answer
   quality held on this model; the price is in latency, not in the output.
4. **Measure the curve before you decide.** "Quantize it" is not automatically right when
   the model already fits; it is automatically right when it does not. In between, the rate
   sweep tells you what you are actually trading.

That closes the arc this series opened. Part 1 found the wall, an ordinary GPU running out
of decode throughput long before it runs out of memory. The middle posts read the trace to
see the wall and pushed it back with scheduling. This finale shows the other direction: when
the model is too big for the GPU, quantization changes the bytes-per-token math so
directly that a model which could not serve usefully becomes one that runs at nearly full speed.
The recurring lesson under all five posts is the same one: **on this hardware, inference is
a bytes-through-memory problem, and every real win comes from moving fewer bytes.**

---

*Reproduce: `run_sweep_9b.sh` (the `QuantTrio/Qwen3.5-9B-AWQ` sweep) and `verify_load.py`
(the memory split) are in the companion repo under `research/serving/qwen35/`; raw logs
`q35_9b_sweep.log`, `q35_9b_warm.log`, and `q35_9b_server.log` back every AWQ number here.
The fp16 9B was not run on this GPU, deliberately: its ~18 GiB weight figure is 9B x 2
bytes, and the "under 3 GiB left for KV" follows from the same 0.9 memory fraction the AWQ
run used, so the "not worth serving in fp16" call is an estimate, not a measured OOM.
Single A10G; absolute numbers are rig-specific, the bytes-read mechanism is not.*
