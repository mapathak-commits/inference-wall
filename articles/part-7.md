---
title: "Speculative decoding: the optimization that speeds up an idle server and slows down a busy one"
permalink: /articles/part-7/
image: /assets/diagrams/d9.jpg
---

*Part 7 of "The Inference Wall". Same rig as the whole series: Qwen3.5-4B, fp16,
one NVIDIA A10G with 23 GB, measured under real load.*

*Manas Pathak · October 9, 2026*

[The Inference Wall]({{ '/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }}) · **Part 7**

Speculative decoding is a technique for getting several tokens of one request out of a
single pass over the model's weights: a cheap guesser proposes a run of tokens and the
model verifies them all at once. The two founding papers,
[Leviathan, Kalman and Matias](https://arxiv.org/abs/2211.17192) and
[Chen et al.](https://arxiv.org/abs/2302.01318), report 2-to-2.5x single-stream
speedups, and serving guides generally present it as close to free. This post measures
vLLM's implementation on the series rig from an idle server to a saturated flood, and
the result splits in two. On a quiet server single-request generation runs up to **1.8x
faster**. Under production-style load the same
configuration **cuts total throughput from ~1,100 tokens a second to ~490** and raises
time-to-first-token from 5 seconds to 22. Which way it goes is set by how busy the
server is, and the documentation does not say where that line falls. The post pins the
line down, explains it through two measurable mechanisms, and ends at a profiler trace
showing that speculation replaces the model's decode loop with a different one.

One fact from [Part 1]({{ '/articles/part-1/' | relative_url }}) carries everything
below: generating text is *memory-bound*. Each token requires streaming the model's
entire weights, 8.6 GB here, out of GPU memory, about 20 ms on this rig. Batching shares
one stream across many requests, [Part 3]({{ '/articles/part-3/' | relative_url }}).
Quantization shrinks the bytes in the stream,
[Part 6]({{ '/articles/part-6/' | relative_url }}). Speculative decoding is the third
lever: several tokens *of the same request* from one stream.

## How speculative decoding works

![The big model is a press whose one wide arm verifies a whole row of proposed tokens in a single pass: accepted tiles come out green, rejected ones are crossed out and tumble off the track, while the small guesser runs ahead sketching the next tiles](../assets/diagrams/d9.jpg)

Call the model you are serving **L**, for large. L pays one full weight-stream per token
because each token of the answer is an *input* to the next: token 13 cannot be computed
until token 12 exists. A prompt's tokens all exist up front, which is why reading a
prompt is fast. The asymmetry speculation exploits is that *checking* k proposed tokens
also costs one pass: L runs them through as if they were a k-token prompt, and that pass
reveals what L itself would have produced at each of the k positions. One weight-stream,
k verdicts. A second, cheap guesser, **S** for small, supplies the proposals:

1. **S proposes** the next k tokens. The depth k is the knob; this post tests five and
   three, written **k5** and **k3**.
2. **L verifies** all k in one pass.
3. Walk the positions in order, accepting while S's token matches L's. At the first
   mismatch, discard that guess and everything after it, take L's own token for that
   position, and go back to step 1.

If all k guesses are right, one weight-stream bought k+1 tokens. If the first is wrong,
the pass bought one token plus the wasted work of checking dead proposals. The economics
reduce to the **acceptance rate**. The founding papers' 2 to 2.5x was measured one
request at a time, with a well-matched drafter, on tasks the drafter could predict; all
three qualifiers matter below.

## What S is in this experiment, and what "ngram" actually means

The classic S is a small language model from the same family, loaded alongside L; vLLM
0.18 also supports trained guessers like [EAGLE](https://arxiv.org/abs/2401.15077) and
[Medusa](https://arxiv.org/abs/2401.10774) that bolt a guessing head onto L itself. All
of those *compute* their candidates, which makes them useful on open-ended text, and all
cost something to run.

This post uses the simplest S available. The name misleads. vLLM's **ngram** method,
also known as [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding),
is **not a language model and keeps no table of n-gram statistics**. It never computes
or scores a candidate. Its one operation is a string search over the text this request
already holds, prompt plus output so far: take the last few tokens L just produced, find
the same phrase earlier in that text, and copy whatever followed it.

A worked example, with words standing in for tokens. The prompt is `Repeat this
sentence: The quick brown fox jumps over the lazy dog.` and L has so far produced `The
quick brown`. The text the server holds for this request is:

```
Repeat this sentence : The quick brown fox jumps over the lazy dog . The quick brown
```

S tries to find the closing phrase somewhere earlier in that text, a four-word phrase
first, then three, then two, the `prompt_lookup_max` and `prompt_lookup_min` settings.
The last four words, `. The quick brown`, appear nowhere earlier. The last three, `The
quick brown`, do: they are inside the prompt. The five tokens that followed them there,
`fox jumps over the lazy`, become the k5 proposal. S did not judge that `fox` is likely.
It copied `fox` because `fox` sat after `The quick brown` the one earlier time that
phrase appeared. L then verifies all five in one pass. Had there been no prompt, so that
the whole text was just `The quick brown`, there would be nothing earlier to search, S
would propose nothing, and the step would run as an ordinary decode.

So the search key is always text L has already committed, the candidates are always a
copy of what once followed it, and L is the only party that ever judges a token. S's
cost is a string search, which pins its share of every measurement below at zero:
**everything measured here is the cost and benefit of L's verification machinery**.

Prompt lookup is not a vLLM quirk. The same guesser ships in HuggingFace `transformers`
as `prompt_lookup_num_tokens`, and in TensorRT-LLM, SGLang, and TGI. Nor is the
inversion below engine-specific: the first mechanism is a property of any server that
batches requests continuously, the second of any hybrid-attention model, whichever
engine serves it.

In practice, ngram helps wherever the output reuses text the request already contains.
Copying from the input: summarization and retrieval-augmented answers that quote their
source, code edits that re-emit a function with a few lines changed, agent loops that
restate tool arguments. Copying from itself: JSON with repeated keys, lists on a fixed
per-item template, code reusing its own names, boilerplate. Where nothing recurs,
open-ended chat, creative writing, reasoning from scratch, the search finds no match and
S contributes nothing; that is the workload draft models and EAGLE exist for. The two
probe prompts below sit at the poles; real traffic lands in between.

## Single stream on an idle server

Single stream, temperature 0, 160 output tokens, three repetitions, spread under 1%:

| Prompt | Spec off | k5 = guess 5 ahead | k3 = guess 3 ahead |
|---|---|---|---|
| predictable, repeat a sentence | 20.05 ms/token | **11.2 ms/token, 1.79x** | 12.25, 1.64x |
| creative, a surreal poem | 20.05 ms/token | 19.4, no change | 21.9, **0.92x, a real penalty** |

The baseline is Part 1's floor re-measured, 20.05 ms per token regardless of the text.
Speculation breaks that in both directions: 1.8x on text the lookup can predict, nothing
on text it cannot, and at k3 an 8% penalty for machinery that never once paid off. The
animation replays both prompts against the k5 server using the measured arrival process,
bursts of about three tokens every 36 ms in the top pane and a steady one token per
19 ms in the bottom. Playback is slowed 4x; the clocks are real.

![Two token streams under k5 speculation: predictable text arrives in bursts and finishes early, creative text ticks token by token](../assets/figures/fig7-token-stream.gif)

On the 1.8x against the papers' 2 to 2.5x, and the 3.7x an earlier small-model study of
mine clocked for this method on a purely repetitive prompt: the repeat-a-sentence prompt
accepts nearly everything early on, but this model drifts into free-form reasoning text
partway through, which a lookup cannot predict, and over the full run the **mean
accepted length is about 2** per verify step out of a possible 6. A probe that catches
only the early window reports 4x; 1.8x is what a real 160-token generation got.

## The same configuration under load

The series' standard measurement: 256 input and 128 output tokens, randomly generated,
the request rate swept from 1 per second to a flood, 200 prompts per point, warm server.
Random tokens are the acceptance-hostile extreme, and the server metrics confirm it: in
the k5 arm **about 21% of drafted tokens are accepted**. Output tokens per second against
offered rate, for all three arms:

![Three throughput curves against offered rate: with speculation off the server climbs to about 1,100 tokens a second; with k3 or k5 speculation it flattens near 540 and 470 from rate 4 onward](../assets/figures/fig7a-sweep-inversion.png)

At rates 1 and 2 the three servers are indistinguishable: the GPU has idle headroom and
wasted verification vanishes into it. From rate 4 the speculation arms fall behind, and
past the knee they collapse. Under flood, k5 sustains **473 tokens a second against the
baseline's 1,096**; re-run for stability, 491 and 493 against 1,097 both times. Median
time-to-first-token under flood is 22 seconds versus 5. The knee moves from rate 6-to-8
down to between 3 and 4: not just a lower ceiling, a halved operating range. k3 shows
the dose-response, 542 at flood, still half the server.

One more probe separates load from workload. Hold the load at 32 simultaneous requests
and change only the text:

| 32 concurrent requests | Spec off | k5 | k3 |
|---|---|---|---|
| predictable text | 1,035 tok/s | 1,183, up 14% | 1,293, up 25% |
| creative text | 1,071 tok/s | **577, down 46%** | 811, down 24% |

The off column does not care what it writes. The speculation columns swing by 2x between
the same two rows. **Under load, acceptance rate decides the sign.**

## Why it inverts, mechanism 1: speculation multiplies the work batching cannot share

A decode step's work has two parts, measured in Part 3:

- **Shared:** streaming the 8.6 GB of weights. One stream serves the whole batch,
  whether 1 request or 146 ride it. This is what batching amortizes.
- **Private:** each request also processes the new token against its own cached
  history. This cannot be shared; the step pays it once per request, every step.

As the batch grows the shared stream is split ever thinner while private work
accumulates; at a batch of about 61 the private work is what fills the step, and a
loaded server lives past that point.

Speculation leaves the shared stream alone; that is the point, same stream, more
verdicts. The **private work it multiplies by k+1**: at k5, verifying a request means
processing six positions against its private history instead of one. On an idle server
the extra private work hides in the stream's shadow, which is why rates 1 and 2 showed
no cost. On a loaded server the private work *is* the step, and multiplying it by six
while only 21% of positions survive means most of every step computes verdicts for
tokens that get thrown away. **Speculation and batching compete for the same headroom,
and under load batching has already spent it.**

## Why it inverts, mechanism 2: state checkpoints cut concurrency

The second mechanism shows up in the startup log. To discard a wrong guess, the server
must *rewind* the model's state to before the guess. For a standard transformer that is
trivial: the model's memory of the conversation is a per-token KV cache, and rewinding
three tokens means truncating three entries. Qwen3.5 is a *hybrid-attention* model. Only
8 of its 32 layers keep that per-token cache; the other 24 keep a **fixed-size recurrent
state**, a single running summary overwritten as each token is processed. There is
nothing to truncate; once updated, the old state is gone. vLLM's solution is
checkpointing: with k5 armed it budgets **six state slots per request**, one per
speculated position plus the base.

Those slots come out of the same memory budget that sets how many requests the server
can hold at once, so each request now costs six seats:

![Two bar panels comparing spec off with k5: maximum concurrency falls from 83.7x to 24.6x at startup, and the running batch under flood falls from 146 requests to 28](../assets/figures/fig7b-seat-collapse.png)

Max concurrency in the startup log falls from **83.71x to 24.64x**, and under flood the
scheduler's `Running:` count falls from **146 requests to 28**, the rest parked in the
waiting queue. This is the admission-control signature from Part 5, where it took a
deliberate 15x cache cut; here an optimization flag did it. Before a single wasted draft
is counted, 28 requests sharing each weight-stream cannot approach the throughput of
146. The two mechanisms compound: wasted verdicts inside each step, fewer requests
allowed into the step.

## The trace: a verify loop in place of the decode loop

The capture: eight long-lived predictable-text decoders against the k5 server, no
arrivals, 35,202 GPU kernels over two seconds.

The decode kernel this series has leaned on since Part 1, the one-token-per-pass
linear-attention recurrence `fused_recurrent_gated_delta_rule`, appears **zero times**.
Every step instead runs the multi-token variant of the same layer,
`fused_sigmoid_gating_delta_rule_update`: 1,320 calls, which at 24 linear-attention
layers per pass is exactly 55 engine steps. Speculation does not bolt machinery onto the
decode loop; it swaps the loop out for a verify loop, a kernel doing prompt-reading-shaped
work at generation time.

A verify step takes **36 ms** where a decode step at this batch size takes about 20,
but it processes six positions per sequence instead of one: **6 ms per position, versus
20**. That is the gain. The risk sits in the same number: at the flood workload's 21%
acceptance, the same 36 ms step keeps only about two tokens, roughly 18 ms per
*accepted* token, before the collapsed batch is counted. And during verify-heavy decode
the GPU is only **80% busy**, against 97 to 99.7% in Part 1's pure decode loop; the
proposer and the accept-reject bookkeeping between steps leave real gaps.

## Does it make the answers worse?

No, by construction. The accept-reject rule makes the output provably come from the same
distribution L alone would produce; the proof is in
[Leviathan et al.](https://arxiv.org/abs/2211.17192). At temperature 0 it needs no math:
a guess is accepted only if it is exactly the token L would have picked.

Two qualifications. Same distribution does not mean same text: within each configuration
our temperature-0 runs were byte-identical across three repetitions, but across
configurations they diverged, k3 writing a different poem than k5. Verification changes
the GPU kernel shapes, kernel shapes perturb logits in their last decimal places, and at
a near-tie a flipped choice cascades, the same nondeterminism batch size already causes.
Neither continuation is worse, but tests that assert exact strings will fail. And the
guarantee belongs to the *strict* acceptance rule; variants that relax it, Medusa's
"typical acceptance" for one, do change the output distribution. To verify rather than
trust, run the eval suite you already use with speculation off and on and compare scores
within noise.

## What to take away

1. **Speculative decoding spends idle capacity to buy latency.** With slack, a single
   stream on a quiet GPU, it converts unused verify headroom into a real speedup, 1.8x
   here on predictable text. Without slack, the spending continues and the buying stops.
2. **Under load it can invert, hard.** On this rig and a low-acceptance workload it cut
   saturated throughput from 1,096 to 473 tokens a second and moved the knee from rate
   6-to-8 to 3-to-4, by multiplying the per-request work batching cannot amortize.
3. **On hybrid models it also cuts concurrency.** Rewinding recurrent state means k+1
   checkpoints per request: max concurrency 83.7x to 24.6x, flooded running batch 146 to
   28. Check your engine's startup concurrency line before and after enabling it.
4. **The decision is a workload measurement.** At the same 32-request concurrency, k5
   gained 14% on predictable text and lost 46% on novel text. The server's acceptance
   metrics tell you where your traffic sits; read them before enabling it and whenever
   traffic changes.
5. **Quality is preserved by design; reproducibility is not.** Strict acceptance keeps
   the output distribution; exact temperature-0 strings will still change, as they do
   with batch size.

Batching, quantization, and speculation are all ways of getting more tokens out of the
same stream of bytes. The first two spend little and stack cleanly. This one pays only
when a guess is accepted, and the cost of a miss grows with load.

---

*Reproduce: the A/B driver, probes, trace analyzers, and figure renderers are in the
companion repo under
[`experiments/07-speculative-decoding/`](https://github.com/mapathak-commits/inference-wall/tree/main/experiments/07-speculative-decoding);
the raw logs, the per-arm server logs carrying the acceptance metrics and `Running:`
lines, and the verify-step trace `spec_verify_trace.json.gz`, openable in
`chrome://tracing`, are in
[`benchmarks/07-speculative-decoding/`](https://github.com/mapathak-commits/inference-wall/tree/main/benchmarks/07-speculative-decoding).
Single A10G, vLLM 0.18.0; the absolute numbers are rig-specific.*

*Further reading: the two founding papers,
[Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)
and [Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318);
[prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding), the
model-free guesser used here; [EAGLE](https://arxiv.org/abs/2401.15077) and
[Medusa](https://arxiv.org/abs/2401.10774), trained-guesser variants; and
[vLLM's speculative-decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode.html)
for the configuration surface.*

---

**Previous:** [Part 6: Quantization to make a model fit]({{ '/articles/part-6/' | relative_url }}) · **Next:** Part 8, coming next Friday · [All posts]({{ '/articles/' | relative_url }})

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
