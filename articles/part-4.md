---
title: "Starving the cache: how a server degrades when it runs out of KV"
permalink: /articles/part-4/
---

*Part 4 of "The Inference Wall", on the series' usual rig: Qwen2.5-7B and Qwen3.5-4B on a
single 23 GB NVIDIA A10G, served under vLLM. Draft.*

*Manas Pathak · draft, September 2026*

[The Inference Wall]({{ '/' | relative_url }}) · [All posts]({{ '/articles/' | relative_url }}) · **Part 4**

An inference server has a fixed amount of memory for KV cache, the running scratchpad it keeps
for every request in flight. When more requests want to run than the cache can seat, something
has to give, and the server has exactly two honest options. It can refuse to start new work
until room frees up — **admission control**, where a request waits in a queue rather than
running with nowhere to store its keys and values. Or it can start the work optimistically and,
when the cache later fills mid-generation, **evict** a running request: throw away its cached
state, hand the blocks to someone else, and reconstruct the evicted request's state later. The
one option nobody ships is the naive one: admit everything, run out of memory mid-decode, and
crash.

Every modern engine picks a point on that spectrum, and the point it picks is a real design
decision with observable consequences. This post is about watching one server, vLLM, actually
degrade under a starved cache — not reasoning about what it should do, but flooding it until the
cache is the binding constraint and reading the counter that says what happened. The short
version is that vLLM lands firmly on the *evict* end of the spectrum, its eviction backstop
fires far more readily than its own documentation's tone suggests, and the cost you pay for it
is not a crash but a latency you can see coming from the memory math. Getting to that answer
also required noticing that the most obvious way to measure it — grepping the server log — is
silently broken, which is a lesson worth more than the result.

## The two ends of the spectrum, across engines

Before the measurements, it helps to see that "admit or evict" is not a vLLM quirk but the axis
every serving engine sorts itself along. The engines cluster into two camps.

**Pure admission control — never evict a running request.** Hugging Face's Text Generation
Inference profiles the GPU at startup to learn how many tokens of KV it can physically hold,
then enforces a strict token budget (`max_batch_total_tokens`) in its Rust router: a request
joins the running batch only if its tokens fit, and otherwise waits in the client-facing queue.
A request that has started decoding is never paused to free its memory; TGI's waiting-vs-running
knobs only ever pause decode briefly to let a queued request run its prefill and *join*, never
to reclaim a running request's blocks. TensorRT-LLM's default capacity policy,
`GUARANTEED_NO_EVICT`, is the same shape drawn conservatively: it admits a request only if it can
reserve enough blocks for the request's entire declared output length up front, so once running,
a request is guaranteed its slot to completion.

**Evict-capable — admit optimistically, pull work back under pressure.** vLLM admits requests
without reserving room for their eventual output, and when the pool runs dry mid-decode it
preempts a running request. SGLang does the same in two stages: it first evicts unreferenced
cached prefixes from its radix tree, and if that is not enough, `retract_decode` pulls running
requests out of the batch and back to the queue. TensorRT-LLM will behave this way too, but only
if you opt into it by setting the capacity policy to `MAX_UTILIZATION` instead of the default.

| Engine | Default under KV exhaustion | Mechanism | Evicts a running request? |
|---|---|---|---|
| **TGI** (Hugging Face) | Cap admission by token budget | `max_batch_total_tokens`, profiled at warmup | **Never** — queues at the router |
| **TensorRT-LLM** (default) | Reserve full output up front | `GUARANTEED_NO_EVICT` | **Never** under default |
| **vLLM** | Preempt and recompute | `RECOMPUTE` (V1 default; host-swap dropped in V1) | **Yes** |
| **SGLang** | Evict prefixes, then retract | radix-cache LRU eviction, then `retract_decode` | **Yes** |
| **TensorRT-LLM** (`MAX_UTILIZATION`) | Pack aggressively, evict on demand | opt-in `capacity_scheduler_policy` | **Yes** |

Two clusters: admit-only (TGI, TRT-LLM default) and evict-capable (vLLM, SGLang, TRT-LLM opt-in).
The rest of this post is about what "evict-capable" actually looks like when you make it happen,
using vLLM as the specimen because it is the one where eviction is on by default and instrumented
with a counter.

## What vLLM actually does, in mechanism

vLLM's recovery mode is called `RECOMPUTE`, and the V1 engine made it the only one that matters:
the older path that swapped a preempted request's KV cache out to host memory was dropped in V1,
because moving gigabytes across the PCIe bus to save a recompute was rarely the better trade. So
preemption in today's vLLM means exactly one thing, and the source spells it out. When the
scheduler preempts a request, it frees that request's KV blocks back to the pool and resets its
progress counter to zero, then puts it back on the waiting queue. On resume, the request
re-runs its prefill from the beginning — every token it had already processed is computed again.
No state is saved; the recompute *is* the recovery.

The reason preemption is possible at all — the reason vLLM can find itself over-committed — is a
single, deliberate scheduling choice: **admission reserves cache for the prompt only, never for
the output.** When a request is admitted, the scheduler allocates blocks for the tokens it needs
to process right now, which at admission time is just the prompt. It does not set aside room for
the hundreds of tokens the request will generate. That is what lets vLLM pack the GPU so
densely, and it is also exactly why a batch that fit comfortably at admission can outgrow the
pool a hundred decode steps later, when every admitted request has grown its KV by a hundred
tokens and collectively they no longer fit. When that happens — an active decode step needs one
more block and the free pool is empty — the scheduler preempts. Admit for the prompt, grow
through decode, evict when growth outruns the pool.

That mechanism makes a sharp prediction about *when* preemption should be worst: it should peak
when admission is cheap but decode growth is large. Short prompts let many requests through the
admission gate cheaply; long outputs then grow all of them until the pool goes dry. Hold that
prediction; the positive control below is built to test it.

## It fires — on every workload, not just the pathological one

I flooded vLLM with 200 concurrent requests at unbounded request rate (so the queue is always
full and the cache is always the binding constraint), across two models and a range of cache
sizes, and counted preemptions. The counting method matters and is its own story — see the
sidebar — but the counter is `vllm:num_preemptions_total`, read from the server's Prometheus
endpoint before and after each flood.

| Arm | Model / attention | Cache (max concurrency) | Workload (in/out) | Preemptions |
|---|---|---|---|---|
| Roomy 4B | 4B, hybrid | natural, 83.7x | 1024/512 fixed | **17** |
| 7B natural, fixed | 7B, dense | natural, 45.2x | 1024/512 fixed | **60** |
| 7B natural, variable | 7B, dense | natural, 45.2x | 512/512 variable | **101** |
| 4B tight | 4B, hybrid | 40-block, 5.7x | 512/512 variable | **41** |
| 7B tight | 7B, dense | 300-block, 2.3x | 512/512 variable | **173** |

Every arm preempts. Not just the deliberately starved ones — even the roomiest cache, a 4B model
with enough KV to seat 83 concurrent requests, preempted 17 times. Preemption is not a
pathological-only event you have to engineer; a flood at unbounded rate transiently over-commits
even when the average headroom looks generous.

And the counts move exactly the way the mechanism says they should. **Tighter cache, more
preemptions:** the dense 7B model under the same variable workload preempts 101 times with its
natural cache and 173 times when I cap the cache to seat barely two requests. Squeeze the pool
and the recompute path works harder. **Higher output-length variance, more preemptions:** the
same 7B model with the same natural cache preempts 60 times under fixed-length outputs and 101
times under variable-length ones. Variance is what lets an admitted request outgrow the batch the
scheduler had sized for it, which is the preemption trigger, so adding variance adds preemptions.

The positive control nails the mechanism down. On a single fixed tight-cache server, varying only
the shape of the workload:

| Workload | in / out | Admission vs. decode growth | Preemptions |
|---|---|---|---|
| Cheap admit, huge growth | 64 / 2000 | many short prompts admitted, each grows 2000 tokens | **205** |
| Balanced | 512 / 512 | admission cost ≈ final footprint | **36** |
| Expensive admit, tiny growth | 2000 / 64 | long prompts gate the batch small at entry | **19** |

This is the prompt-only admission rule made visible. Short prompts with long outputs are the
worst case by a wide margin — 205 preemptions — because admission waves them all in and decode
grows them all past the pool. Reverse it, long prompts and short outputs, and admission itself
holds the running set small, leaving little growth to preempt over. That the balanced 512/512
case still preempts 36 times, and even the long-prompt case 19, is the point restated: because
admission never reserves for output, a burst of arrivals can transiently over-commit under almost
any shape.

## Two regimes: onset transient vs. steady-state churn

Counting total preemptions tells you *how much* but not *when*, and the *when* turns out to
separate the roomy case from the tight case into two genuinely different behaviors. Sampling the
metrics every three seconds through a flood draws the picture.

On the roomy 4B cache, preemptions arrive in **two bursts at the start** and then stop cold. The
count climbs from 0 to 11 in the opening seconds as the running set overshoots to 104 requests at
flood onset, ticks up to 17 as the queue drains into steady state, and then holds flat at 17 for
the entire rest of the run while the running set sits steady around 97. The preemption here is a
*load-onset rebalancing transient*: the scheduler over-commits in the first moments of the flood,
sheds a few requests to recover, and once it settles it never preempts again.

On the tight 7B cache, the count climbs **monotonically the whole run** — 2, 6, 12, 25, 40, and
on up to 181 — while the running set stays pinned at three to seven requests. There is no
settling. The cache is small enough that the server is permanently over-committed relative to
demand, so it evicts and recomputes continuously as a steady-state cost of operating past its
comfortable capacity. Same counter, same mechanism, two different regimes: a transient you pay
once at load onset, and a churn you pay every second you run the cache too tight.

## What it costs, and where to sit on the spectrum

The good news in all of this is what did *not* happen: nothing crashed. Across every arm, every
one of the 200 requests completed. That is the graceful degradation the spectrum is supposed to
buy, and vLLM delivers it — the eviction backstop engages exactly when the naive server would
have run out of memory, and it keeps the server up. The bad news is that the work has to go
somewhere, and it goes into latency.

The cost lands as the cache tightens, and it lands hard. Median time-to-first-token climbs from
about 12 seconds on the dense model with its natural cache, to 2.7 minutes when the cache is
capped tight on the hybrid model, to 5.7 minutes on the tightest dense arm — and the 99th
percentile end-to-end latency on that tightest arm is **12.9 minutes** for a request that would
complete in seconds on an unloaded server. That is recompute work plus queueing, not a crash, but
for an interactive workload it is indistinguishable from an outage.

One honest limit on what I could measure. vLLM's token counters cannot quantify the recompute
*cost* directly. The natural instinct is to look for wasted work — total prompt tokens processed
minus the tokens actually requested — but that number comes back as exactly zero on every arm,
even ones that preempted 181 times. It is not that recompute is free; it is that the counter is
structurally blind to it. vLLM's prompt-token counter sums the *logical* prompt lengths, which
don't change when a request is re-prefilled, and its one "recomputed tokens" counter tracks a
narrow prefix-cache optimization that stays at zero when there is no prefix reuse. No counter in
this version accumulates the physical re-prefill work that preemption causes. So the recompute
cost is real — the source confirms every preempted request re-runs its prefill from scratch — but
it is visible only indirectly, as the inflated latency above and depressed throughput, never as a
clean "percent of compute wasted" figure. I would rather say that plainly than invent a number.

So: where should you sit on the spectrum? If you run vLLM, you are on the evict-capable end by
default, and the practical takeaway is to size the KV cache around the concurrency you actually
need. Under-provision it and the server won't fall over — it will preempt, recompute, and inflate
your tail latency into the minutes, quietly, while every request still eventually succeeds. That
failure mode is gentler than a crash and far harder to notice, which is its own kind of hazard: a
starved cache doesn't page you, it just makes everything slow. If your workload cannot tolerate
that tail, either give the cache more room or pick an engine that sits on the admission-control
end and makes the backpressure explicit as a queue rather than implicit as latency.

## Sidebar: the counter that lies, and the one that doesn't

This post nearly reported the opposite conclusion, and the reason is a measurement bug worth
warning about. The obvious way to count preemptions in vLLM is to grep the server log for the
line it prints when it preempts — a clause that reads `Preemptions: N`. I did that first, and it
reported **zero preemptions on every arm**, which is what led an earlier draft to conclude that
vLLM "stays in admission control and never evicts." That conclusion was wrong, and it was wrong
for a purely mechanical reason.

That log clause is dead code in the version I ran. The logging routine resets its running
statistics — including the preemption count — to zero *before* it reaches the check that decides
whether to print the `Preemptions:` clause. The check is `if preemptions > 0`, and by the time
it runs, the count has already been zeroed, so the branch is never taken. Unlike the throughput
stats, which are copied aside before the reset, the preemption count is simply lost. The clause
can never print. Every "zero" it gave me was a false negative from a line of code that cannot
fire.

The fix is to read the Prometheus counter `vllm:num_preemptions_total` from the `/metrics`
endpoint instead, which is incremented at the source and survives. Every number in this post is
the delta of that counter across a flood. The general lesson is the same one Part 8 arrived at
from the other direction: **a measurement that reads zero is not evidence of absence until you
have confirmed the instrument can produce a non-zero.** A positive control — deliberately forcing
the behavior and checking the counter moves — is not a formality; here it is the entire
difference between the right conclusion and its exact opposite.

## Reproduce

All runs are vLLM 0.18.0 on a single NVIDIA A10G (23 GB), `--dtype float16`, `--max-model-len
2048`, `--max-num-seqs 256`, `--gpu-memory-utilization 0.9`, driven by `vllm bench serve` with
`--dataset-name random --request-rate inf --ignore-eos --num-prompts 200 --seed 0`. Fixed-length
arms use `--random-range-ratio 0.0`; variable arms use `0.9`. Tight caches are set with
`--num-gpu-blocks-override` (40 blocks for the hybrid 4B, 300 for the dense 7B). Models are
`Qwen/Qwen2.5-7B-Instruct` (dense) and `Qwen/Qwen3.5-4B` (hybrid attention). Preemptions are the
delta of `vllm:num_preemptions_total` scraped from `curl localhost:8000/metrics` before and after
each flood — never the server-log clause, which cannot print in this version. The positive
control, the three-second metric timelines behind the two-regime finding, and the full logs live
in the companion repo under `experiments/`.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
