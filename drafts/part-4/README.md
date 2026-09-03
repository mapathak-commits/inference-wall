# Starving the cache: the server that refused to break the way I expected

*Draft 1. Part 4 of "The Inference Wall." Same rig: Qwen3.5-4B, fp16,
one NVIDIA A10G (23 GB), under real load.*

---

Every post in this series so far has been about the same bottleneck seen from different
angles: how fast the GPU can generate tokens, the decode loop of Part 1.
Part 1 found the wall, Part 2 watched one big prompt stall everyone else's answer, Part 3
measured what running requests together buys against that wall. This post was supposed to
be about the *other* resource, the one everybody warns you about first when you self-host:
the **KV cache** (the model's short-term memory of the conversation so far) running out of
GPU memory. I set out to starve it and watch the server break.

It broke, but not the way I predicted, and the gap between what I expected and what
happened is the whole lesson. I will tell it in that order, because the wrong prediction is
the useful part.

## The thing I expected: graceful preemption

Recall the KV cache from the primer: the per-token K and V vectors the model stores so it
need not reprocess the whole conversation each step. Unlike the fixed weights, it **grows with
every token of every active request**, so it is the part of GPU memory that expands under load
and, on a busy server, usually the first thing to run out.

vLLM has a documented answer for when it does run out: **preemption.** If the running
requests collectively need more cache than exists, the scheduler *evicts* one, frees its
cache blocks, lets the others proceed, and later *recomputes* the evicted request from
scratch when room opens up. The point of the design is that the server degrades gracefully:
under memory pressure it does not crash or reject requests, it just does some work twice,
so you pay in throughput rather than in errors.

So the plan was simple: starve this 4B's KV cache and put a number on the preemption
penalty. It did not survive contact with the model.

## The problem: on this model, KV is almost impossible to exhaust

Part 1 already hinted at why. This is a **hybrid-attention** model: only 8 of its 32 layers
hold a growing KV cache; the other 24 are linear-attention layers with a fixed-size state.
So per-token KV growth is about a quarter of what a normal transformer of this size would
pay, and on the short 256/128 workload the cache had room for **83 concurrent max-length
requests** and never came close to filling. There is nothing to starve.

To make KV *bind* at all, I had to change the workload and then actively cripple the
server:

1. **Longer sequences.** I switched to a 1024-in / 512-out workload (1,536 tokens per
   request instead of 384), so each request holds a much bigger KV footprint, and flooded
   the server so many run at once.
2. **A hard cap on the cache.** vLLM's `--num-gpu-blocks-override` lets you dictate how many
   KV blocks exist regardless of how much memory is free. I ran three servers: a **roomy**
   one (the full ~77,000-token cache), a mildly starved one (~67,000 tokens), and a
   **brutally starved** one capped at 40 blocks, which on this model's 132-token block size
   is just **5,280 KV tokens, room for only about six of these 1,536-token requests at
   once** (5.71x max concurrency, down from 83.71x). Note it is *six*, not the naive
   5,280 / 1,536 ≈ 3: the same hybrid-KV discount from Part 1 applies, so a request's real
   footprint is smaller than its nominal length, and vLLM's concurrency figure accounts for
   it.

If preemption was ever going to fire, the 40-block server flooded with 1,536-token requests
was the setup to make it happen. It should have been drowning.

## What actually happened: it throttled admission instead

Here is the comparison. Roomy versus the two starved configs, flooded with 200 of the long
requests. (TTFT is time-to-first-token, the wait before the answer starts; E2E is
end-to-end, the total time for the whole answer.)

| Config | KV budget | Achieved req/s | Output tok/s | Median TTFT | P99 E2E | Preemptions logged |
|---|---|---|---|---|---|---|
| **Roomy** | 77,088 tok (83.7x) | 1.85 | 947 | 19,382 ms | 107,914 ms | **0** |
| Starved | 67,584 tok (73.1x) | 1.81 | 928 | 46,121 ms | 110,289 ms | **0** |
| **Severely starved** | 5,280 tok (5.7x) | 0.49 | 253 | 196,836 ms | 404,039 ms | **0** |

Two things fall out, and the second is the whole post.

**First, starvation is real and it is graceful.** The severely-starved server did not
crash, did not reject a single request (all 200 succeeded in every run), and did not error.
It just got dramatically slower: throughput fell **3.7x** (947 → 253 tok/s) and median TTFT
rose **10x** (19 s → 197 s). That is the graceful-degradation headline the post was after,
a memory-starved server slows to a crawl instead of falling over. If you have ever watched
production latency creep up under load with no errors in the logs, this is one of the shapes
it takes.

**Second, and this is the surprise: preemption never fired.** Zero preemption events in
every log, including the brutally-starved one. The mechanism I came to measure did not
happen at all. The server got slow a *completely different way*, and I only understood how
by reading the scheduler's own status lines instead of trusting my assumption.

![Two panels contrasting the failure modes: on the left, preemption yanks a running request out mid-generation and throws away its work to recompute later; on the right, a calm bouncer at a velvet rope admits only as many requests as the cache can seat and holds the rest in a waiting line, never evicting anyone](d7.jpg)

## Reading the scheduler: admission control, not eviction

vLLM prints a status line every few seconds. On the 40-block server, deep into the flood,
they all look like this:

```
Running: 6 reqs, Waiting: 194 reqs, GPU KV cache usage: 92.3%
```

That line is the entire explanation. Of the 200 requests flooding in, the scheduler admits
exactly **6 into the running batch**, parks the other **194 in a waiting queue**, and holds
KV usage at ~92% without ever exceeding it. It is doing **admission control**: it looks at
the tiny cache, calculates that only about six of these 1,536-token requests can fit their
KV at once, and simply *refuses to start the seventh* until one of the six finishes and
frees its blocks. Because it never over-commits the cache, it never has to evict anyone.
Preemption is the recovery mechanism for when you admit too much; if you never admit too
much, you never recover.

![A time series through the 200-request flood on the severely-starved server: the Running line is pinned flat at about 6 requests while the Waiting line falls steadily from 194 to 0 as the queue drains single-file, with zero preemptions logged across every run](fig4-admission-control.png)

So the slowdown is not eviction-and-recompute overhead. It is a **concurrency cap**. On the
roomy server ~100 requests run at once; on the starved one only 6 do, and Part 3 taught us
exactly what a small running batch means: less weight-read amortization and a long serial
queue. The 194 waiting requests drain single-file through 6 slots, which is why TTFT hits
197 seconds, the same queueing physics as Part 3's `max_num_seqs=1` disaster, except here
the small batch is imposed by *KV capacity* rather than by a batch-size flag. **A starved
KV cache degrades into a small-batch server**, and a small-batch server is a slow one.

Why admission control instead of the preemption I expected? Because the scheduler prefers
it: parking a request that has not started costs nothing, while evicting a running request
throws away the work it has already done. vLLM only falls back to preemption when requests
*already admitted* grow their cache mid-flight faster than expected (long generations under
a batch that was sized when they were short). My flood let it size the batch conservatively
from the start, so it never got into the over-committed state preemption exists to rescue.
Preemption is the mechanism when demand *surprises* the scheduler; here demand was
saturating but never surprising.

## What this actually means

The real result is less tidy than the preemption story, and more useful: **under memory
pressure this server does not thrash on eviction, it throttles admission.** The failure mode
is a collapsed running batch, not a storm of preemptions. So the symptom to watch for in the
logs is not "preempted" lines, it is a `Running:` count far below your `max_num_seqs` while a
`Waiting:` queue piles up.

Two caveats on how far this generalizes. First, Qwen3.5-4B is **hybrid-attention** (Part 1);
the **dense** transformers most readers run, such as Llama or Mistral, pay per-token KV on
*every* layer. A dense model of similar size holds far less in the same cache, so it would
hit this admission-control wall much sooner, without the 15x cut it took to pressure this
one. But it should hit the *same* wall:
admission control, not eviction.

Second, and this is the bigger caveat: the workload was fixed-length, which let the scheduler
size the batch safely from the start, exactly the condition (from the section above) under
which it never has to evict. Real traffic, where generation lengths vary and a request can
outrun the batch it was admitted into, is the case that can actually force preemption, and
this experiment engineered that away. So read the headline as scoped to predictable-length
load; a variable-length run is the experiment that would test it, and this was not one.

## What to take away

1. **A memory-starved LLM server degrades gracefully, not catastrophically.** Cutting the
   KV cache 15x slowed the 4B by 3.7x in throughput and 10x in TTFT, with zero errors and
   zero rejected requests. Silent slowdown, not a crash, is the signature of KV pressure.
2. **The mechanism was admission control, not preemption.** The scheduler capped the running
   batch at what the tiny cache could hold (6 requests) and queued the rest, rather than
   over-committing and evicting. Preemption is a recovery path for over-commitment; a
   conservative scheduler that never over-commits never needs it.
3. **A starved cache is really a small-batch problem.** The slowdown is Part 3's cliff in
   disguise: fewer concurrent requests means less weight-read amortization and a longer
   serial queue. KV capacity and batch size are two dials that move the same underlying
   thing, how many requests run at once.
4. **Watch `Running:` vs `Waiting:`, not just "preempted."** On this class of model the tell
   for memory pressure is a running batch stuck far below your configured cap with a growing
   waiting queue, KV usage pinned near 100%. That is the server quietly telling you the cache
   is the bottleneck.
5. **The wrong prediction was the point.** I expected preemption and measured its absence.
   Reporting what the server actually did, instead of the tidier story I planned to tell,
   is where the real finding was, that on a hybrid model KV is hard to exhaust, and when you
   do, it throttles admission rather than thrashing on eviction.

Next, and last, in the series: the finale, where a 9B model that fp16 cannot serve *usefully*
on this GPU (its weights leave too little room for a working KV cache) is quantized to 4-bit,
and serves at ~80% of the 4B's speed, the way you beat a hardware limit instead of just
measuring it.

---

*Reproduce: `run_starvation.sh` (the roomy-vs-starved comparison) and `run_starved_only.sh`
(the severe 40-block arm), with outputs `starvation.log`, `starvation_starved40.log`, and
the per-config `starve_server_*.log` files (whose `Running:`/`Waiting:` status lines back
the admission-control finding), are in the companion repo under
`research/serving/qwen35/`. Single A10G; absolute numbers are rig-specific, the mechanism is
not.*
