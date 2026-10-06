---
title: "All posts"
permalink: /articles/
---

*One real model, one ordinary GPU: turn a single knob until something breaks, report
the number, and read the trace that explains why. New parts publish weekly.*

| | | |
|---|---|---|
| **Primer** | [How an LLM actually serves a request]({{ '/articles/primer/' | relative_url }}) | the machine the series breaks: weights, KV cache, prefill, decode, batching |
| **Primer 2** | [What actually happens inside an LLM]({{ '/articles/primer-2/' | relative_url }}) | opens the forward-pass black box: token to vector, what a block does, query/key/value, the KV cache, next-token prediction |
| **Part 1** | [Hit the wall]({{ '/articles/part-1/' | relative_url }}) | an 8.6 GB model on a 23 GB GPU tops out at 7 req/s — and the wall isn't memory |
| **Part 2** | [The prefill freeze]({{ '/articles/part-2/' | relative_url }}) | one fat prompt stalls everyone's stream; one scheduler flag cuts the stutter 2.4x |
| **Part 3** | [The batching cliff]({{ '/articles/part-3/' | relative_url }}) | turning batching off drops the server 22x; the win flattens at batch ~64 |
| **Part 4** | [The attention sink]({{ '/articles/part-4/' | relative_url }}) | on a CPU, a deep head pours its whole attention onto the first token, whose state towers 12x over the rest, and why that limits what you can evict from a long context |
| **Part 5** | [Starving the cache]({{ '/articles/part-5/' | relative_url }}) | flood vLLM until the KV cache binds and it preempts on every workload, not just the pathological one; the cost is tail latency you can see coming, not a crash |
| **Part 6** | [Quantization to make a model fit]({{ '/articles/part-6/' | relative_url }}) | a 9B that fp16 can't serve on this GPU is quantized to 4-bit and serves at ~75% of a 4B's speed, because decode moves bytes, not parameters |
| **Part 7** | [Speculative decoding]({{ '/articles/part-7/' | relative_url }}) | a flag that runs single-stream generation 1.8x faster cuts a flooded server's throughput from 1,096 to 473 tok/s and triples time-to-first-token, because speculation multiplies the per-request work batching can't share and eats concurrency on a hybrid model |
| Part 8 | FlashAttention at the scale where it matters | *coming soon* |
| Part 9 | Off the rig: decode on a CPU | *coming soon* |

Every post ends with a reproduce section; the scripts, raw logs, and profiler traces
live in the [companion repo](https://github.com/mapathak-commits/inference-wall)
under `experiments/` and `benchmarks/`, one folder per part.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
