---
title: "All posts"
permalink: /articles/
---

*One real model, one ordinary GPU: turn a single knob until something breaks, report
the number, and read the trace that explains why. New parts publish weekly.*

| | | |
|---|---|---|
| **Primer** | [How an LLM actually serves a request]({{ '/articles/primer/' | relative_url }}) | the machine the series breaks: weights, KV cache, prefill, decode, batching |
| **Part 1** | [Hit the wall]({{ '/articles/part-1/' | relative_url }}) | an 8.6 GB model on a 23 GB GPU tops out at 7 req/s — and the wall isn't memory |
| **Part 2** | [The prefill freeze]({{ '/articles/part-2/' | relative_url }}) | one fat prompt stalls everyone's stream; one scheduler flag cuts the stutter 2.4x |
| **Part 3** | [The batching cliff]({{ '/articles/part-3/' | relative_url }}) | turning batching off drops the server 22x; the win flattens at batch ~64 |
| Part 4 | Starving the cache | *coming Sep 11* |
| Part 5 | Quantization as a fit-enabler | *coming Sep 18* |
| Part 6 | Speculative decoding | *coming Sep 25* |
| Part 7 | FlashAttention at the scale where it matters | *coming Oct 2* |

Every post ends with a reproduce section; the scripts, raw logs, and profiler traces
live in the [companion repo](https://github.com/mapathak-commits/inference-wall)
under `experiments/` and `benchmarks/`, one folder per part.

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
