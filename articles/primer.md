---
title: "A primer: how an LLM actually serves a request"
permalink: /articles/primer/
---

*A primer for "The Inference Wall". Read this before
[Part 1]({{ '/articles/part-1/' | relative_url }}) if the words KV cache, prefill, decode,
or batch are fuzzy. It explains the machine the five posts go on to break; it deliberately
stops before any of their findings.*

The five posts in this series each take a working LLM server, turn one knob until something
breaks, and read why. To follow *why* each break happens, you need a mechanical picture of
what the server is doing between the moment a request arrives and the moment its answer
finishes streaming. That picture is small, it is not math-heavy, and once you have it every
post is a variation on it. This primer builds it once. No benchmarks here, no surprises, just
the machine.

## The model is a pile of weight matrices in memory

A language model is, physically, a large collection of **weight matrices**, fixed numbers
learned during training. For the model this series uses (Qwen3.5-4B), that collection is
**8.6 GB**. When you start the server, those 8.6 GB are loaded once into the GPU's memory and
they stay there, unchanged, for the life of the server.

The GPU has two relevant parts. There is its **memory** (called HBM), which is large, holds
those 8.6 GB comfortably, but is relatively slow to read from. And there are its **compute
cores**, which do the actual multiplying, are extremely fast, but have almost no storage of
their own. This split is the single most important fact in the whole series, so hold onto it:
**the weights live in the slow, roomy memory; the fast cores that use them cannot keep the
weights parked next to themselves.**

![The two-speed memory: HBM holds the weights and KV cache; the narrow weight-stream feeds the fast compute cores]({{ '/assets/diagrams/d1.jpg' | relative_url }})

## Producing one token = streaming all the weights through the cores

"Running the model on a token" means taking that token, represented as a vector of numbers,
and multiplying it through every weight matrix in turn, layer by layer (this model has 32
layers), until numbers come out the other end that tell you the next token. That single sweep
through all the matrices is called a **forward pass**.

Because the cores cannot hold 8.6 GB, doing a forward pass means **streaming** all 8.6 GB of
weights out of HBM and through the cores. The multiplying itself is quick; the *moving* of
those bytes is the slow part. You will see the series lean on this again and again: the cost
of producing a token is dominated by how many bytes of weights have to be streamed to produce
it, not by the arithmetic done with them.

## Two phases: prefill reads the prompt, decode writes the answer

![Prefill pushes the whole prompt through in one bulk pass; decode loops one token at a time]({{ '/assets/diagrams/d3.jpg' | relative_url }})

Every request runs in two distinct phases, and they behave very differently.

**Prefill** is the first phase: the model reads your whole prompt. Crucially, all the prompt
tokens already exist (you typed them), so they can all be pushed through the forward pass
*together*, in one sweep. A 100-token prompt is one forward pass over 100 tokens. Prefill is
where the model does a lot of arithmetic at once, because every prompt token interacts with
every other (a prompt of length N does roughly N-by-N work as each token looks at all the
others).

**Decode** is the second phase: generating the answer, one token at a time. Here is the
constraint that shapes everything downstream: to produce output token 2, the model needs
output token 1 as input, because a language model predicts each token from the ones before
it. So the tokens of an answer *cannot* be produced together, the way a prompt's tokens can.
Each output token is its **own** forward pass, over just one new token, and each such pass
streams all 8.6 GB of weights again. Prefill amortizes one weight-stream over the whole
prompt; decode is stuck paying one weight-stream per output token. That asymmetry is why
decode, not prefill, is the phase this series spends most of its time on.

## The KV cache: why decode does not reread the whole conversation

If each output token needs "the tokens before it," you might think every decode step reprocesses
the entire conversation so far. It does not, and the thing that saves it is the **KV cache**.

When the model processes a token, part of its work produces two vectors for that token, a
**key** and a **value** (K and V), which together are how later tokens will "look back" at this
one. The KV cache simply *stores* those K and V vectors for every token the model has already
seen. So when the model generates the next token, it does not recompute the past, it looks up
the cached K and V of every earlier token and attends to them.

Two things to keep straight, because they trip people up:

- The KV cache stores **per-token data (K and V vectors)**, not weight matrices. The 8.6 GB of
  weights are one thing; the KV cache is a separate, much smaller pile that grows as the
  conversation grows. On this model a cached token is around 130 KB, so a hundred cached
  tokens is barely ten megabytes, tiny next to 8.6 GB.
- Because of the cache, a decode step feeds the model only each request's **single most recent
  token**, not its whole history. The history is already in the cache; only the newest token
  is new.

The KV cache lives in the GPU's memory alongside the weights, and unlike the weights it grows
with every token of every active request. That makes it the part of memory that can fill up
under load, which is why one of the posts is entirely about starving it.

## Serving many requests at once: one weight-stream, a whole batch

![One shared weight-stream advances the whole batch by one token]({{ '/assets/diagrams/d2.jpg' | relative_url }})

A real server is not answering one request; it is answering many at once. Here is how, and it
is the mechanism the whole series turns on.

The server runs a loop. Each turn of the loop is one **step**: one forward pass, shared by
every request currently being worked on. That set of requests is the **running batch**. A
single step does this:

1. Take each active request's most recent token, one vector per request, and stack them into
   one taller matrix. If ten requests are active, that is ten vectors stacked together.
2. Stream the weight matrices from HBM **once**, and multiply them against that whole stack at
   the same time. A weight matrix multiplied by ten stacked tokens costs the same *streaming*
   as multiplying it by one, because it is the same weight matrix read once; only the
   arithmetic grows, and the arithmetic was the cheap part.
3. Out comes one new token for **every** request in the batch, all produced by that single
   forward pass.

So the expensive thing, streaming 8.6 GB of weights, is **shared across the entire batch in a
single step**. Ten requests get their next token for the price of one weight-stream. This is
called **batching**, and it is the single most important reason one GPU can serve many users
at once. Each request also does a little of its own private work in the step (attending to its
*own* KV cache, which is different from everyone else's), but the big shared cost is the one
weight-stream.

Then the loop repeats. The next step feeds in the tokens just produced, streams the weights
again, and advances every request by one more token. To generate a hundred-token answer takes
about a hundred steps, a hundred weight-streams, each one shared across whatever batch is
running.

Two numbers describe this batch that the posts will refer to. The **arrival rate** is how fast
requests come in (say, 10 per second); it is not the batch size, because each request lives in
the server for a while, so many are in flight at once. And `max_num_seqs` is a configured
**ceiling** on the batch, the most requests the server will run together; the actual running
batch is whatever the load produces, up to that ceiling.

## The knobs the posts will turn

That is the whole machine: weights streamed from HBM per step, a KV cache that lets decode
feed just the latest token, and a batch that shares each weight-stream. Everything the series
does is push on one part of it. Three levers show up by name, so here they are in one line
each:

- **`max_num_seqs`**: the cap on how many requests run in a batch at once.
- **chunked prefill**: instead of letting a long prompt's prefill occupy whole steps by
  itself, slice it and interleave the pieces into steps alongside the decode work.
- **quantization**: store the weights in fewer bits (4 instead of 16), so there are fewer
  bytes to stream per step.

You do not need to know how any of these work internally yet; the posts introduce each where
it matters. What you need is the picture above: **a serving step is one shared stream of the
weights that advances a whole batch of requests by one token, and the KV cache is what lets
each request contribute just its newest token.** With that in hand, Part 1 can ask the
question the series is really about, which is what happens to this machine when you push it
until it breaks.

---

**Next:** [Part 1 — An 8.6 GB model that serves only 7 requests a second]({{ '/articles/part-1/' | relative_url }})

---

*Disclaimer: This blog is written and published in my personal capacity. The opinions,
findings, and conclusions expressed herein are solely my own and do not necessarily
represent the views, policies, or endorsements of my current or past employers.*
