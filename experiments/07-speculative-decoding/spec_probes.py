#!/usr/bin/env python3
"""
Single-stream + concurrent probes against a running vLLM server, for the Post 6
speculative-decoding study. Prints one PROBE line per measurement (grep-friendly).

Usage: python spec_probes.py <tag>
  tag: label for the server config (e.g. off, ngram-k5) -- goes into every line.

Probes:
  1. single-stream predictable prompt, 160 tok, temp 0, 3 reps -> ms/token
  2. single-stream creative prompt,    160 tok, temp 0, 3 reps -> ms/token
  3. correctness: first 200 chars of the temp-0 completion for each prompt
     (compare across tags; must be identical for the A/B to be honest)
  4. concurrent predictable: 32 simultaneous repeat-prompts, 128 tok each
     -> aggregate output tok/s (does the spec win survive concurrency?)
  5. concurrent creative: same but creative prompts (overhead under load)
"""
import asyncio, json, sys, time
import aiohttp

TAG = sys.argv[1]
URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-4B"

PREDICTABLE = ("Repeat this exact sentence 20 times: "
               "'The quick brown fox jumps over the lazy dog.' "
               "The quick brown fox jumps over the lazy dog. "
               "The quick brown fox jumps over the lazy dog.")
CREATIVE = "Invent a surreal poem about a clockwork octopus dreaming in colors."


async def one(session, prompt, max_tokens):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True}
    t0 = time.perf_counter()
    async with session.post(URL, json=body) as r:
        j = await r.json()
    dt = time.perf_counter() - t0
    n = j["usage"]["completion_tokens"]
    return dt, n, j["choices"][0]["text"]


async def main():
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # warm the exact paths once (not measured)
        await one(s, PREDICTABLE, 160)
        await one(s, CREATIVE, 160)

        for label, prompt in (("predictable", PREDICTABLE), ("creative", CREATIVE)):
            texts = []
            for rep in range(3):
                dt, n, text = await one(s, prompt, 160)
                texts.append(text)
                print(f"PROBE tag={TAG} kind=single label={label} rep={rep} "
                      f"tokens={n} seconds={dt:.3f} ms_per_tok={1000*dt/n:.2f}",
                      flush=True)
            same = all(t == texts[0] for t in texts)
            print(f"PROBE tag={TAG} kind=selfconsistency label={label} "
                  f"identical_reps={same}", flush=True)
            print(f"PROBETEXT tag={TAG} label={label} "
                  f"head={json.dumps(texts[0][:200])}", flush=True)

        for label, prompt in (("predictable", PREDICTABLE), ("creative", CREATIVE)):
            n_conc, max_tokens = 32, 128
            t0 = time.perf_counter()
            results = await asyncio.gather(
                *[one(s, prompt, max_tokens) for _ in range(n_conc)])
            dt = time.perf_counter() - t0
            total_tok = sum(r[1] for r in results)
            print(f"PROBE tag={TAG} kind=concurrent label={label} n={n_conc} "
                  f"total_tokens={total_tok} seconds={dt:.3f} "
                  f"agg_tok_per_s={total_tok/dt:.1f}", flush=True)

    print(f"PROBE tag={TAG} kind=done", flush=True)


asyncio.run(main())
