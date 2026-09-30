#!/usr/bin/env python3
"""S0 de-risk probe: single-stream prefill-time sweep vs prompt length.

Streams the completion so time-to-first-token is pure prefill (fixed decode
overhead excluded). Warm pass per length, median of 3. Prints a table and a
quadratic fit; the exit test is whether the per-token cost bends superlinearly
before the VRAM/context limit.
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
BASE = f"http://localhost:{PORT}/v1"

# Target prompt lengths in tokens (approx; we build them from a repeated filler).
LENGTHS = [256, 1024, 2048, 4096, 8192, 16384, 24576, 30720]
REPS = 3

def count_tokens(text):
    req = urllib.request.Request(
        f"{BASE}/completions",
        data=json.dumps({"model": MODEL, "prompt": text, "max_tokens": 1,
                         "temperature": 0.0, "echo": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return d["usage"]["prompt_tokens"]

def make_prompt(target_toks):
    # A pseudo-random-ish filler; ~0.75 tokens/word for English-like text.
    filler = ("the quick brown fox jumps over the lazy dog while counting "
              "seven eight nine ten eleven twelve numbers and letters ")
    words = filler.split()
    text = " ".join(words * (target_toks // len(words) + 4))
    # Trim to approx target by binary-ish char cut, then verify.
    approx_chars = int(len(text) * target_toks / max(count_tokens(text), 1))
    return text[:approx_chars]

def stream_ttft(prompt, max_tokens=16):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    last = t0
    itls = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            txt = d.get("choices", [{}])[0].get("text", "")
            if txt:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                ntok += 1
    return ttft, ntok, itls

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])

def main():
    print(f"# S0 prefill sweep  model={MODEL} port={PORT}", flush=True)
    print(f"# {'target':>7} {'actual_tok':>10} {'ttft_s':>9} {'per_tok_ms':>11} "
          f"{'decode_ms':>10}", flush=True)
    rows = []
    for L in LENGTHS:
        try:
            prompt = make_prompt(L)
            actual = count_tokens(prompt)
            # warm
            stream_ttft(prompt)
            samples = []
            dec = []
            for _ in range(REPS):
                ttft, ntok, itls = stream_ttft(prompt, max_tokens=16)
                samples.append(ttft)
                if itls:
                    dec.append(median(itls))
            ttft = median(samples)
            decode_ms = 1000*median(dec) if dec else float("nan")
            per_tok_ms = 1000*ttft/actual
            rows.append((actual, ttft))
            print(f"  {L:>7} {actual:>10} {ttft:>9.3f} {per_tok_ms:>11.3f} "
                  f"{decode_ms:>10.2f}", flush=True)
        except Exception as e:
            print(f"  {L:>7} ERROR {type(e).__name__}: {e}", flush=True)
            break
    # quadratic fit t = a + b n + c n^2 via least squares (numpy-free)
    if len(rows) >= 3:
        n = [r[0] for r in rows]; t = [r[1] for r in rows]
        # build normal equations for [1, n, n^2]
        import itertools
        X = [[1.0, ni, ni*ni] for ni in n]
        # X^T X (3x3), X^T t (3)
        XtX = [[sum(X[k][i]*X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
        Xtt = [sum(X[k][i]*t[k] for k in range(len(X))) for i in range(3)]
        # solve 3x3 by Gaussian elimination
        A = [row[:]+[Xtt[i]] for i,row in enumerate(XtX)]
        for i in range(3):
            p = A[i][i]
            for j in range(i,4): A[i][j] /= p
            for k in range(3):
                if k!=i:
                    f=A[k][i]
                    for j in range(i,4): A[k][j]-=f*A[i][j]
        a,b,c = A[0][3],A[1][3],A[2][3]
        print(f"# fit: t = {a:.4f} + {b*1e6:.1f}us*n + {c*1e12:.1f}ps*n^2", flush=True)
        for probe in (2048, 8192, 32768):
            lin = b*probe; quad = c*probe*probe
            share = quad/(lin+quad) if (lin+quad)>0 else 0
            print(f"#   at n={probe:>5}: quad share of marginal = {share*100:.1f}%", flush=True)

if __name__ == "__main__":
    main()
