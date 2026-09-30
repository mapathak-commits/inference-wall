#!/usr/bin/env python3
"""S1 probe: prefill-TTFT + decode-at-depth sweep vs prompt length, one arm.

For each prompt length: stream a 64-token completion. TTFT (first-token time) is
pure prefill; decode ITL is the median inter-token gap over the generated tokens
(first two gaps dropped as warmup). Warm pass per length, median of REPS.

Args: PORT MODEL LABEL
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-7B-Instruct"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "arm"
BASE = f"http://localhost:{PORT}/v1"

LENGTHS = [256, 1024, 2048, 4096, 8192, 16384, 24576, 30720]
REPS = 3
GEN = 64

def count_tokens(text):
    req = urllib.request.Request(
        f"{BASE}/completions",
        data=json.dumps({"model": MODEL, "prompt": text, "max_tokens": 1,
                         "temperature": 0.0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["usage"]["prompt_tokens"]

def make_prompt(target_toks):
    filler = ("the quick brown fox jumps over the lazy dog while counting "
              "seven eight nine ten eleven twelve numbers and letters ")
    words = filler.split()
    text = " ".join(words * (target_toks // len(words) + 4))
    approx = int(len(text) * target_toks / max(count_tokens(text), 1))
    return text[:approx]

def stream(prompt, max_tokens=GEN):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(f"{BASE}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; last = t0; itls = []
    with urllib.request.urlopen(req, timeout=900) as r:
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
    return ttft, itls

def median(xs):
    xs = sorted(xs); n = len(xs)
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1]+xs[n//2])

def qfit(rows):
    n = [r[0] for r in rows]; t = [r[1] for r in rows]
    X = [[1.0, ni, ni*ni] for ni in n]
    XtX = [[sum(X[k][i]*X[k][j] for k in range(len(X))) for j in range(3)] for i in range(3)]
    Xtt = [sum(X[k][i]*t[k] for k in range(len(X))) for i in range(3)]
    A = [row[:]+[Xtt[i]] for i, row in enumerate(XtX)]
    for i in range(3):
        p = A[i][i]
        for j in range(i, 4): A[i][j] /= p
        for k in range(3):
            if k != i:
                f = A[k][i]
                for j in range(i, 4): A[k][j] -= f*A[i][j]
    return A[0][3], A[1][3], A[2][3]

def main():
    print(f"# S1 arm={LABEL}  model={MODEL} port={PORT} gen={GEN}", flush=True)
    print(f"# {'target':>7} {'actual':>7} {'ttft_s':>9} {'pref_ms/tok':>12} "
          f"{'decode_ms':>10}", flush=True)
    rows = []; decrows = []
    for L in LENGTHS:
        try:
            prompt = make_prompt(L)
            actual = count_tokens(prompt)
            stream(prompt)  # warm
            ttfts = []; decs = []
            for _ in range(REPS):
                ttft, itls = stream(prompt)
                if ttft is not None:
                    ttfts.append(ttft)
                if len(itls) > 4:
                    decs.append(median(itls[2:]))  # drop warmup gaps
            ttft = median(ttfts)
            decode_ms = 1000*median(decs) if decs else float("nan")
            rows.append((actual, ttft)); decrows.append((actual, decode_ms))
            print(f"  {L:>7} {actual:>7} {ttft:>9.3f} {1000*ttft/actual:>12.3f} "
                  f"{decode_ms:>10.2f}", flush=True)
        except Exception as e:
            print(f"  {L:>7} ERROR {type(e).__name__}: {e}", flush=True)
            break
    if len(rows) >= 3:
        a, b, c = qfit(rows)
        print(f"# prefill fit: t = {a:.4f} + {b*1e6:.1f}us*n + {c*1e12:.1f}ps*n^2", flush=True)
        for p in (2048, 8192, 32768):
            lin = b*p; quad = c*p*p; sh = quad/(lin+quad) if lin+quad > 0 else 0
            print(f"#   quad share @ n={p:>5}: {sh*100:.1f}%", flush=True)
        d0 = decrows[0][1]; dN = decrows[-1][1]
        if d0 == d0 and dN == dN:  # not nan
            print(f"# decode ITL: {d0:.1f} ms @ {decrows[0][0]} -> {dN:.1f} ms @ "
                  f"{decrows[-1][0]}  (+{100*(dN-d0)/d0:.0f}%)", flush=True)

if __name__ == "__main__":
    main()
