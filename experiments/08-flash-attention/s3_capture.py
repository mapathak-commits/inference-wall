#!/usr/bin/env python3
"""S3: capture one bounded profiler window per (phase, length).

Drives the running server's /start_profile and /stop_profile around a single
controlled request so the trace is a clean, small window. Prefill windows profile
one full prefill of the target length (max_tokens=1). Decode windows first send a
priming request to build the KV context, then profile a short generation on top of
it so the trace is (mostly) steady-state decode.

Args: PORT MODEL LABEL PHASE LENGTH [GENSTEPS]
  PHASE = prefill | decode
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; LABEL = sys.argv[3]
PHASE = sys.argv[4]; LENGTH = int(sys.argv[5])
GENSTEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 30
BASE = f"http://localhost:{PORT}/v1"
ADMIN = f"http://localhost:{PORT}"

def post(path, body=None, base=ADMIN, timeout=900):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def count_tokens(text):
    d = json.loads(post("/v1/completions",
        {"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0.0}))
    return d["usage"]["prompt_tokens"]

def make_prompt(target):
    filler = ("the quick brown fox jumps over the lazy dog while counting "
              "seven eight nine ten eleven twelve numbers and letters ")
    words = filler.split()
    text = " ".join(words * (target // len(words) + 4))
    approx = int(len(text) * target / max(count_tokens(text), 1))
    return text[:approx]

def completion(prompt, max_tokens):
    return post("/v1/completions",
        {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0.0})

def main():
    prompt = make_prompt(LENGTH)
    actual = count_tokens(prompt)
    print(f"# S3 capture arm={LABEL} phase={PHASE} len={actual}", flush=True)
    if PHASE == "prefill":
        # warm the length once (compilation etc.), then profile a single prefill
        completion(prompt, 1)
        post("/start_profile")
        completion(prompt, 1)
        post("/stop_profile")
    elif PHASE == "decode":
        # warm, then profile a short generation on top of the built context
        completion(prompt, 4)
        post("/start_profile")
        completion(prompt, GENSTEPS)
        post("/stop_profile")
    else:
        print("unknown phase", PHASE); sys.exit(2)
    # give the writer a moment to flush the trace
    time.sleep(8)
    print(f"# done {LABEL} {PHASE} {actual}", flush=True)

if __name__ == "__main__":
    main()
