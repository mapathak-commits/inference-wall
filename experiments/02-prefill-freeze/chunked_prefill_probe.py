"""
Chunked-prefill probe (online). Question: when a big long-prompt request arrives
while short requests are mid-decode, does it stall their token stream? Chunked
prefill slices the long prefill so decodes keep flowing; without it, the monolithic
prefill monopolizes a step and freezes the decoders.

Design: run several short "victim" streams continuously and measure their
inter-token latency (ITL). Midway, inject a few long-prompt requests. Compare the
victims' ITL during the injection window, server with chunked prefill ON vs OFF.
Run this script against a server started each way; it prints the victims' ITL
percentiles so the two runs can be compared.

Usage: python chunked_prefill_probe.py <label>
"""
import sys, time, threading, statistics, urllib.request, json

LABEL = sys.argv[1]
URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen3.5-4B"

def stream_request(prompt, max_tokens, record_itl=None):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    last = time.time()
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.strip()
            if not line or not line.startswith(b"data:"): continue
            if line == b"data: [DONE]": break
            now = time.time()
            if record_itl is not None:
                record_itl.append((now - last) * 1000)  # ms since previous token
            last = now

SHORT = "Tell me about the number seven. "
LONG = "Summarize the following document. " + ("context sentence. " * 1800)  # ~6k tokens, fits max_model_len 8192

victim_itls = []
stop = threading.Event()

def victim_loop():
    # continuously fire short streaming requests, recording their ITLs
    while not stop.is_set():
        stream_request(SHORT, 64, record_itl=victim_itls)

# start several concurrent victim streams
threads = [threading.Thread(target=victim_loop) for _ in range(6)]
for t in threads: t.start()
time.sleep(3)  # let them reach steady state

# injection window: fire long-prompt requests that must be prefilled.
# injection count is arg2 (default 12 = the "heavy" level); pass 3 for a "light" level.
N_INJ = int(sys.argv[2]) if len(sys.argv) > 2 else 12
inj = [threading.Thread(target=stream_request, args=(LONG, 32)) for _ in range(N_INJ)]
mark = len(victim_itls)
for t in inj: t.start()
for t in inj: t.join()
after = len(victim_itls)

time.sleep(2)
stop.set()
for t in threads: t.join()

# ITLs recorded during the injection window
window = victim_itls[mark:after] if after > mark else []
allitl = [x for x in victim_itls if x < 10000]  # drop absurd outliers
def pct(xs, p):
    return statistics.quantiles(xs, n=100)[p-1] if len(xs) > 2 else float("nan")
print(f"{LABEL} victim_tokens={len(allitl)} inj_window_tokens={len(window)}", flush=True)
print(f"{LABEL} ITL_ms p50={pct(allitl,50):.1f} p99={pct(allitl,99):.1f} max={max(allitl):.1f}", flush=True)
if window:
    print(f"{LABEL} INJ_WINDOW ITL_ms p50={pct(window,50):.1f} p99={pct(window,99):.1f} max={max(window):.1f}", flush=True)
print("=== DONE ===", flush=True)
