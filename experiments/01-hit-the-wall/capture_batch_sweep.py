"""
Capture a steady-state decode torch-profiler trace at a chosen batch size (number of
concurrent long-lived decoders), with NO arrivals during the profiled window.

Usage: python capture_batch_sweep.py <n_decoders>

Daemon threads so the process exits right after stop_profile (the decoders are abandoned;
we only need the profiled window). The server writes the trace to VLLM_TORCH_PROFILER_DIR.
"""
import urllib.request, json, threading, time, sys

BASE = "http://localhost:8000"
URL = BASE + "/v1/completions"
MODEL = "Qwen/Qwen3.5-4B"
SHORT = "Tell me about the number seven. "
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAXTOK = 6000  # keep every decoder alive well past the profiled window

def post(path):
    req = urllib.request.Request(BASE + path, data=b"", method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status

def stream(prompt, mt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            for _ in r:
                pass
    except Exception:
        pass

for _ in range(N):
    t = threading.Thread(target=stream, args=(SHORT, MAXTOK), daemon=True)
    t.start()

# Let all N reach steady decode (past prefill). Longer settle for bigger batches.
time.sleep(6 if N <= 64 else 9)

print(f"N={N} start_profile:", post("/start_profile"), flush=True)
time.sleep(2)   # pure decode, no arrivals
print(f"N={N} stop_profile:", post("/stop_profile"), flush=True)
time.sleep(1)   # let the dump flush
print(f"=== BATCH {N} CAPTURE DONE ===", flush=True)
