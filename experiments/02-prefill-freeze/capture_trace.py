"""
Capture a torch-profiler trace of the chunked-prefill decode-freeze mechanism.

The chunked-prefill study showed (via ITL p99) that a long prefill perturbs the
victim decoders' token stream. This script *shows* that mechanism in a profiler
trace: it drives a few short "victim" decode streams, arms the server-side torch
profiler (/start_profile), injects one ~6k-token long prompt whose prefill must be
scheduled amid the decodes, then stops the profiler (/stop_profile). The resulting
trace (written to the server's torch_profiler_dir) contains the decode steps with the
prefill interleaved — the GPU-timeline view of what the ITL numbers implied.

The profiled window is kept tiny (server auto-limits to active_iterations=5 engine
steps) so the trace .json stays small and chrome://tracing-openable.

Usage: python capture_trace.py
"""
import urllib.request, json, threading, time

BASE = "http://localhost:8000"
URL = BASE + "/v1/completions"
MODEL = "Qwen/Qwen3.5-4B"
SHORT = "Tell me about the number seven. "
LONG = "Summarize the following document. " + ("context sentence. " * 1800)  # ~6k tok

def post(path):
    req = urllib.request.Request(BASE + path, data=b"", method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status

def stream(prompt, mt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for _ in r:
            pass

stop = threading.Event()
def victim_loop():
    while not stop.is_set():
        stream(SHORT, 64)

# 4 steady victim decode streams
vs = [threading.Thread(target=victim_loop) for _ in range(4)]
for t in vs: t.start()
time.sleep(3)  # reach steady-state decode

# arm profiler, inject one heavy prefill during the profiled window, stop
print("start_profile:", post("/start_profile"), flush=True)
inj = threading.Thread(target=stream, args=(LONG, 16))
inj.start()
time.sleep(2)          # let the prefill land amid the decodes (profiler caps at 5 steps)
inj.join()
print("stop_profile:", post("/stop_profile"), flush=True)

stop.set()
for t in vs: t.join()
print("=== CAPTURE DONE ===", flush=True)
