"""
Capture a torch-profiler trace of steady decode WITH ngram speculation active.

Same discipline as capture_trace_steady.py (fixed set of long-lived decoders, no
arrivals during the profiled window), but the decoders generate PREDICTABLE text
(repeat-a-sentence prompts) so the ngram proposer drafts successfully and the
profiled steps are real verify steps: k+1 tokens per sequence per step through
the decode path. That multi-token decode signature, plus the state-checkpoint
handling, is what the trace is for.

Run against a server started with ngram spec + the torch profiler armed.
Usage: python capture_spec_trace.py
"""
import urllib.request, json, threading, time

BASE = "http://localhost:8000"
URL = BASE + "/v1/completions"
MODEL = "Qwen/Qwen3.5-4B"
PROMPT = ("Repeat this exact sentence forever: "
          "'The quick brown fox jumps over the lazy dog.' "
          "The quick brown fox jumps over the lazy dog. "
          "The quick brown fox jumps over the lazy dog.")
NREQ = 8
MAXTOK = 1500  # long-lived; all 8 still decoding through the profiled window

def post(path):
    req = urllib.request.Request(BASE + path, data=b"", method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status

def stream(prompt, mt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": mt,
                       "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            for _ in r:
                pass
    except Exception:
        pass  # abandoned decoders are fine

threads = [threading.Thread(target=stream, args=(PROMPT, MAXTOK), daemon=True)
           for _ in range(NREQ)]
for t in threads:
    t.start()

time.sleep(6)  # everyone past prefill, mid-verify-decode
print("start_profile:", post("/start_profile"), flush=True)
time.sleep(2)
print("stop_profile:", post("/stop_profile"), flush=True)
print("=== SPEC TRACE CAPTURE DONE ===", flush=True)
