"""Profile one long-running single-request decode: start one streaming request, wait until it is
well into decode, then capture a short profiler window with nothing else running."""
import json, sys, threading, time, urllib.request
BASE = "http://localhost:8000"; MODEL = sys.argv[1]

def post(path):
    with urllib.request.urlopen(urllib.request.Request(BASE + path, data=b"", method="POST")) as r:
        return r.status

def stream():
    body = json.dumps({"model": MODEL, "prompt": "Write a long story about a lighthouse keeper.",
                       "max_tokens": 1500, "temperature": 0.0, "ignore_eos": True, "stream": True}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", data=body,
                                    headers={"Content-Type": "application/json"})) as r:
            for _ in r:
                pass
    except Exception:
        pass

threading.Thread(target=stream, daemon=True).start()
time.sleep(4)
print("start_profile:", post("/start_profile"), flush=True)
time.sleep(1.0)
print("stop_profile:", post("/stop_profile"), flush=True)
print("=== TRACE CAPTURE DONE", MODEL, "===", flush=True)
