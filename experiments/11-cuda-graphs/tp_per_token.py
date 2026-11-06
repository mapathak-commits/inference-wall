"""Kernels per token in a single-request decode trace. Finds a marker kernel that runs once per
token (its count equals the number of graph replays in the graph-mode trace), then divides each
trace's kernel count by its marker count. Usage: python tp_per_token.py GRAPH_TRACE OTHER_TRACE..."""
import gzip, json, sys, os, collections
def load(p):
    with (gzip.open if p.endswith(".gz") else open)(p, "rt") as f:
        return json.load(f)["traceEvents"]
def kernels(ev): return [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
g = load(sys.argv[1])
glaunch = sum(1 for e in g if e.get("cat") in ("cuda_runtime","cuda_driver") and "GraphLaunch" in e.get("name",""))
gc = collections.Counter(e["name"] for e in kernels(g))
cands = [n for n, c in gc.items() if abs(c - glaunch) <= 1]
print("graph replays in graph trace:", glaunch, "| once-per-token candidates:", len(cands))
for n in cands[:6]: print("   ", gc[n], n[:90])
for p in sys.argv[1:]:
    ev = load(p); k = kernels(ev); c = collections.Counter(e["name"] for e in k)
    marks = [c[n] for n in cands if c[n] > 0]
    if not marks: print(os.path.basename(os.path.dirname(p)), "no shared marker"); continue
    toks = sorted(marks)[len(marks)//2]
    span = (k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]) / 1e3
    print(f"{os.path.basename(os.path.dirname(p)):34s} tokens in window {toks:4d} | kernels/token {len(k)/toks:6.1f} | ms/token under profiler {span/toks:6.2f} | GPU kernel time/token {sum(e['dur'] for e in k)/toks/1e3:5.2f} ms")
