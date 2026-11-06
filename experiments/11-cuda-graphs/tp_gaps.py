"""Kernel-gap census for one profiler trace (json or json.gz): GPU busy fraction, the gap
between consecutive GPU kernels, kernel count, and CPU-side launch calls (one per kernel in
eager mode, one per replayed graph with CUDA graphs). Usage: python tp_gaps.py TRACE [TRACE ...]"""
import gzip, json, sys, os, collections

def load(p):
    with (gzip.open if p.endswith(".gz") else open)(p, "rt") as f:
        return json.load(f)["traceEvents"]

for path in sys.argv[1:]:
    ev = load(path)
    k = sorted((e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0), key=lambda e: e["ts"])
    rt = collections.Counter(e["name"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver"))
    if not k:
        print(os.path.basename(path), "no kernels"); continue
    span = k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]
    iv = [(e["ts"], e["ts"] + e["dur"]) for e in k]
    busy, (cs, ce) = 0, iv[0]
    gaps = []
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    gaps.sort()
    q = lambda f: gaps[min(len(gaps) - 1, int(f * (len(gaps) - 1)))] if gaps else 0
    launches = sum(v for n, v in rt.items() if "LaunchKernel" in n or "LaunchCooperative" in n)
    graphs = sum(v for n, v in rt.items() if "GraphLaunch" in n)
    print(f"=== {os.path.basename(os.path.dirname(path))} ===")
    print(f"  window {span/1e3:.0f} ms, GPU busy {100*busy/span:.1f}%, kernels {len(k)}, mean kernel {sum(e['dur'] for e in k)/len(k):.1f} us")
    print(f"  gaps between kernels: n={len(gaps)} total {sum(gaps)/1e3:.1f} ms, median {q(.5):.1f} us, p90 {q(.9):.1f} us, p99 {q(.99):.1f} us")
    print(f"  CPU launch calls: kernel launches {launches}, graph launches {graphs}")
