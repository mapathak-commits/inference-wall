"""Kernel-family census of the spec-decode trace: counts + CUDA time, plus the
decode-vs-chunk path split for the linear-attention layers."""
import json, glob, sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
    "/var/tmp/vllm-study/traces_spec/rank0.*.pt.trace.json")[0]
ev = json.load(open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
cnt, dur = Counter(), Counter()
for e in k:
    n = e["name"].split("(")[0][:64]
    cnt[n] += 1
    dur[n] += e["dur"]
print("total kernels:", len(k))
print(f"{'CUDA ms':>9}  {'calls':>6}  family")
for name, d in dur.most_common(14):
    print(f"{d/1e3:9.1f}  {cnt[name]:6d}  {name}")
print("---")
for probe in ("fused_recurrent", "chunk_gated", "chunk_fwd", "varlen", "flash"):
    tot_c = sum(c for n, c in cnt.items() if probe in n)
    tot_d = sum(d for n, d in dur.items() if probe in n)
    print(f"family~{probe}: {tot_c} calls, {tot_d/1e3:.1f} ms")
