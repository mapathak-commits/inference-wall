"""Step timing from the spec trace: group the per-layer delta-rule update kernels
into engine steps (24 linear-attention layers per step), report step time."""
import json, glob, sys
path = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
    "/var/tmp/vllm-study/traces_spec/rank0.*.pt.trace.json")[0]
ev = json.load(open(path))["traceEvents"]
k = sorted((e for e in ev if e.get("cat") == "kernel" and e.get("dur", 0) > 0),
           key=lambda e: e["ts"])
dr = [e for e in k if e["name"].startswith("fused_sigmoid_gating_delta_rule_update")]
n = len(dr)
steps = n // 24
# first kernel of each step = every 24th delta-rule kernel
firsts = [dr[i * 24]["ts"] for i in range(steps)]
gaps = [(firsts[i + 1] - firsts[i]) / 1e3 for i in range(steps - 1)]
gaps.sort()
p = lambda q: gaps[min(len(gaps) - 1, int(q / 100 * (len(gaps) - 1)))]
print(f"delta-rule update kernels: {n} -> {steps} steps")
print(f"step time ms: p50={p(50):.1f} p90={p(90):.1f} p99={p(99):.1f}")
span = (k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]) / 1e3
# busy fraction
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k)
cov, (cs, ce) = 0, iv[0]
for s, e in iv[1:]:
    if s > ce:
        cov += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
cov += ce - cs
print(f"window {span:.0f} ms, GPU busy {100*cov/ (k[-1]['ts']+k[-1]['dur']-k[0]['ts']):.1f}%")
