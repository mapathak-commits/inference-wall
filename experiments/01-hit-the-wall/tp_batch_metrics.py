"""
Given a steady-decode trace .json, report the metrics that test the bytes-vs-compute
ceiling model: running batch size, per-step time, per-token cost, GPU-busy fraction.

Usage: python tp_batch_metrics.py <trace.json> [label]
"""
import json, sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
ev = json.load(open(path))["traceEvents"]
k = [e for e in ev if e.get("cat") == "kernel" and "ts" in e and "dur" in e and e["dur"] > 0]
k.sort(key=lambda e: e["ts"])

# decode kernels: linear-attention recurrence, one group of 24 per decode step
dr = [e for e in k if e["name"].startswith("fused_recurrent_gated_delta_rule")]
n_dr = len(dr)

# window = first to last decode kernel
if dr:
    t0 = dr[0]["ts"]; t1 = dr[-1]["ts"] + dr[-1]["dur"]
else:
    t0 = k[0]["ts"]; t1 = k[-1]["ts"] + k[-1]["dur"]
span = t1 - t0
win = [e for e in k if e["ts"] >= t0 and e["ts"] < t1]

# GPU busy fraction (union of kernel intervals / span)
intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in win)
cov = 0; cs, ce = intervals[0]
for s, e in intervals[1:]:
    if s > ce:
        cov += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
cov += ce - cs
busy = cov / span if span else 0

# steps: 24 linear-attention layers per decode step -> steps = n_dr / 24
steps = n_dr / 24 if n_dr else 0
step_ms = (span / 1000) / steps if steps else 0

# batch size: how many tokens produced per step. Each step emits 1 token per running seq.
# Infer running batch from the elementwise/gather fan-out is unreliable; instead we pass the
# intended N via the trace's request count is not in the trace, so report step-level only and
# let the caller pair it with the intended N. Per-token cost = step_ms / batch (caller divides).
print(f"{label}: decode_kernels={n_dr} span_ms={span/1000:.1f} steps~={steps:.0f} "
      f"step_ms={step_ms:.2f} busy={busy*100:.1f}%")
