from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace="chunked_prefill_trace.json")

rows = list(tp.query("""
  select ts, dur from slice
  where name like 'fused_recurrent_gated_delta_rule%'
  order by ts
"""))
ts  = [r.ts for r in rows]
dur = [r.dur for r in rows]
n = len(ts)

def pct(x, p):
    x = sorted(x)
    i = min(len(x)-1, int(round(p/100*(len(x)-1))))
    return x[i]

def report(label, vals):
    vals_us = [v/1e3 for v in vals]
    print(f"{label:40s} n={len(vals):4d}  p50={pct(vals_us,50):7.0f}  p90={pct(vals_us,90):7.0f}  p99={pct(vals_us,99):7.0f}")

print(f"total decode kernels: {n}\n")

# hypothesis A: start-to-start, all kernels
s2s = [ts[i+1]-ts[i] for i in range(n-1)]
report("A. start->start, all", s2s)

# hypothesis B: end-to-start (true idle gap), all kernels
e2s = [ts[i+1]-(ts[i]+dur[i]) for i in range(n-1)]
report("B. end->start (idle gap), all", e2s)

# The trace has multiple decode "bursts" separated by big gaps. Identify bursts:
# a gap > 50ms (50e6 ns) between consecutive decode kernels = new burst boundary.
GAP = 50e6
bursts = []
cur = [0]
for i in range(1, n):
    if ts[i]-ts[i-1] > GAP:
        bursts.append(cur); cur = []
    cur.append(i)
bursts.append(cur)
print(f"\ndecode kernels split into {len(bursts)} burst(s); sizes: {[len(b) for b in bursts]}")

# hypothesis C: within-burst start-to-start (drop cross-burst jumps)
s2s_wb = []
e2s_wb = []
for b in bursts:
    for j in range(len(b)-1):
        a, c = b[j], b[j+1]
        s2s_wb.append(ts[c]-ts[a])
        e2s_wb.append(ts[c]-(ts[a]+dur[a]))
report("C. start->start, within-burst", s2s_wb)
report("D. end->start, within-burst", e2s_wb)

tp.close()
print("\nRECONCILE_OK")
