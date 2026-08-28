import json, sys

# True GPU-busy test: over the pure-decode window, take ALL GPU kernels, merge their
# time intervals, and compare the covered time to wall-clock span. That is the real
# "is the GPU saturated / any idle gaps" measurement, unlike the same-kernel-to-same-kernel
# gap (which is just filled with the intervening layers' kernels).

path = sys.argv[1] if len(sys.argv) > 1 else "steady_decode_trace.json"
ev = json.load(open(path))["traceEvents"]
k = [e for e in ev if e.get("cat")=="kernel" and "ts" in e and "dur" in e and e["dur"] > 0]
k.sort(key=lambda e: e["ts"])
print(f"{path}: {len(k)} GPU kernels")

# restrict to the decode-dominated core window: from first to last delta_rule kernel
dr = [e for e in k if e["name"].startswith("fused_recurrent_gated_delta_rule")]
if dr:
    t0 = dr[0]["ts"]; t1 = dr[-1]["ts"] + dr[-1]["dur"]
else:
    t0 = k[0]["ts"]; t1 = k[-1]["ts"] + k[-1]["dur"]
win = [e for e in k if e["ts"] >= t0 and e["ts"] < t1]
span = (t1 - t0)
print(f"window span: {span/1000:.1f} ms, {len(win)} kernels in window")

# merge intervals (kernels on multiple streams can overlap -> union coverage)
intervals = sorted((e["ts"], e["ts"]+e["dur"]) for e in win)
merged_cov = 0
cur_s, cur_e = intervals[0]
holes = []
for s, e in intervals[1:]:
    if s > cur_e:
        merged_cov += cur_e - cur_s
        holes.append(s - cur_e)   # idle hole between kernel activity
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
merged_cov += cur_e - cur_s
busy_frac = merged_cov / span
print(f"GPU busy fraction (union of kernel intervals / span): {busy_frac*100:.1f}%")
holes.sort()
if holes:
    def pct(x,p):
        i=min(len(x)-1,int(round(p/100*(len(x)-1)))); return x[i]
    print(f"idle holes between GPU activity: count={len(holes)} "
          f"p50={pct(holes,50):.0f}us p90={pct(holes,90):.0f}us max={max(holes):.0f}us "
          f"total_idle={sum(holes)/1000:.1f}ms")
else:
    print("no idle holes: GPU kernels are fully back-to-back")

# per-step picture: delta kernels per step and step time
if dr:
    n = len(dr)
    dts = [e["ts"] for e in dr]
    # steps: 24 linear layers per step -> group; estimate step time from delta count
    print(f"\ndecode delta kernels: {n}; if 24 linear layers/step -> ~{n/24:.0f} steps over {span/1000:.0f}ms "
          f"= ~{span/1000/(n/24):.1f} ms/step")
print("OCCUPANCY_OK")
