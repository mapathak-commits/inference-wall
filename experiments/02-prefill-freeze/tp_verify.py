from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace="chunked_prefill_trace.json")

def one(sql):
    for r in tp.query(sql):
        return r
    return None

print("total slices:", one("select count(*) n from slice").n)

print("\n-- kernel families (count, total CUDA ms, avg us) --")
for r in tp.query("""
  select name, count(*) c, sum(dur)/1e6 ms, avg(dur)/1e3 avg_us
  from slice
  where name like '%delta_rule%' or name like '%chunk_fwd%' or name like '%flash%'
  group by name order by c desc limit 10
"""):
    print(f"{r.name[:52]:52s} {r.c:6d} {r.ms:9.1f}ms {r.avg_us:8.1f}us")

# decode-to-decode start-to-start gap distribution
print("\n-- decode kernel gap (fused_recurrent_gated_delta_rule), start-to-start us --")
rows = list(tp.query("""
  select ts from slice
  where name like 'fused_recurrent_gated_delta_rule%'
  order by ts
"""))
ts = [r.ts for r in rows]
gaps = sorted((ts[i+1]-ts[i])/1e3 for i in range(len(ts)-1))
def pct(x,p):
    import math
    i = min(len(x)-1, int(round(p/100*(len(x)-1))))
    return x[i]
print(f"decode kernels: {len(ts)}   p50={pct(gaps,50):.0f}  p90={pct(gaps,90):.0f}  p99={pct(gaps,99):.0f}")

tp.close()
print("PERFETTO_QUERY_OK")
