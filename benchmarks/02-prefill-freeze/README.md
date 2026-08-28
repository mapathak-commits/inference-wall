# Benchmarks 02 — the prefill freeze

Raw measurement logs and the profiler trace behind [Part 2](../../articles/part-2.md).
Nothing here is post-processed; the numbers in the post are transcribed from these files.

| File | Produced by | Backs |
|---|---|---|
| `probe_cp_on.log` | `chunked_prefill_probe.py` | victim ITL with chunked prefill ON: p50 ~21 ms, p99 345/348 ms |
| `probe_cp_off.log` | `chunked_prefill_probe.py` | victim ITL with chunked prefill OFF: p50 ~21 ms, p99 831/833 ms — the 2.4x headline |
| `server_cp_on.log`, `server_cp_off.log` | `start_server_cp.sh` | the A/B server configs |
| `chunked_prefill_trace.json.gz` | `capture_trace.py` | the GPU timeline: 36,794 kernel events, one ~6k prefill amid 4 decode streams |
| `chunked_prefill_profiler_summary.txt` | `capture_trace.py` | the op table: 408 decode-kernel calls vs 13,905 prefill-kernel calls |

## Reading the trace yourself

No GPU needed. Gunzip and open in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or run the `tp_*.py` analyzers from
`experiments/02-prefill-freeze/` against it for the kernel counts and the
decode-to-decode gap percentiles quoted in the post.
