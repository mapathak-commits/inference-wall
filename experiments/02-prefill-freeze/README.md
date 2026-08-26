# Experiment 02 — the prefill freeze

Scripts behind [Part 2](../../articles/part-2.md): the victim-ITL probe that measures
what one fat prompt does to everyone else's token stream, the chunked-prefill A/B, and
the trace capture that shows the freeze kernel by kernel.

| Script | What it does | Output (in `benchmarks/02-prefill-freeze/`) |
|---|---|---|
| `chunked_prefill_probe.py` | Runs 6 streaming victims, injects ~6k-token prompts, records victim ITL | `probe_cp_on.log`, `probe_cp_off.log` |
| `start_server_cp.sh` | Launches the server with chunked prefill on or off | `server_cp_on.log`, `server_cp_off.log` |
| `capture_trace.py` | Torch-profiler capture of one injected prefill amid 4 decode streams | `chunked_prefill_trace.json.gz`, `chunked_prefill_profiler_summary.txt` |
| `tp_verify.py` | Kernel-family counts + decode-to-decode gap percentiles from the trace | the p50/p90/p99 gap numbers in the post |
| `tp_gap_reconcile.py` | The gap-figure methodology: idle time between consecutive decode kernels | — |
| `tp_occupancy.py` | GPU-busy fraction over the profiled window | — |
| `render_heartbeat.py` | Renders the step-cadence figure from the trace (one tick per engine step) | `assets/figures/fig2b-step-heartbeat.png` |

The trace analysis needs no GPU: `pip install perfetto`, gunzip the trace from
`benchmarks/02-prefill-freeze/`, and point the `tp_*.py` scripts at it.
