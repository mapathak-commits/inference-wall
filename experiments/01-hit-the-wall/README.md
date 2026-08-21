# Experiment 01 — hit the wall

Scripts behind [Part 1](../../articles/part-1.md): the request-rate sweep that finds the
knee, the startup memory report, and the steady-decode profiling that fits the
`19.1 ms + 0.31 ms x batch` step-cost model.

| Script | What it does | Output (in `benchmarks/01-hit-the-wall/`) |
|---|---|---|
| `run_sweep.sh` | Warm request-rate sweep (256-in/128-out, rates 1 to inf) via `vllm bench serve` | `q35_sweep_main.log`, `q35_fill400.log`, `q35_sweep_warm.log` |
| `verify_load.py` | Prints vLLM's startup memory split (weights / KV cache / max concurrency) | `verify.log` |
| `capture_batch_sweep.py` | Holds N steady decoders and profiles ~5 engine steps per batch size | `traces/batch{1,8,64,150}.json.gz` |
| `run_batch_sweep.sh` | Driver for the batch-size trace sweep | — |
| `tp_batch_metrics.py` | Reads step time / per-seq cost / GPU-busy out of a captured trace (PerfettoSQL) | the step-cost fit quoted in the post |

Server launch scripts are shared across experiments and live in
[`../../scripts/`](../../scripts/): `start_server.sh` (the serving config under test) and
`start_server_prof.sh` (same, with the torch profiler armed).

The trace analysis needs no GPU: `pip install perfetto`, then point `tp_batch_metrics.py`
at a gunzipped trace from `benchmarks/01-hit-the-wall/traces/`.
