# Experiment 06 — quantization as a fit-enabler

Scripts behind [Part 6](../../articles/part-6.md): standing up `Qwen3.5-9B-AWQ`
(4-bit) on the same A10G as the 4B, sweeping its serving curve, and profiling the
per-sequence decode cost that explains why the bigger model serves at roughly
three-quarters of the 4B's speed.

| Script | What it does | Output (in `benchmarks/06-quantization/`) |
|---|---|---|
| `verify_load.py` | Loads a model offline and prints the weights-vs-KV memory split | `q35_9b_server.log` has the same split at serve time |
| `start_server_9b_prof.sh` | Launches the 9B-AWQ server with the torch profiler enabled | `q35_9b_server.log` |
| `run_sweep_9b.sh` | Request-rate sweep (1→inf) over the fixed 256-in/128-out workload | `q35_9b_sweep.log`, `q35_9b_warm.log` |
| `start_server_9b_prof.sh` + `run_batch_sweep_9b.sh` + `capture_batch_sweep_9b.py` | Steady-state decode traces at batch 1/8/64/150, no arrivals in the window | `traces/batch9b{1,8,64,150}.json.gz` |
| `tp_batch_metrics.py` | Reads a trace and reports step time / per-seq cost | the fit in `PROFILING_ANALYSIS.md` |
| `PROFILING_ANALYSIS.md` | The two-term decode-cost fit for both models, with the per-seq byte-ratio result | — |

The 4B comparison curve and its decode traces are Part 1's, in
[`benchmarks/01-hit-the-wall/`](../../benchmarks/01-hit-the-wall/) (`q35_sweep_main.log`,
`traces/batch{1,8,64,150}.json.gz`); this experiment does not duplicate them.

The shared server-launch scripts are in [`../../scripts/`](../../scripts/).
