# Benchmarks 07 — Speculative decoding

Raw measurement logs and the profiler trace behind [Part 7](../../articles/part-7.md).
Nothing here is post-processed; the numbers in the post are transcribed from these files.
Single A10G, Qwen3.5-4B, fp16, vLLM 0.18.0.

| File | Produced by | Backs |
|---|---|---|
| `spec_study.log` | `run_spec_study.sh` | the single-stream table (1.79x k5 / 0.92x k3), the 32-concurrent table (predictable +14%/+25%, creative −46%/−24%), and the full rate sweep (flood off/k5/k3 = 1,096 / 473 / 542 tok/s; TTFT 5s vs 22s). Carries the `SpecDecoding metrics` lines: k5 avg acceptance ~21%, per-position 0.32/0.22/0.18/0.15/0.10. |
| `spec_followup.log` | `run_spec_followup.sh` | k5 flood stability re-runs (491, 493 tok/s) against the off baseline (1,097 both times). |
| `server_off.log` | `start_server.sh` (spec off) | startup `Maximum concurrency ... 83.71x`; flooded scheduler `Running: 146 reqs`. |
| `server_ngram-k5.log` | `start_server.sh` (k5) | startup `Maximum concurrency ... 24.64x`; flooded scheduler `Running: 28 reqs`. |
| `server_ngram-k3.log` | `start_server.sh` (k3) | the k3-arm server log for the sweep. |
| `server_fu_off.log`, `server_fu_k5.log` | `start_server.sh` | the follow-up stability servers. |
| `traces/spec_verify_trace.json.gz` | `capture_spec_trace.py` | the verify-step trace: 35,202 GPU kernels over a 2 s window, no arrivals. |

## The trace

Run `experiments/07-speculative-decoding/tp_spec_kernels.py` against the trace for the
kernel census and `tp_spec_steps.py` for the step timing. The headline lines:

- `fused_recurrent_gated_delta_rule` (the one-token decode recurrence) appears **0 times**.
- `fused_sigmoid_gating_delta_rule_update` (the multi-token verify path of the same layer)
  runs **1,320 calls**, which at 24 linear-attention layers per pass is **55 engine steps**.
- Per-step time **p50 36.2 ms** (vs ~20 ms for an ordinary decode step at this batch),
  processing six positions per sequence, so 6 ms per position.
- GPU busy **80.2%** across the window, against 97 to 99.7% in Part 1's pure decode loop.

## Reading the trace yourself

No GPU needed. Gunzip `traces/spec_verify_trace.json.gz` and open in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or run the two analyzers above against it. Each
GPU kernel is a chrome-trace event with a category, a timestamp, and a duration.
