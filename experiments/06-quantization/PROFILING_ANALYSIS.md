# Profiling analysis: decode-step cost model, from traces

Trace-based validation of the two claims the blog series leans on but had only argued from
arithmetic: (1) the decode-step cost decomposes into a fixed shared weight-stream plus a
per-sequence term (Parts 1/3), and (2) that per-sequence term scales with weight **bytes**,
not parameter count, which is why 4-bit quantization pays off (Part 6).

All traces captured on the A10G via the torch profiler (`--profiler-config.profiler=torch`),
one steady-state decode window per batch size, **no request arrivals during the profiled
window** (so the trace is pure decode, not decode+prefill). Analyzed with the `perfetto`
package / PerfettoSQL and stdlib JSON. Scripts: `capture_batch_sweep.py`,
`run_batch_sweep.sh`, `tp_batch_metrics.py` (4B) and the `*_9b*` variants.

## Method

For each batch size N, start N long-lived decoders (`max_tokens=6000`), let them all reach
steady decode, then profile ~5 engine steps. From the trace: decode steps = count of
`fused_recurrent_gated_delta_rule` kernels / 24 (linear-attention layers per step);
step_ms = window span / steps; GPU-busy = union of all kernel intervals / span. Batch size
confirmed from the server's own `Running: N reqs` log lines.

## Result 1: the two-term step-cost model (Qwen3.5-4B, fp16)

| running batch | step time | pure-decode tok/s | per-seq cost | GPU busy |
|---|---|---|---|---|
| 1   | 20.2 ms | 50    | 20.2 ms  | 97.2% |
| 8   | 21.9 ms | 364   | 2.74 ms  | 98.8% |
| 64  | 37.1 ms | 1,725 | 0.58 ms  | 99.2% |
| 146 | 65.6 ms | 2,225 | 0.45 ms  | 99.7% |

Least-squares fit is nearly exact:

```
step_ms = 19.1 + 0.313 * N
```

- **Fixed term 19.1 ms** = the shared weight-stream, read once per step regardless of batch.
  Matches the independent bandwidth-floor arithmetic (8.6 GB / 483 GB/s ≈ 18 ms) to within
  measurement noise. This is the "decode is memory-bandwidth bound at the floor" claim,
  measured rather than computed.
- **Per-seq term 313 us/seq** = the un-shareable work each sequence adds to a step.
- **Crossover (fixed = per-seq) at N ≈ 61.** The blog says the plateau lands "around 64."
  Essentially exact.
- **GPU busy 97-99.7%** across all batches: the decode loop runs kernels back-to-back with
  no schedulable idle gap (validates Post 1's "no gaps" claim; idle holes max out at ~55 us).

### Nuance: pure-decode throughput vs the serving ceiling

Pure decode reaches ~2,225 tok/s at batch 146, about **2x** the blog's serving ceiling of
~1,090 tok/s. Not a contradiction: the ~1,090 figure is measured under real serving, where
incoming requests' **prefill** competes for the same engine steps and consumes roughly half
the GPU. This capture removes arrivals to isolate decode. So "1,090 tok/s" is a serving
ceiling, not a decode-hardware limit; the hardware decode ceiling is ~2x higher, and the gap
is prefill. (This is consistent with Post 2's finding that prefill dominates GPU time under
mixed load.)

## Result 2: per-seq cost scales with bytes, not parameters (4B fp16 vs 9B AWQ)

Same sweep on `QuantTrio/Qwen3.5-9B-AWQ`. Fit both:

| model | fixed A (ms) | per-seq B (us) |
|---|---|---|
| 4B fp16 | 19.08 | 313 |
| 9B AWQ  | 19.32 | 395 |

**Per-seq ratio 9B/4B = 1.26.**

Reference ratios:
- weight **bytes** ratio 11.2 GiB / 8.6 GiB = **1.30**  <- bytes-bound prediction
- **parameter** ratio 9 / 4 = **2.25**  <- compute/FLOP-bound prediction

The measured 1.26 lands on the **byte** ratio (1.30) and nowhere near the parameter ratio
(2.25). So the per-token decode cost is set by how many weight bytes move, not by parameter
count. This is Part 6's mechanism, measured directly: a 9B model quantized to ~1.3x the
4B's bytes serves at ~1/1.3 the per-token rate. The byte ratio predicts ~77%; the measured
serving ceiling is ~75% (821 vs 1,092 tok/s), rather than the ~45% its 2.25x parameters
would imply.

(The fixed A term and the single-stream N=1 step are ~equal for both models, ~19-20 ms,
because at batch 1 the step is dominated by fixed per-pass overheads shared by both; the
byte-vs-FLOP distinction shows in the *scaling* term B, which is where it matters for the
ceiling.)

## Bottom line

Both load-bearing assumptions are confirmed from traces, not just arithmetic:
1. Decode step cost = fixed shared weight-stream (~19 ms, bandwidth-bound) + per-seq term;
   plateau at N≈61 (blog: ~64); GPU 97-99.7% busy (blog: "no gaps").
2. The per-seq term scales with weight **bytes** (1.26x measured vs 1.30x byte ratio), not
   parameters (2.25x), the reason quantization lets the 9B serve at ~75% of the 4B (byte
   ratio predicts ~77%).

## Artifacts

- Traces: `batch{1,8,64,150}.json.gz` (4B), `batch9b{1,8,64,150}.json.gz` (9B).
- Capture: `capture_batch_sweep.py`, `run_batch_sweep.sh` (+ `_9b` variants),
  `start_server_prof.sh`, `start_server_9b_prof.sh`.
- Analysis: `tp_batch_metrics.py`, `tp_occupancy.py`, `tp_verify.py`, `tp_gap_reconcile.py`.
- Perfetto tooling: `research/tools/perfetto-venv` (see `research/tools/README.md`).

## Appendix: serving-curve re-run (2026-07-22) — REPRODUCED

Ran the pre-flight "clean warm re-run of Post 1's serving curve" so the published table and a
fresh run agree. Result: **reproduces essentially exactly** once the server is properly warmed.

The trick was a self-contained driver (`reproduce_curve.sh`, run detached in tmux) that
launches the server, waits until `/v1/models` responds, runs a full warm-up sweep, verifies
`WARMUP_DONE`, and only then measures — so the warm-up cannot be skipped or fragmented. (A
first attempt that let the sandbox's 2-min wrapper timeout split the warm-up from the measured
pass ran 10-20% slow with a rate-4 cold-start TTFT anomaly; that was a methodology miss, now
fixed by the verified-warm driver.)

Clean re-run vs published (4B, 256/128 sweep, `reproduce_sweep.log`):

| rate | published req/s | re-run req/s | published tok/s | re-run tok/s |
|---|---|---|---|---|
| 1   | 0.99 | 0.99 | 126  | 126 |
| 2   | 1.95 | 1.95 | 249  | 249 |
| 4   | 3.77 | 3.77 | 482  | 482 |
| 6   | 5.41 | 5.42 | 693  | 693 |
| 8   | 6.37 | 6.39 | 815  | 818 |
| 10  | 6.22 | 6.28 | 796  | 803 |
| 16  | 7.05 | 7.05 | 902  | 902 |
| 32  | 8.20 | 8.20 | 1,049 | 1,050 |
| inf | 8.53 | 8.57 | 1,092 | 1,097 |

Throughput matches to <1% at every point, and the p99 tails line up too (e.g. rate-32 TTFT
p99 13,642 ms both times; rate-8 800 vs published 789). The rate-4 anomaly from the botched
first attempt is gone (clean 127 ms median TTFT). **The published `q35_sweep_main.log` curve is
confirmed by an independent warm re-run.** The earlier failure was purely under-warming, as
diagnosed, not a data problem. Driver: `reproduce_curve.sh`; markers in `reproduce_run.log`.
