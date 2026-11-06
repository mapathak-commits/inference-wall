# Benchmarks 11: CUDA graphs and the cost of launching work

Raw outputs behind [Part 11](../../articles/part-11.md). One A10G, vLLM 0.18.0. Lines tagged
`WARMUP_` in the logs are the discarded warm-up passes.

| File | Produced by | Backs |
|---|---|---|
| `study.log` | `run_study.sh` | `graph` and `eager` arms. Median ms per token at 1 request: 0.5B 3.25 and 14.76, 3B 14.89 and 21.22, 4B 19.98 and 36.84. 4B flood: 1,087.7 tok/s with graphs, 1,047.7 eager. Server ready times. |
| `study_nograph.log` | `run_nograph.sh` | `compileonly` arm, the post's "graphs off". Median ms per token at 1, 4, 16, 64 requests: 0.5B 14.24, 14.56, 14.34, 16.80; 3B 28.81, 28.94, 29.00, 30.79; 4B 29.57, 28.87, 29.59, 41.14. 4B flood 1,082.5 tok/s. |
| `server_<model>_<arm>.log` | the speed-run servers | compilation mode and `cudagraph_mode` per arm, `torch.compile took` (17 to 36 s on a cold cache), `Graph capturing finished in 3 to 6 secs, took 0.41 to 0.62 GiB`, and the captured batch sizes, 1, 2, 4, 8, then every 8 to 256 and every 16 to 512. |
| `server_trace_<model>_<arm>.log` | the profiling servers | the same, from the trace runs, with the compile cache warm. |
| `traces/<model>_<arm>.trace.json.gz` | `capture_trace.py` | one-second single-request decode windows for the 0.5B and 4B in all three arms. `tp_per_token.py`: GPU kernel time per token 3.0 ms on the 0.5B and 19.6 ms on the 4B in every arm; kernels per token 287 to 335 and 513 to 530. `tp_gaps.py`: median gap between kernels about 0.1 microseconds with graphs; 90th-percentile gap above 140
microseconds in the `compileonly` traces, 164 on the 0.5B and 144 on the 4B. Graph-mode 0.5B: 237 replays and 5,214 separate launches, about 22 per token outside the graph. |

## Derived figures in the post

- GPU working share for one request: traced GPU time per token divided by unprofiled ms per token.
  0.5B 3.04 / 3.25 = 94% with graphs and 3.03 / 14.24 = 21% without; 4B 19.58 / 19.98 = 98% and
  19.62 / 29.57 = 66%.
- CPU time per kernel without graphs, where the CPU sets the pace: all of the CPU's per-token work
  divided by kernels, 14.24 ms / 287 = 50 microseconds on the 0.5B and 29.57 ms / 514 = 58 on the 4B.
- Fixed cost left with graphs: 3.25 - 3.04 = 0.21 ms on the 0.5B and 19.98 - 19.58 = 0.40 ms on the
  4B; without graphs 14.24 - 3.03 = 11.2 ms and 29.57 - 19.62 = 9.95 ms.

## Reading a trace

No GPU needed. Gunzip a trace and open it in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or run the analyzers in `experiments/11-cuda-graphs`.
The profiler slows CPU launching, so gaps in the graphs-off traces look longer than they are
without it.
