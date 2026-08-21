# Benchmarks 01 — hit the wall

Raw measurement logs and profiler traces behind [Part 1](../../articles/part-1.md). Nothing
here is post-processed; the numbers in the post are transcribed from these files.

| File | Produced by | Backs |
|---|---|---|
| `q35_sweep_main.log` | `run_sweep.sh` (200-prompt warm sweep) | the rate-vs-latency knee table |
| `q35_fill400.log` | `run_sweep.sh` (400-prompt knee-fill) | the sharper knee between rate 6 and 8 |
| `q35_sweep_warm.log` | `run_sweep.sh` (warm re-run) | the warm-vs-cold 200x TTFT trap |
| `verify.log` | `verify_load.py` | the memory split: 8.61 GiB weights / 9.45 GiB KV / 77,088 tokens / 83.71x concurrency |
| `traces/batch{1,8,64,150}.json.gz` | `capture_batch_sweep.py` | the step-cost fit (19.1 ms + 0.31 ms/seq), the batch-61 crossover, GPU 97 to 99.7% busy |

## Reading a trace yourself

No GPU needed. Gunzip and open in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or query it with the
[`perfetto`](https://pypi.org/project/perfetto/) Python package via
`experiments/01-hit-the-wall/tp_batch_metrics.py`.

## Pulling a cited number straight from a log

The memory split, exactly as quoted in the post:

```bash
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency" verify.log
```
