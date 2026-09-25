# Benchmarks 06 — quantization as a fit-enabler

Raw measurement logs behind [Part 6](../../articles/part-6.md). Nothing here is
post-processed; the numbers in the post are transcribed from these files.

| File | Produced by | Backs |
|---|---|---|
| `q35_9b_server.log` | `start_server_9b_prof.sh` | the memory table: 11.21 GiB weights, 6.65 GiB KV, 54,384 tokens, 59.0x concurrency, `quantization=awq_marlin` |
| `q35_9b_sweep.log` | `run_sweep_9b.sh` | the 9B serving curve: 6.42 req/s and 821 tok/s at the ceiling |
| `q35_9b_warm.log` | `run_sweep_9b.sh` (warm rows) | the warm low-rate points |
| `traces/batch9b{1,8,64,150}.json.gz` | `run_batch_sweep_9b.sh` | the 9B per-sequence decode fit (395 us/seq) |

The 4B side of every comparison (8.61 GiB weights / 77,088 tokens / 83.7x; the
`q35_sweep_main.log` serving curve topping out at 8.53 req/s and 1,092 tok/s; the
313 us/seq decode fit) is Part 1's, in
[`../01-hit-the-wall/`](../01-hit-the-wall/). The `~75%` serving ratio and the
`1.26x` per-seq ratio are read across the two parts.

## Pulling a cited number straight from a log

```bash
grep -iE "Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency" q35_9b_server.log
```

The captured traces need no GPU: gunzip and open in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev).
