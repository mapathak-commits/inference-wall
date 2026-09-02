# Benchmarks 03 — the batching cliff

Raw measurement logs behind [Part 3](../../articles/part-3.md). Nothing here is
post-processed; the numbers in the post are transcribed from these files.

| File | Produced by | Backs |
|---|---|---|
| `batching_sweep.log` | `run_batching_sweep.sh` | the whole sweep table: 49 tok/s at cap 1 to 1,091 at cap 256, the ~22x cliff, the 61-second cap-1 TTFT |
| `batch_server_{1,4,16,64,256}.log` | one server per cap | per-cap configs and the `Running:` counts behind the batch-~150 plateau claim |

## Pulling a cited number straight from a log

```bash
grep -A12 "max_num_seqs=1 " batching_sweep.log | grep -E "throughput|TTFT"
```
