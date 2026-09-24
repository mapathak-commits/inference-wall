# Benchmarks 05 — starving the cache

Raw measurement logs behind [Part 5](../../articles/part-5.md). Nothing here is
post-processed; the numbers in the post are transcribed from these files. Every
preemption count is the delta of `vllm:num_preemptions_total` across a flood.

| File | Produced by | Backs |
|---|---|---|
| `remeasure.log` | `run_counter_remeasure.sh` | the five-arm table: preemptions of 17 (roomy 4B), 60 (7B natural fixed), 41 (4B tight), 173 (7B tight); the median-TTFT / P99-E2E latency figures |
| `crosscheck.log` | `run_recompute_crosscheck.sh` | the fifth arm (7B natural variable = 101 preemptions) and the recompute cross-check |
| `pc_driver.out` | `run_preempt_positive_control.sh` | the positive-control table: 205 (64/2000), 36 (512/512), 19 (2000/64), and `PREEMPT_LOG_LINES 0` confirming the dead log clause |
| `cc_timeline_ROOMY_4B.log` | 3 s metric sampling | the onset-transient regime: preemptions climb to 17 then hold flat while the running set settles ~97 |
| `cc_timeline_TIGHT_7B.log` | 3 s metric sampling | the steady-state-churn regime: preemptions climb monotonically to 181 while the running set stays pinned at 3–7 |

## Pulling a cited number straight from a log

```bash
grep -E "PREEMPT_COUNTER_DELTA|##########" remeasure.log
grep "PREEMPT_COUNTER_DELTA" pc_driver.out
```
