# Experiment 05 — starving the cache

Scripts behind [Part 5](../../articles/part-5.md): flood vLLM until the KV cache is the
binding constraint, and count preemptions with the authoritative Prometheus counter
(`vllm:num_preemptions_total`) rather than the dead server-log clause.

| Script | What it does | Output (in `benchmarks/05-cache-starvation/`) |
|---|---|---|
| `run_counter_remeasure.sh` | Re-runs all five arms (roomy 4B, natural/tight 7B dense, tight 4B hybrid) counting preemptions from the metrics endpoint | `remeasure.log` |
| `run_preempt_positive_control.sh` | On one fixed tight-cache server, varies only workload shape (64/2000, 512/512, 2000/64) to isolate admission-vs-growth | `pc_driver.out` |
| `run_recompute_crosscheck.sh` | Re-runs the natural-cache 7B arm and samples the metrics every 3 s, cross-checking the counter delta against the timeline | `crosscheck.log` |
| `run_varlen_starvation.sh` | The variable-length flood used for the tight-cache arms | (server/bench logs, summarized in `remeasure.log`) |

Each script launches its own `vllm serve` inline; the counter is scraped with
`curl -s localhost:8000/metrics | grep vllm:num_preemptions` before and after each flood.
The scripts hardcode a model cache and venv path from the original box (`/var/tmp/...`,
`/home/mpathak/...`); grep and point them at your own locations.
