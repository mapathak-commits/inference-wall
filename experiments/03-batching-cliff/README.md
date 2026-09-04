# Experiment 03 — the batching cliff

Script behind [Part 3](../../articles/part-3.md): the `max_num_seqs` sweep that turns
batching off and back on.

| Script | What it does | Output (in `benchmarks/03-batching-cliff/`) |
|---|---|---|
| `run_batching_sweep.sh` | Launches one server per cap (1, 4, 16, 64, 256) and floods each with the fixed 256-in/128-out workload | `batching_sweep.log`, per-cap `batch_server_*.log` |

The shared server-launch script is in [`../../scripts/`](../../scripts/).
