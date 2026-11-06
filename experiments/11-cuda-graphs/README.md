# Experiments 11: CUDA graphs and the cost of launching work

Scripts behind [Part 11](../../articles/part-11.md). One NVIDIA A10G, vLLM 0.18.0, three models:
`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, and the series' `Qwen/Qwen3.5-4B`, all
served with `--dtype float16 --max-model-len 2048 --max-num-seqs 256 --gpu-memory-utilization 0.9`.

Three arms per model:

| arm | server flag | compilation | CUDA graphs |
|---|---|---|---|
| `graph` | none, the vLLM default | on | on |
| `compileonly` | `--compilation-config '{"cudagraph_mode":"NONE"}'` | on | off |
| `eager` | `--enforce-eager` | off | off |

The post's "graphs off" is `compileonly`, so the comparison changes only the graphs.

| File | What it does |
|---|---|
| `run_study.sh` | the `graph` and `eager` arms: starts a server per model and arm, discards a warm-up pass, runs `vllm bench serve` at 1, 4, 16, and 64 concurrent requests with 128-token prompts and 256 output tokens, adds the series' 200-request flood for the 4B, then restarts each 0.5B and 4B server with the torch profiler armed and captures a trace. |
| `run_nograph.sh` | the same protocol for the `compileonly` arm. |
| `capture_trace.py` | starts one long streaming decode, waits four seconds, and profiles a one-second window with nothing else running. |
| `tp_gaps.py` | per trace: GPU busy fraction, gaps between consecutive kernels, and CPU-side launch calls, one per kernel without graphs and one per replay with graphs. |
| `tp_per_token.py` | kernels per token and GPU kernel time per token, using a marker kernel that runs once per token, identified from the number of graph replays in the graph-mode trace. |
| `render_figs.py` | renders `fig11a-graphs-vs-concurrency.png` and `fig11b-gpu-busy.png` from the numbers in `benchmarks/11-cuda-graphs`. |

The traces come from separate server runs from the speed measurements, because the profiler
slows the CPU's launching and would otherwise distort the timings. GPU work per token is taken
from the traces and time per token from the unprofiled runs.
