# Experiments 09: decode on a CPU

Scripts behind [Part 9](../../articles/part-9.md) and [Part 10](../../articles/part-10.md).
Two machines, one engine (`llama.cpp`, driven by `llama-bench`), and three models:
`Qwen/Qwen3.5-4B` as GGUF in bf16, F16, Q8_0, and Q4_K_M for Part 9, plus the official
Qwen2.5-0.5B and 1.5B GGUFs for Part 10's model-size runs:

| machine | what it is | llama.cpp build |
|---|---|---|
| EPYC slice | AWS `g5.4xlarge`: AMD EPYC 7R32 (Zen 2), 8 physical cores / 16 threads, 62 GiB DDR4, one NUMA node, measured 33 GB/s. Holds the series' NVIDIA A10G. | `458681e` |
| M4 Pro | Apple MacBook Pro, M4 Pro, 10 P-cores + 4 E-cores, 48 GB LPDDR5X unified memory, 20-core GPU, measured 238 GB/s. | `dfc29b6` |

The 33 GB/s is the cloud slice's ceiling, not an EPYC socket's (a full 8-channel socket
is ~205 GB/s). `-ngl 0` runs on the CPU, `-ngl 99` on the GPU (CUDA on the A10G, Metal on
the M4 Pro). `pp512` is prefill tok/s, `tg128` decode tok/s.

**Why llama.cpp and not vLLM.** vLLM 0.18's CPU backend has no code path for Qwen3.5's
Gated-DeltaNet linear attention (the kernels are Triton, GPU only); `vllm_cpu_smoke.py`
is the attempt and `benchmarks/09-cpu-decode/vllm_cpu_smoke.log` the failure. llama.cpp
implements it on CPU, CUDA and Metal from one build per machine.

| File | What it does |
|---|---|
| `roofline.py` | EPYC slice and A10G ceilings from the hardware: fp32 GEMM peak, nominal memory bandwidth, the 18x bandwidth and 43x (non-tensor) compute predictions. |
| `stream.c`, `bench_mem.py` | threaded STREAM triad on the EPYC slice, 1 to 16 threads; the 33 GB/s ceiling. |
| `bench_gemm.py` | OpenBLAS sgemm at LLM shapes (M=512 prefill, M=1 decode GEMV), thread sweep. |
| `run_infer_bench.sh`, `bench_infer.sh`, `bench_q4.sh` | the quant ladder x thread sweep with `llama-bench` on the EPYC slice. |
| `opt_sweep.sh` | the CPU tuning sweep on Q4_K_M: core pinning, CCX packing, KV q8_0, flash-attn, poll. |
| `closeout.sh`, `run_closeout.sh` | the byte-identical bf16 vs F16 A/B on the EPYC slice, and the A10G runs through llama.cpp CUDA. |
| `run_small.sh` | downloads the official Qwen2.5-0.5B and 1.5B GGUFs and runs `llama-bench` on both engines plus a same-session 4B anchor; the model-size follow-up in `benchmarks/09-cpu-decode/small_model_analysis.md`. The Mac mirror of this run is `mac/PROMPT_small_model.md` in the research directory. |
| `run_batched.sh`, `run_batched_gpu.sh` | `llama-batched-bench` with 128-token prompts and 128 generated tokens per sequence: on the EPYC slice (cores 0-7 pinned, 1 to 32 concurrent sequences, bf16 and Q4_K_M) and on the A10G via CUDA (1 to 128, F16 and Q4_K_M). Same tool and build on both, so the batched comparison is same-engine. The CPU run needs `-np` set to the largest batch and must not share the prompt across sequences (`-pps` fails on this hybrid model). |
| `vllm_cpu_smoke.py`, `build_vllm_cpu.sh` | the vLLM CPU-backend attempt that fails on this model. |
| `render_figs.py` | renders `fig9a-gap-vs-bandwidth.png`, `fig9b-roofline-fit.png`, `fig9c-bf16-vs-f16.png`, `fig9d-batched-decode.png`, and `fig9e-gap-vs-size.png` from the verified numbers. |

**Mac scripts.** The M4 Pro side (`stream_triad.c`, `gemm_bench.c`, `ledger.py`,
`run_stage5_robust.sh`, `run_stage4b_clean.sh`, the `powermetrics` capture) was run
natively on the laptop and its scripts were not copied back; the raw outputs are in
`benchmarks/09-cpu-decode/mac/`. The Stage 5 protocol that produced the headline 1.089x
is: four quants, three rounds, CPU and Metal cells interleaved so each pair shares a
thermal state, ratio computed within each round, median over the twelve pairs.
