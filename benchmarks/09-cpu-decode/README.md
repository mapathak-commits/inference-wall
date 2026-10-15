# Benchmarks 09: decode on a CPU

Raw measurement outputs behind [Part 9](../../articles/part-9.md). Nothing here is
post-processed; the numbers in the post are transcribed from these files. EPYC slice
(`g5.4xlarge`) files at top level, llama.cpp build `458681e`; M4 Pro files under `mac/`,
build `dfc29b6`.

## EPYC slice and A10G

| File | Produced by | Backs |
|---|---|---|
| `stream_sweep.jsonl` | `stream.c` via `bench_mem.py` | STREAM triad 21.8 / 32.4 / 33.4 / 32.7 / 32.4 GB/s at 1 / 2 / 4 / 8 / 16 threads: the 33 GB/s ceiling, reached by 2 threads. |
| `gemm_sweep.jsonl` | `bench_gemm.py` | prefill GEMM 96 to 655 GFLOP/s scaling with cores; decode GEMV flat at ~15 GFLOP/s. |
| `infer_bench.md` | `run_infer_bench.sh` | the quant ladder x thread sweep: bf16 3.92, Q8_0 6.88, Q4_K_M 9.92 tok/s decode at 8 threads; prefill 39.8 / 31.5 / 42.1; the 16-thread collapse to 0.21. |
| `decode_bandwidth_analysis.md` | hand analysis of `infer_bench.md` | the effective-bandwidth ledger 34.0 / 31.8 / 29.9 GB/s. Its batching paragraph is superseded by `batched_bench.md`. |
| `opt_sweep.md` | `opt_sweep.sh` | decode pinned at 9.45 to 9.96 tok/s under every knob; prefill 47.5 pinned vs 40.3 default. |
| `closeout_ab.md` | `closeout.sh` | bf16 4.02 / 3.99 vs F16 4.04 / 4.03 tok/s on byte-identical 8,665,620,192-byte files (1.005x); A10G via llama.cpp CUDA: F16 54.58 tok/s, Q4_K_M 121.34 tok/s. |
| `gpu_baselines.md` | transcribed from Parts 1 and 3 | vLLM reference points (49 single-stream, 1,091 batched tok/s). Different engine; kept for reference only, the batching comparison in the post uses `batched_bench_gpu.md`. |
| `vllm_cpu_smoke.log` | `vllm_cpu_smoke.py` | vLLM 0.18 CPU backend cannot start Qwen3.5-4B (Gated-DeltaNet kernels are GPU only). |
| `batched_bench.md` | `run_batched.sh` | batched decode on the EPYC slice, 1 to 32 concurrent sequences: bf16 3.53 to 26.28 tok/s, Q4_K_M 9.17 to 26.13. Both weight sizes converge at ~26 tok/s by 32, the CPU's compute ceiling. |
| `mac/AUDIT.md` | extract of mac/FINDINGS.md Stage 9 | the Mac study's own audit ledger: every reran claim, what changed, and the stated open gaps. |
| `batched_bench_gpu.md` | `run_batched_gpu.sh` | the same tool on the A10G via CUDA, 1 to 128 sequences: F16 53.6 to 552.5 tok/s, Q4_K_M 117.2 to 885.2. Same-engine batched gap: 15x at one sequence, 21x (F16) and 34x (Q4) at 128 against the CPU's 32-sequence ceiling. |

## M4 Pro (`mac/`)

| File | Produced by | Backs |
|---|---|---|
| `stream_rerun.jsonl` | `stream_triad.c` | read bandwidth 83 GB/s on one P-core, 226 at 4 threads, 238.2 peak at 14 (plateau median 233.5); 273 GB/s nominal. |
| `gemm.jsonl` | `gemm_bench.c` (Accelerate) | prefill GEMM 3.1 TFLOP/s measured; the 2.7x compute ratio against the GPU's derived 8.08 TFLOP/s. |
| `bench_cpu_*.md`, `bench_metal_*.md` | `llama-bench` | the Stage 3 matrix: CPU thread sweep and Metal decode (Q4 57.49, Q8 43.21, bf16 25.88 tok/s). Stage 3 cells were measured minutes apart and drift up to 20%; cross-engine ratios come from `stage5_summary.txt`, not from here. |
| `s4_threads_cpu.md` | `llama-bench` | decode saturating by 4 to 5 threads while prefill scales to 9 P-cores. |
| `stage5_summary.txt` | `run_stage5_robust.sh` | the headline: 12 interleaved CPU/Metal pairs, median 1.089x (0.969 to 1.277); the roofline fits (1/slope 209.2 GB/s CPU, 204.9 Metal); bf16 vs F16 3.92 vs 20.45 tok/s (5.22x). |
| `stage4b_clean/t6_*.md` | `run_stage4b_clean.sh` | the partial-offload sweep, `-t 6` arm (the only arm whose `-ngl 0` drift was under 2%): decode 0.80 to 0.87x of all-CPU for `-ngl` 4 to 16, break-even near 20 layers. |
| `stage6/*.md`, `stage6b/*.md` | `llama-bench -d` | long-context decode: KV q8_0 vs f16 at depth 0 to 32k; crossover at ~16k. Not used in the post. |
| `powermetrics_summary.run1.json`, `run2.json` | `powermetrics` | CPU decode 42.4 / 43.4 W marginal vs Metal run 26.0 / 27.3 W (GPU 21.4 to 21.5 W plus ~5 W CPU); 1.78x and 1.73x less energy per token on Metal. |

## Reading the Mac numbers

Absolute tok/s on the laptop drift with temperature over a session, so only
`stage5_summary.txt` ratios (computed within a round) are used for CPU-versus-GPU
comparisons. Stage 3 absolute cells are kept for the thread-scaling shape and the Metal
decode values.
