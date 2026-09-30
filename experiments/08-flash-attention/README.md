# Experiments 08 — FlashAttention at the scale where it matters

Scripts behind [Part 8](../../articles/part-8.md). Everything ran on a single NVIDIA A10G
(23 GB) serving Qwen2.5-7B-Instruct in fp16 through vLLM, single stream. The attention
backend is selected at server start with the `VLLM_ATTENTION_BACKEND` environment variable
(`FLASH_ATTN` against `FLEX_ATTENTION`); nothing else changes between arms.

| File | What it does |
|---|---|
| `start_server.sh` | launches `vllm serve` for one arm. Args: `MODEL BACKEND MAXLEN PORT`. Exports `VLLM_ATTENTION_BACKEND` and puts `ninja`/`nvcc` on `PATH` for backend JIT. |
| `s0_prefill_sweep.py` | S0 de-risk: single-stream streamed-TTFT prefill sweep over 256..30720 tokens, quadratic least-squares fit. |
| `s1_probe.py` | S1 core probe: per length, records TTFT (prefill) and median inter-token latency (decode at depth). Args: `PORT MODEL LABEL`. |
| `s3_capture.py` | S3: drives `/start_profile` and `/stop_profile` around one controlled request to capture a bounded profiler window. Args: `PORT MODEL LABEL PHASE LENGTH [GENSTEPS]`. |
| `tp_census.py` | buckets every GPU kernel in a trace into attention / matmul / other by kernel name and prints family shares plus the top kernels. Args: `TRACE ...`. |
| `render_figs.py` | renders `fig8a-prefill-divergence.png` and `fig8b-kernel-family-split.png` from the S1 numbers and the S3 census. |
| `run_s0.sh`, `run_s1.sh`, `run_s3.sh` | drivers that start a server per arm, wait for readiness, run the probe or captures, then tear the server down. |
| `run_census.sh` | runs `tp_census.py` over the captured engine-process traces for both arms in capture order. |

## Backend note

FlexAttention is PyTorch's general-purpose compiled attention kernel, not a naive
materialize-the-whole-score-grid implementation. The gaps in the post are therefore what
FlashAttention's specialization (its memory-traffic tuning and its split-KV decode kernel)
buys over a competent generic baseline, not flash-versus-nothing.

## Reading a trace without a GPU

No GPU is needed to read a captured trace, only to record one. Gunzip any file in
`benchmarks/08-flash-attention/traces/` and open the `.json` in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev), or run `tp_census.py` against the `.json.gz`
directly for the kernel-family census quoted in the post.
