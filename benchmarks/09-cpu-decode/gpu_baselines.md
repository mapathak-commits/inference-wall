# GPU baselines (from the blog — A10G, vLLM 0.18.0)

The CPU numbers are measured against these. All GPU numbers are already published /
drafted in "The Inference Wall"; source draft cited per row. Same box as this study.

## Qwen3.5-4B fp16 (Parts 1 & 3)

| metric | value | source |
|---|---|---|
| Single-stream decode (`max_num_seqs=1`) | **49 tok/s** | Part 3 sweep, row 1 |
| Batched throughput (`max_num_seqs=256`) | **1,091 tok/s** | Part 3 sweep, row 3 |
| Median ITL, single stream | 20.0 ms | Part 3 |
| Batching lift | ~22x (49 → 1,091) | Part 3 headline |
| fp16 weights on GPU | 8.6 GiB | Part 1 / Part 5 |

## Qwen3.5-9B AWQ 4-bit (Part 5)

| metric | value | source |
|---|---|---|
| tok/s at offered rate 4 | 478 | Part 5 |
| Sustained batched ceiling | ~820 tok/s | Part 5 |
| 4-bit weights on GPU | 11.2 GiB | Part 5 |

## The comparisons this study draws

- **CPU fp16/bf16 single-stream decode** vs **GPU 4B fp16 single-stream = 49 tok/s**.
  Roofline predicts the CPU is ~18x slower here (bandwidth ratio 600/33), i.e. ~2-3 tok/s.
- **CPU Q4_K_M single-stream decode** vs the same 49 tok/s GPU line (and against the
  Part 5 quantized-GPU story). Q4 cuts CPU bytes/token ~4x, so ~8-12 tok/s expected.
- **CPU batched throughput** vs **GPU batched 1,091 tok/s** — the widest gap, because the
  GPU's batching amortizes the weight read across ~150 sequences and the CPU cannot
  (it saturates memory bandwidth at ~2 threads; see stream_sweep.jsonl).
- **Prefill**: CPU GEMM peak ~0.7 TFLOP/s vs A10G ~31 TFLOP/s (cuda) → ~43x, the
  compute-bound gap, much wider than decode's bandwidth-bound gap.
