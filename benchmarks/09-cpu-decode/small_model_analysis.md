# Small-model follow-up: does the CPU-to-GPU decode gap depend on model size?

Server run 2026-10-15 (`run_small.sh`, raw in `small_model_bench.md`): EPYC slice CPU (8
physical cores pinned) vs A10G via llama.cpp CUDA, build 458681e, same session, `-p 128 -n 128
-r 3`. Mac run the same day (`mac/results/small_model_bench.md`, build dfc29b6, `-t 9`, two
sessions). Official Qwen2.5 GGUFs, byte-identical on both machines (all five files verified).

## The axis: streamed bytes, not file bytes

The official Qwen2.5 GGUFs store `output.weight` separately from `token_embd.weight`, so the
embedding table is read one row per token, not streamed. File bytes overstate streamed bytes
by 25 to 29% for the 0.5B files and 14 to 15% for the 1.5B files. Qwen3.5-4B stores them
tied, so its file bytes are honest (the earlier note in this file suggesting a 15% correction
for the 4B was wrong and is withdrawn). Streamed GB per token, from the GGUF headers: 0.5B
Q4 0.392, Q8 0.525, F16 0.988; 1.5B Q4 0.980, F16 3.088; 4B F16 8.655. On file bytes the
one-line fit gives negative intercepts on every engine; on streamed bytes it fits with
residuals under 10% and physical intercepts.

## Server slice CPU vs A10G

| file | streamed GB | CPU tg128 | A10G tg128 | gap | CPU eff GB/s | A10G eff GB/s |
|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 0.392 | 69.33 | 511.28 | **7.4x** | 27.2 | 200 |
| 0.5B Q8_0 | 0.525 | 56.17 | 471.46 | 8.4x | 29.5 | 248 |
| 1.5B Q4_K_M | 0.980 | 29.47 | 293.76 | 10.0x | 28.9 | 288 |
| 0.5B F16 | 0.988 | 31.42 | 339.59 | 10.8x | 31.0 | 336 |
| 1.5B F16 | 3.088 | 10.23 | 139.73 | 13.7x | 31.6 | 432 |
| 4B F16 (anchor) | 8.655 | 3.80 | 54.34 | **14.3x** | 32.9 | 470 |

Fits `ms_per_token = GB / BW + t_fixed` on streamed GB, six points:

| engine | BW | t_fixed | R2 | max residual |
|---|---|---|---|---|
| EPYC slice CPU | 33.2 GB/s | 3.06 ms (F16-only 3.27) | 0.9998 | 6.0% |
| A10G | 503 GB/s | 1.15 ms (F16-only 0.94) | 0.9993 | 9.0% |

## M4 Pro CPU vs Metal (from the Mac file)

| file | Metal/CPU run 1 | idle repeat |
|---|---|---|
| 0.5B Q4_K_M | 0.899x | 0.948x |
| 0.5B Q8_0 | 0.863x | 0.901x |
| 0.5B F16 | 1.034x | 0.992x |
| 1.5B Q4_K_M | 1.066x | 1.062x |
| 1.5B F16 | 1.098x | 1.111x |
| 4B F16 anchor | 0.987x | 1.036x |

Session fits on streamed GB: CPU 209.1 / 221.6 GB/s with t_fixed 0.91 / 1.17 ms; Metal 207.7 /
231.8 GB/s with t_fixed 0.75 / 1.21 ms. Energy at 0.5B Q4: Metal 112 vs CPU 132 mJ/token,
1.18x (4B: 1.75x), because the CPU draws 13.6 W feeding the GPU at ~256 tok/s.

## Reconciliation

1. **Prediction failed.** The brief predicted the Metal lead would grow at small sizes because
   Stage 5's CPU intercept (4.6 ms) was larger than Metal's (2.4 ms). With the format held
   fixed both intercepts are about 1 ms. Stage 5's ladder changed format together with size,
   so the per-byte dequantization cost of Q4/Q6/Q8 was absorbed into the intercept.
2. **Fixed cost per token:** A10G ~1 ms, M4 CPU ~1 ms, Metal ~1 ms, EPYC slice CPU ~3 ms.
3. **Both machines bend toward the CPU at small size.** Server: 14.3x -> 7.4x, because the
   GPU's 1 ms is over half of a 0.4 GB token while the CPU's 3 ms is a fifth of its 12 ms
   stream; as size -> 0 the gap tends to the fixed-cost ratio, ~2.7x. Mac: ~1x at every size
   for F16 (engines tie at the 4B anchor); CPU wins 0.86 to 0.95x on 0.5B Q4/Q8 because
   Metal's quantized kernels fall further below their own line at hidden size 896
   (hypothesis: poor GPU fill; not measured).
4. **Stage 5's 1.089x is format-dependent, not fixed-cost.** At F16 the engines tie; Metal's
   ~10% lead on the quantized 4B rows is consistent with cheaper quantized kernels at that
   size, which is an inference from the format-cost explanation, not a measurement.
5. These are dense 24/28-layer transformers; Qwen3.5-4B is hybrid with 33 GGUF blocks. The
   fixed cost is model-specific as well as engine-specific.
