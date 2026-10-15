# Small-model follow-up: does the CPU-to-GPU decode gap depend on model size?

Run 2026-10-15 (`run_small.sh`, raw in `small_model_bench.md`). EPYC slice CPU (8 physical
cores pinned) vs A10G via llama.cpp CUDA, build 458681e, same session, `-p 128 -n 128 -r 3`.
Official Qwen GGUFs so the Mac run can use byte-identical files.

| file | GB | CPU tg128 | A10G tg128 | gap | CPU ms/tok | A10G ms/tok | CPU eff GB/s | A10G eff GB/s |
|---|---|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 0.491 | 69.33 | 511.28 | **7.4x** | 14.42 | 1.96 | 34.1 | 251 |
| 0.5B Q8_0 | 0.676 | 56.17 | 471.46 | 8.4x | 17.80 | 2.12 | 38.0 | 319 |
| 1.5B Q4_K_M | 1.117 | 29.47 | 293.76 | 10.0x | 33.93 | 3.40 | 32.9 | 328 |
| 0.5B F16 | 1.266 | 31.42 | 339.59 | 10.8x | 31.83 | 2.94 | 39.8 | 430 |
| 1.5B F16 | 3.560 | 10.23 | 139.73 | 13.7x | 97.75 | 7.16 | 36.4 | 498 |
| 4B F16 (anchor) | 8.666 | 3.80 | 54.34 | **14.3x** | 263.2 | 18.40 | 32.9 | 471 |

**Answer: the gap shrinks with model size on the discrete-GPU box, from 14.3x at 8.7 GB to
7.4x at 0.5 GB, and the reason is the GPU's fixed per-token cost, not the CPU's.**

Fit `ms_per_token = GB / BW + t_fixed` over the six points:

| engine | 1/slope | intercept | R2 |
|---|---|---|---|
| EPYC slice CPU | 32.9 GB/s | about 0 ms (fit gives -3.7 ms, within the residuals) | 0.998 |
| A10G | 497 GB/s | **0.71 ms/token** | 0.995 |

The CPU stays at its 33 GB/s wall at every size, so its time per token is purely
bytes / bandwidth and its fixed cost is negligible on this slice (the earlier "~10 ms"
estimate was an artifact of dividing by 33 instead of the fitted 32.9). The A10G's
effective bandwidth falls from ~480 GB/s at 3 to 9 GB to 251 GB/s at 0.5 GB because its
0.7 ms per-token floor (kernel launches, one per layer) is a growing share of a token that
only takes 2 ms. In the bandwidth regime the ratio is 497 / 32.9 = 15x and model size
cancels; the fixed cost is what bends it.

Prediction for the M4 Pro (to be measured; see `mac/PROMPT_small_model.md`): there the CPU
has the larger intercept (Stage 5: a few ms vs Metal 2.4 ms), so the Metal/CPU gap should
move the other way, rising above 1.09x for small models.

## Caveat on "bytes per token"

Effective bandwidth here, and throughout the study, is `tok/s x file bytes`. The file
includes the token-embedding table, which is looked up one row per token, not streamed; only
the output head is read in full. For the 0.5B F16 file that table is about 272 MB of the
1.27 GB (21%), which is why the CPU reads as 39.8 GB/s, above its 33 GB/s wall; corrected
for the table it is about 31 GB/s, at the wall. For the 4B the share is smaller (about 15%)
and the same correction applies uniformly across formats, so every ratio and every
"constant across quants" claim is unaffected, but absolute effective-bandwidth figures in
the study overstate streamed bytes by that share. These are standard dense transformers (24
and 28 layers) rather than Qwen3.5's hybrid 32, so the fixed cost per token is
model-specific as well as engine-specific.
