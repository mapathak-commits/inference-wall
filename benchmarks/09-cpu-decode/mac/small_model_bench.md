# Small-model follow-up (Stage 10): M4 Pro, CPU vs Metal decode

Run 2026-10-15, 10:01–10:04 local, on AC power. llama.cpp `dfc29b6` (same build as Stage 5;
`results/stage10/build_commit.txt`). Runner `bench/run_stage10_small.sh`, analysis
`bench/analyze_stage10.py` (full output: `results/stage10_summary.txt`, `results/stage10_summary.json`).
Raw cells: `results/stage10/r<round>_<engine>_<file>.{md,jsonl}`.

Protocol as specified: for each file, CPU (`-ngl 0 -t 9`) then Metal (`-ngl 99`), `-p 128 -n 128 -r 3`,
two rounds, 4B F16 anchor at the end of each round. A cell is the median of the two rounds (with two
rounds that is their mean). A ratio is the median of the two within-round ratios.

## Files

| file | bytes (`ls -l`) | Linux bytes | arch / blocks | output head | streamed GB/token |
|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_k_m | 491,400,032 | 491,400,032 ✓ | qwen2 / 24 | untied | 0.392 |
| qwen2.5-0.5b-instruct-q8_0 | 675,710,816 | 675,710,816 ✓ | qwen2 / 24 | untied | 0.525 |
| qwen2.5-0.5b-instruct-fp16 | 1,266,425,696 | 1,266,425,696 ✓ | qwen2 / 24 | untied | 0.988 |
| qwen2.5-1.5b-instruct-q4_k_m | 1,117,320,736 | not given | qwen2 / 28 | untied | 0.980 |
| qwen2.5-1.5b-instruct-fp16 | 3,560,416,288 | not given | qwen2 / 28 | untied | 3.088 |
| Qwen_Qwen3.5-4B-F16 (anchor) | 8,665,620,192 | — | qwen35 / 33 | tied | 8.655 |

The three 0.5B files match the Linux run byte for byte. The brief gives no Linux sizes for the two
1.5B files, so those **could not be verified**.

**Bytes streamed per token are not file bytes for these Qwen2.5 files.** The official Qwen2.5 GGUFs
store `output.weight` as a separate tensor from `token_embd.weight`, so decode reads only one row of
the embedding table per token. File bytes exceed streamed bytes by 28% (0.5B F16), 29% (0.5B Q8),
25% (0.5B Q4), 15% (1.5B F16) and 14% (1.5B Q4). Qwen3.5-4B is tied, so its file bytes are honest
(less 11 MB of metadata). Streamed GB is total tensor bytes minus `token_embd` when the head is
untied (`bench/gguf_tensors.py`, a stdlib GGUF header reader). The brief's analysis uses file GB, and
it is reported below as asked. Every fit is also given on streamed GB, which is the physically
meaningful axis.

The GGUF `block_count` for the Qwen3.5-4B anchor is 33, not the 32 stated in the brief.

## 1. Decode table (tg128, tok/s, median of 2 rounds)

| file | bytes | CPU tg128 | Metal tg128 | Metal/CPU | CPU pp128 | Metal pp128 |
|---|---|---|---|---|---|---|
| 0.5B Q4_K_M | 491,400,032 | 324.30 | 291.72 | **0.899x** | 1577 | 5608 |
| 0.5B Q8_0 | 675,710,816 | 305.10 | 263.23 | **0.863x** | 2902 | 6013 |
| 0.5B F16 | 1,266,425,696 | 187.12 | 193.38 | **1.034x** | 1382 | 6096 |
| 1.5B Q4_K_M | 1,117,320,736 | 162.04 | 172.70 | 1.066x | 640 | 2067 |
| 1.5B F16 | 3,560,416,288 | 66.54 | 73.02 | 1.098x | 447 | 2201 |
| 4B F16 anchor | 8,665,620,192 | 23.54 | 23.24 | 0.987x | 156 | 752 |

Per round (tok/s; per-rep samples in `results/stage10_summary.txt`):

| file | r1 CPU / Metal / ratio | r2 CPU / Metal / ratio |
|---|---|---|
| 0.5B Q4_K_M | 329.23 / 304.82 / 0.926x | 319.37 / 278.62 / 0.872x |
| 0.5B Q8_0 | 299.75 / 266.85 / 0.890x | 310.44 / 259.61 / 0.836x |
| 0.5B F16 | 183.55 / 192.06 / 1.046x | 190.68 / 194.70 / 1.021x |
| 1.5B Q4_K_M | 161.60 / 172.51 / 1.067x | 162.47 / 172.89 / 1.064x |
| 1.5B F16 | 66.17 / 73.15 / 1.105x | 66.90 / 72.90 / 1.090x |
| 4B F16 | 23.11 / 23.12 / 1.001x | 23.97 / 23.35 / 0.974x |

All 12 within-round ratios: median 1.011x, range 0.836–1.105x.

**Drift check.** The anchor's CPU decode is 23.54 tok/s against Stage 5's F16 median of 21.68,
**+8.5%**, inside the 10% limit, so pooling is permitted. The difference is mostly not thermal:
Stage 5 ran at `-t 10` and this stage at `-t 9`, and Stage 8e measured t=9 beating t=10 by about 8%.
Metal's anchor is 23.24 against Stage 5's 22.24 (+4.5%).

**Short-window caveat, and a check.** A tg128 rep at 0.5B lasts about 0.4 s, the kind of window Stage 9
warned about. The inversion at 0.5B Q4 and Q8 is the headline result, so it was rechecked in the same
session with `-p 0 -n 1024 -r 3` (about 3.5 s per rep), interleaved over two rounds
(`results/stage10/longwin/`). This is a supplement and is kept out of the main table:

| file | r1 CPU / Metal / ratio | r2 CPU / Metal / ratio |
|---|---|---|
| 0.5B Q4_K_M | 311.5 / 297.4 / 0.955x | 305.2 / 298.8 / 0.979x |
| 0.5B Q8_0 | 301.0 / 279.2 / 0.928x | 299.5 / 282.7 / 0.944x |

With the longer window the CPU still wins at 0.5B quantized, but by 3–7%, not 10–14%. A 1024-token
run also averages over a deeper KV cache, so this tests the sign, not the exact tg128 magnitude.

## 2. Roofline fits, `ms/token = GB / BW + t_fixed`

"Pooled" (A) is the five small files, the anchor, and Stage 5's four medians, n = 10. "Session" (B) is
the five small files plus the anchor, n = 6. The ranges in brackets come from refitting (B) over all
2⁶ = 64 choices of which round supplies each point (the analogue of Stage 5's 81-combination check).

### On file GB (as specified)

| engine | fit | BW GB/s | t_fixed ms | R² | max residual |
|---|---|---|---|---|---|
| CPU | A pooled | 195.0 | 0.62 | 0.97808 | +33.0% (0.5B F16) |
| CPU | B session | 206.9 [202.5–211.4] | **−0.14** [−0.33–0.04] | 0.99428 | −27.6% (0.5B Q4) |
| Metal | A pooled | 197.9 | 0.14 | 0.98460 | +32.4% (1.5B F16) |
| Metal | B session | 206.0 [204.4–207.5] | **−0.28** [−0.38–−0.18] | 0.98673 | −38.6% (0.5B Q4) |

Residuals for fit A (positive means the fit is slower than the measurement):

| point | CPU file GB / meas ms / err | Metal meas ms / err |
|---|---|---|
| 0.5B Q4_K_M | 0.491 / 3.08 / +1.7% | 3.43 / −23.5% |
| 0.5B Q8_0 | 0.676 / 3.28 / +24.5% | 3.80 / −6.4% |
| 0.5B F16 | 1.266 / 5.34 / +33.0% | 5.17 / +26.5% |
| 1.5B Q4_K_M | 1.117 / 6.17 / +2.8% | 5.79 / −0.1% |
| 1.5B F16 | 3.560 / 15.03 / +25.6% | 13.69 / +32.4% |
| 4B F16 anchor | 8.666 / 42.49 / +6.0% | 43.04 / +2.1% |
| S5 Q4_K_M | 3.013 / 19.19 / −16.3% | 17.73 / −13.4% |
| S5 Q6_K | 3.563 / 21.69 / −12.9% | 19.87 / −8.7% |
| S5 Q8_0 | 4.622 / 26.49 / −8.2% | 23.98 / −2.0% |
| S5 F16 | 8.666 / 46.13 / −2.3% | 44.96 / −2.3% |

A negative intercept is unphysical. On file bytes the one-line model fails: residuals reach 33%,
against ≤4% in Stage 5. The main reason is that the axis counts embedding bytes that are never read.

### On streamed GB (corrected axis)

| engine | fit | BW GB/s | t_fixed ms | R² | max residual |
|---|---|---|---|---|---|
| CPU | A pooled | 198.1 | 1.50 | 0.98685 | +26.7% (0.5B Q8) |
| CPU | **B session** | **209.1** [204.6–213.6] | **0.91** [0.74–1.07] | **0.99916** | −9.8% (0.5B Q4) |
| CPU | C Qwen2.5 only (n=5) | 223.1 | 1.23 | 0.99475 | +9.2% |
| CPU | D Qwen2.5 F16 only (n=2) | 216.8 [212.8–221.0] | 0.79 [0.60–0.98] | — | — |
| Metal | A pooled | 201.1 | 1.02 | 0.99323 | +19.5% (1.5B F16) |
| Metal | **B session** | **207.7** [206.2–209.3] | **0.75** [0.66–0.85] | 0.99564 | −23.1% (0.5B Q4) |
| Metal | C Qwen2.5 only (n=5) | 260.1 | 1.78 | 0.99650 | +8.0% |
| Metal | D Qwen2.5 F16 only (n=2) | 246.3 [244.6–248.1] | 1.16 [1.10–1.22] | — | — |

Session-fit residuals on streamed GB, CPU / Metal: 0.5B Q4 −9.8% / −23.1%, 0.5B Q8 +4.3% / −13.7%,
0.5B F16 +5.4% / +6.5%, 1.5B Q4 −9.4% / −5.6%, 1.5B F16 +4.3% / +14.0%, anchor −0.5% / −1.4%.

**Does the intercept agree between pooled and session-only?** Not well. On streamed GB the pooled
intercept is 1.50 ms CPU and 1.02 ms Metal, against 0.91 and 0.75 for the session fit. Both are near
1 ms and far below Stage 5's 4.64 / 2.40. But the pooled fit is poor, because all four Stage 5 points
sit slower than the session line: on the CPU by 26% (Q4_K_M), 21% (Q6_K), 15% (Q8_0) and 9% (F16), on
Metal by 4–17%, worst at Q4. Those points are a different session, `-t 10`, and quantized
Qwen3.5, and this session has no quantized Qwen3.5 cell, so the offset cannot be split between thread
count, thermal state, and an architecture- or format-specific cost.

**The quantized points are systematically slow on both engines.** At almost the same streamed bytes,
0.5B F16 (0.988 GB) decodes in 5.34 ms on the CPU and 1.5B Q4_K_M (0.980 GB) in 6.17 ms, 16% slower
(Metal 5.17 vs 5.79 ms, 12%). The Q4 model has 2.8x the parameters and 4 more layers, so this one
comparison cannot say whether the extra cost is per dequantized weight or per layer. Either way, a
single-slope model whose only per-model term is a constant cannot express it. On Stage 5's ladder,
format changes together with bytes (Q4 at the small end, F16 at the large end), so that format cost
was absorbed into the intercept. This is the most likely reason Stage 5's CPU intercept read 4.6 ms.

**The intercept range that can now be defended:**

- **Qwen2.5 dense, this machine, `-t 9`:** CPU t_fixed is **0.6–1.1 ms**, Metal **0.7–1.2 ms**
  (streamed-GB session fit and F16-only fit, each over every choice of round). The two engines' fixed
  costs are about equal. The CPU's is not the larger one.
- **Qwen3.5-4B (the Stage 5 model):** this stage cannot pin its intercept, because the short-lever
  points belong to a different architecture. One assumption-light bound does follow from the anchor:
  CPU decode cannot stream faster than the measured 238.2 GB/s STREAM ceiling, so at `-t 9`
  t_fixed ≤ 42.49 − 8.655/0.2382 = **6.15 ms**. If Qwen3.5 shared the session slope (209.1 GB/s), its
  intercept would be 1.1 ms. Applied to Stage 5's own `-t 10` F16 cell (46.13 ms), the same ceiling
  bound gives ≤9.8 ms, which does not tighten Stage 5's 0.77–9.72 ms.

## 3. Effective bandwidth per cell (tok/s × GB)

| file | CPU, file GB | Metal, file GB | CPU, streamed | Metal, streamed |
|---|---|---|---|---|
| 0.5B Q4_K_M | 159.4 | 143.4 | 127.1 | 114.3 |
| 0.5B Q8_0 | 206.2 | 177.9 | 160.2 | 138.2 |
| 0.5B F16 | 237.0 | 244.9 | 184.9 | 191.1 |
| 1.5B Q4_K_M | 181.0 | 193.0 | 158.8 | 169.3 |
| 1.5B F16 | 236.9 | **260.0** | 205.4 | 225.5 |
| 4B F16 | 204.0 | 201.4 | 203.7 | 201.1 |

On file bytes, two Qwen2.5 F16 cells sit at the 238 GB/s CPU STREAM ceiling, and Metal's 1.5B F16 is
above it (260 GB/s). That is the Stage 9 alarm again: a number at or above a ceiling was the first
sign that the axis was wrong. On streamed bytes, every cell is below the ceiling. The departure at
small size is then plain to see: at 0.5B Q4 the CPU reaches 53% of 238 GB/s and Metal only 48%,
against 78–95% for the F16 cells.

## 4. Answer

**No: the Metal/CPU gap did not rise for small models. It fell, and at the two smallest quantized
files it inverted.** At 0.5B F16 the ratio is **1.034x**. That is +4.8% over the same-session 4B
anchor (0.987x) but below Stage 5's 1.089x, so it is not the predicted rise. At 0.5B Q4_K_M and Q8_0
the CPU is faster: 0.899x and 0.863x at tg128, and 0.97x and 0.94x on a 1024-token window. The
prediction's premise does not hold for these models. With the axis corrected for unread embedding
bytes, the two engines' fixed costs are roughly equal (CPU 0.91 ms, Metal 0.75 ms), so the roofline
alone predicts a ratio near 1.0 at every size. What actually moves the ratio is that Metal falls
further below its own line than the CPU does on small *quantized* models. Against each engine's
F16-only line, 0.5B Q8_0 costs the CPU +0.07 ms and Metal +0.51 ms. A likely cause is that the
quantized mat-vec kernels fill the GPU poorly at hidden size 896, but this is a hypothesis, not
measured. That puts the Apple side in the *same* direction as the prediction for the Linux box (the
gap shrinks at small sizes), so the "opposite directions" test fails here. The Linux numbers are not
in this directory and were not checked. Whether the fitted CPU intercept falls inside Stage 5's
0.77–9.72 ms depends on the axis. On file bytes (−0.14 ms) it falls outside, and below zero. On
streamed bytes (0.91 ms, 0.74–1.07 over round choices) it falls just inside the bottom edge and
straddles it. Either way it is roughly 5x below Stage 5's point estimate of 4.64 ms, and it is a
Qwen2.5 intercept, so the fixed cost is model-specific and Stage 5's number should not be carried
over to other architectures.

## Not done / not verified

- ~~powermetrics energy capture: skipped.~~ Done later; see the energy section below.
- 1.5B byte sizes were not checked against Linux, because no reference sizes were given.
- The Linux-side result for the "opposite directions" test was not available here.
- Two rounds per cell, as specified. The 64-combination ranges show how much the fit moves with the
  choice of round, not run-to-run variance in general.
- The mechanism for Metal's slowdown on small quantized models (kernel occupancy versus per-dispatch
  cost) was not measured.

## Repeat on an idle machine (added later the same day)

Full protocol repeated 11:02:55–11:05:01 into `results/stage10_rerun/` (analysis:
`results/stage10_rerun_summary.txt`), because part of the first run's day overlapped a Google Meet call.

| file | CPU tg128 | Metal tg128 | Metal/CPU (run 1) |
|---|---|---|---|
| 0.5B Q4_K_M | 305.59 | 289.36 | **0.948x** (0.899x) |
| 0.5B Q8_0 | 310.45 | 279.79 | **0.901x** (0.863x) |
| 0.5B F16 | 195.62 | 194.01 | **0.992x** (1.034x) |
| 1.5B Q4_K_M | 163.68 | 173.77 | 1.062x (1.066x) |
| 1.5B F16 | 66.48 | 73.71 | 1.111x (1.098x) |
| 4B F16 anchor | 24.85 | 25.73 | 1.036x (0.987x) |

- **Drift:** anchor CPU is +14.5% over Stage 5's 21.68 tok/s, which is over the 10% limit, so this
  session is not pooled with Stage 5. Only session-only fits are reported.
- **Session fit, streamed GB:** CPU 221.6 GB/s, t_fixed **1.17 ms** [1.05–1.29]; Metal 231.8 GB/s,
  t_fixed **1.21 ms** [1.19–1.23].
- **Session fit, file GB:** CPU 219.1 GB/s, t_fixed 0.17 ms; Metal 229.5 GB/s, 0.27 ms.
- **Ceiling bound on the Qwen3.5-4B CPU intercept:** ≤3.9 ms (anchor 40.25 ms per token).
- **Long window (`-n 1024`):** Q4_K_M 0.979x / 0.939x, Q8_0 0.934x / 0.949x. The Mac idle-slept for 16
  minutes at 11:06 during this check. The three affected cells (about 130 tok/s) are kept in
  `longwin/discarded_sleep/`, and round 2 was redone as adjacent pairs under `caffeinate -i` at 13:15.

**Combined conclusion:** for Qwen2.5 the fixed cost is 0.7–1.3 ms on the CPU and 0.7–1.2 ms on Metal
across both sessions. The CPU wins at 0.5B quantized in every pair measured.

## Energy per token at 0.5B Q4_K_M (added later)

Capture: `results/profile_small/` (run by the user under sudo, 14:54–14:55). Rates: `results/profile_small/toks/`
(14:56–14:57, `-t 9 -p 0 -n 512 -r 5`, two interleaved rounds).

| | CPU decode | Metal decode |
|---|---|---|
| CPU W / GPU W | 40.91 / 0.00 | 13.61 / 15.11 |
| marginal W | 40.82 | 28.63 |
| tok/s | 309.5 | 256.1 |
| **mJ/token** | **132** | **112** |

Metal is **1.18x** more energy-efficient per token (about 1.12–1.24x given Metal's ±5% tok/s spread),
against ~1.75x at 4B (§7a). The CPU draws 13.6 W even during Metal decode, because at ~256 tok/s the
host-side work of encoding and submitting each token's GPU commands is a large share of the total.
Metal ran at `-t 9` here (the profiling script's fixed setting) and decoded at 256 tok/s, against about
290 at the default thread count. The energy figure is for `-t 9` only.
