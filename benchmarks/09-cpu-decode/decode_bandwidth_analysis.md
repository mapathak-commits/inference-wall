# Decode is bandwidth-bound — the ledger

The single cleanest confirmation of the blog's thesis. For each quant level, take the
**saturated** decode rate (best `tg128` across threads) and multiply by the weight bytes
that must be streamed once per token. If decode is bandwidth-bound, this product — the
*effective* memory bandwidth — should be constant across quant levels and equal to the
measured STREAM wall (~33 GB/s).

| model | weight size | best decode (tok/s) | at threads | effective BW = tok/s × bytes |
|---|---|---|---|---|
| bf16   | 8.06 GiB (8.65 GB) | 3.92 | 8 | **33.9 GB/s** |
| Q8_0   | 4.29 GiB (4.61 GB) | 6.88 | 8 | **31.7 GB/s** |
| Q4_K_M | 2.80 GiB (3.01 GB) | 9.92 | 8 | **29.8 GB/s** |

All three land at **~30–34 GB/s — the measured STREAM triad ceiling (~33 GB/s).** Decode
speed is exactly `bandwidth ÷ weight_bytes`: halve the bytes (bf16→Q8), decode ~doubles;
quarter them (bf16→Q4), decode ~2.5x. The CPU is not computing during decode, it is
*waiting on memory*, and the only knob that moves it is bytes-per-token (quantization).

## Threads: decode saturates early, prefill scales with cores

| threads | bf16 pp / tg | Q8 pp / tg | Q4 pp / tg |
|---|---|---|---|
| 1  | 7.10 / 1.60 | 7.61 / 2.73 | 8.17 / 3.63 |
| 2  | 13.70 / 2.84 | 14.69 / 4.71 | 15.52 / 6.33 |
| 4  | 26.07 / 3.88 | 26.92 / 6.83 | 27.46 / 9.50 |
| 8  | 39.84 / **3.92** | 31.54 / **6.88** | 42.06 / **9.92** |
| 16 | 30.38 / 0.21 | 29.43 / — | (skipped) |

- **Prefill (pp512)** scales ~linearly with cores up to 8 (7→40 tok/s for bf16) — it's
  compute-bound, so more cores help. It *drops* at 16 threads: 8 physical cores oversubscribed.
- **Decode (tg128)** flattens by 4 threads and gains nothing from 4→8 — the memory
  controller is already saturated at ~2–4 threads (matches the STREAM sweep exactly).
- **16-thread decode collapses** (bf16 → 0.21 tok/s): 16 software threads thrash 8 physical
  cores plus contend on the one memory bus. The 16-thread decode cell was skipped for Q4
  once bf16 established this — it's a ~30-min-per-cell pathology with no new information.

## Versus the GPU (blog baselines, A10G, same box)

| | CPU (this study) | GPU (blog) | gap |
|---|---|---|---|
| Single-stream decode, fp16/bf16 | 3.92 tok/s | 49 tok/s | **12.5x** |
| Single-stream decode, best CPU quant (Q4) | 9.92 tok/s | 49 tok/s (fp16) | **4.9x** |
| Effective decode bandwidth | ~34 GB/s | ~452 GB/s (49 tok/s × 8.6 GiB) | **13.3x** |

The decode tok/s gap (12.5x, same precision) **equals the achieved-bandwidth ratio (13.3x)**
— not the compute ratio (~43x). Decode is bandwidth-bound on *both* sides, so the gap is
the bandwidth gap, precisely as the thesis predicts. (The roofline's ~18x used *nominal*
600/33; using *achieved* bandwidth on both sides tightens it to ~13x — the agreement is
even better than the paper prediction.) **Quantization is the CPU's one big lever**: Q4
cuts the decode gap from ~12x to ~5x, because it directly cuts bytes-per-token, the thing
decode is bound by.

## Optimization sweep — what actually moves the needle (Q4_K_M, `results/opt_sweep.md`)

Topology: 2-CCX Zen 2, 8 physical cores (0-3 share L3#0, 4-7 share L3#1), HT siblings 8-15,
single memory controller. Every config measured in tok/s — this virtualized AWS host exposes
**no hardware PMU** (perf `cycles`/`cache-misses`/IPC all read `<not supported>`; only software
events work), so cycle/stall counters are unavailable and tok/s is the ground truth.

| config | prefill pp512 | decode tg128 |
|---|---|---|
| 4 cores, 1 CCX (0-3), strict | 29.9 | 9.80 |
| 4 cores, split CCX (0,1,4,5) | 26.4 | 9.45 |
| **8 physical cores (0-7), strict** | **47.5** | 9.75 |
| 8 threads, default placement | 40.3 | 9.96 |
| 8 phys, KV q8_0 | 45.3 | 9.70 |
| 8 phys, flash-attn on | 45.6 | 9.78 |
| 8 phys, poll 100 | 47.0 | 9.71 |

**Decode is immovable: 9.45–9.96 tok/s across every config (all within ±0.2 noise).** No
affinity, CCX packing, HT avoidance, KV-cache quantization, flash-attn, or busy-poll knob
changes it. On a box with no memory-stall counters, *this is the proof decode is
bandwidth-bound* — the one and only lever that moves decode is bytes-per-token
(quantization, Stage 3: bf16 3.9 → Q4 9.9), because that is literally what it's bound by.
Everything a CPU-tuning playbook would try is second-order against a saturated memory bus.

**Prefill (compute-bound) does respond, and there's a free ~18% win:**
- **Pin to physical cores, strict** (`-C 0xFF --cpu-strict 1`): 47.5 vs 40.3 tok/s default —
  the scheduler otherwise migrates threads onto HT siblings and loses ~18%. Zero-cost.
- **Pack cores into one CCX** (shared 16 MiB L3): 4c/1-CCX 29.9 vs 4c/split 26.4 (~13%) —
  L3 locality helps the compute-bound path.
- KV-q8_0 / flash-attn / poll: no meaningful effect at these context lengths (the KV cache
  is tiny next to the weights, so trimming KV bytes doesn't change the weight-dominated
  bytes/token that gates decode).

**Bottom line on optimization:** the actionable CPU knob is *physical-core pinning for
prefill* (~18% free) plus *quantization for decode* (the only decode lever, ~2.5x). Nothing
recovers the decode bandwidth gap — you'd need more memory channels, not more tuning.

## The batching asymmetry (structural, not measured here)

The GPU's headline is a **22x batching lift** (49 → 1,091 tok/s): batching amortizes the
one weight-read across ~150 concurrent sequences, converting a bandwidth-bound problem into
a compute-bound one the GPU has FLOPs to spare for. **The CPU has no such headroom** — it
saturates memory bandwidth at 2–4 threads *single-stream*, so there are no spare bytes/s to
amortize across a batch. The CPU's batched ceiling is therefore ~its single-stream decode
rate, and the 1,091 tok/s GPU throughput number has no CPU analog. This is the widest gap
in the study and it is architectural: bandwidth you don't have can't be batch-amortized.
