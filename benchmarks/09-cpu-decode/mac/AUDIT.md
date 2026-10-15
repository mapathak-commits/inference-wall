# Mac study audit ledger (Stage 9 of mac/FINDINGS.md)

Every stage above was re-derived from its raw output, and every result that looked surprising was
rerun. This section is the ledger, because the pattern in the failures turned out to be more
interesting than any individual correction.

| # | claim as first written | verdict | now |
|---|---|---|---|
| 1 | STREAM ceiling **232 GB/s**, declining past t=8 from E-core contention | **flawed → reran** | **238.2 GB/s** read peak, flat to t=14. The decline was noise; windows were 2.3 to 2.9 ms with best/mean to 2.43x |
| 2 | `decode_gemv_attn` reaches **230 GFLOP/s** | **flawed → reran** | **85 to 87 GFLOP/s**. 230 GF implies 460 GB/s = 1.9x the chip's ceiling , it was a cache-residency artifact, inflated up to 2.7x |
| 3 | Decode GEMV is **84%** of the bandwidth ceiling | **corrected** | 81% (mlp) / 72% (attn); and §2b now shows this is the *vendor GEMV limit*, not ggml slack |
| 4 | Prefill GEMM **3.0 TFLOP/s** | **holds** | 3.13 TFLOP/s; rerun moved individual cells −2.6% to +4.3% |
| 5 | §7b: decode is bandwidth-bound **without being stalled**; a stall counter is the wrong instrument | **falsified by my own new anchor** | A saturated 9-thread read loop stalls **78.5%**, not 34.3%. The old low anchor was single-threaded at 35% of chip bandwidth |
| 6 | §4b: partial offload at `-ngl 12` **halves** decode (0.51x) | **flawed → reran twice** | Broad plateau at **0.80 to 0.87x** over `-ngl 4` to `16`; break-even at `-ngl` ~20 to 21. The U-shape is real; the depth was a thermal-ordering artifact and pass 2's right-hand end was a contention artifact. Pass 3 also yields the mechanism: a **~7 ms/token step for splitting the graph at all**, worth ~0.35 ms/layer to buy back, predicting break-even at 20.4 layers |
| 6b | §4b pass 2: full offload is 1.43x | **flawed → reran** | `-ngl 32` is thread-independent (53.70 / 54.34 tok/s). Against the CPU's *own* optimum (`-t 8`) it is **1.06 to 1.09x**, matching §5a's 1.089x. Pass 2's denominator was a `-t 10` baseline that had collapsed to 31.85 tok/s |
| 7 | §6c: KV `q8_0` crossover **~8 to 12k**, a win by 16k | **reran; sign change survives, location moved** | Crossover **at ~16k** (median exactly 1.000x over 6 rounds); worst at 8k (0.843x); wins 1.128x at 32k |
| 8 | §6a: the roofline fit predicted an unseen cell to **0.4%** | **overclaim** | That cell's own error bar is ±4.3%. The prediction lands inside it , which is all the test can show |
| 9 | §5b: CPU fit `BW=209.2, t_fixed=4.64 ms` | **slope robust, intercept not** | Slope holds within 1% across all 81 leave-two-out subsets; `t_fixed` is only pinned to **0.77 to 9.72 ms** |
| 10 | §5a: `r3_cpu_Q8_0` outlier at 32.57 tok/s | **genuine transient** | A real hot-state excursion. The median (1.089x) is the right statistic and already handles it |
| 11 | §5a/§5b table transcription | **clean** | Matches `avg_ts` exactly, cell for cell |
| 12 | §7a: decode clocks **higher** than prefill (3882 vs 3641 MHz) at the same power | **replicated** | Second `powermetrics` window: **3883 vs 3622 MHz**. Decode's clock reproduced to **1 MHz** |
| 13 | §7a: decode draws slightly **less** power than prefill (−3.9%) | **noise** | Run 2 has decode −0.4%; the two phases' ranges overlap and the sign flips. Removed |
| 14 | §7a: Metal decode uses **1.78x** less energy per token | **replicated** | 1.73x in run 2; report as **~1.75x ±3%** |

**The failures have one shape.** Items 1, 2 and 6 are all the same mistake: **best-of-N over a
window too short to be a rate measurement.** A 57 µs `sgemm` timed once, a 2.4 ms STREAM sweep,
a decode cell averaged over three consecutive repeats while the machine heated , in each case the
statistic was reported to three significant figures and in each case it was measuring something
other than the thing named. Item 5 is a variant: an *anchor* rather than a window, but the same
failure to check that the reference was doing what its label said. And item 2 is the one with a
built-in alarm I initially walked past: **230 GFLOP/s at 0.5 flops/byte implies 460 GB/s on a
273 GB/s machine.** A number above a physical ceiling is not a surprising result, it is a proof of
a bug, and it was sitting in a table I had already written conclusions on top of.

**The second shape is a bad denominator, and it cost three findings.** Items 2, 5 and 6b are one
error: a ratio reported against a reference that was never at the limit its label implied. §2b
divided by a GEMV whose B matrix was cache-resident; §7b divided by a STREAM anchor running on one
thread at 35% of chip bandwidth; §4b divided by a `-t 10` decode baseline that had collapsed to
31.85 tok/s. In all three the numerator was fine. **A ratio inherits the weakest claim of its
denominator, and "this reference is at its limit" has to be measured, not assumed** , which is why
every reference workload in this study now prints its own achieved rate.

The practical countermeasures, all now in the code: time batches sized to a target window rather
than single calls (`TARGET_WINDOW_S` in `gemm_bench.c`), report best/mean as a noise indicator
alongside every rate, rotate working sets past every cache when measuring DRAM, interleave
round-robin so ratios are computed *within* a thermal state, and make every reference workload
print its own achieved rate so it can be checked against the ceiling it claims to represent.

**What the audit did not shake.** The thesis and its quantitative core came through unchanged:
decode time is linear in weight-bytes at R² = 0.9998 with a slope equal to measured bandwidth;
the CPU→GPU decode ratio is 1.089x against a 1.0x bandwidth ratio; the prefill ratio is 2.9 to 3.5x
against a 2.7x FLOP ratio; §5c's byte-identical bf16-vs-F16 control is 5.22x. None of those rest
on a short window, and two of them (§5b, §5c) are built specifically to be ratio-robust. The
corrections landed on mechanism stories and on secondary knobs , which is roughly where you would
expect them to land, and is not a reason to be comfortable about the ones that were wrong.

### Known gaps, stated rather than papered over

- ~~**§7a's clock inversion rests on one `powermetrics` window.**~~ **Closed.** A second window was
  captured 2.5 h later and reproduced decode's clock to 1 MHz (3882 → 3883) and the energy ratio to
  ±3%. It also showed that the "decode draws less power" detail was noise. Both captures are kept
  (`results/profile/pm_*.run{1,2}.txt`, `results/powermetrics_summary.run{1,2}.json`).
- **§5b's CPU intercept cannot be fixed by repetition.** Pinning `t_fixed` needs a *shorter lever*
  , a 0.5 to 1 B model added to the quantization ladder, so the fit has a point where fixed overhead
  is a large fraction of the token. That is a new experiment, not a rerun.
- **§7b's Delivery slot is assigned by elimination, never measured.** `bench/gen_icache.py` would
  need regenerating with several independent accumulators per function so the code footprint, not
  the dependency chain, is the binding constraint. Left undone because Delivery is <1% everywhere
  it is used.
- **§6c's non-monotonic 8k trough is unexplained.** A two-term bytes-saved-minus-dequant-cost model
  is monotonic in depth and cannot produce it. My hypothesis (f16 and q8_0 crossing the cache
  boundary at different depths) is untested.
- **§4b's ~7 ms split-graph step is measured; its mechanism is not.** A flat per-token boundary cost
  and an elevated per-CPU-layer cost inside a split graph fit the seven interior points equally
  well, and §5b's CPU intercept (0.77 to 9.72 ms) is too loose to arbitrate. The prefill contrast
  (0.21 ms vs 5.79 ms per token of work) favours a per-graph-evaluation cost but does not prove it.
  Settling it needs a per-token timeline (Instruments `os_signpost` around the CPU/Metal handoff)
  or a model with a different layer count to separate per-layer from per-token terms.
- **The Metal side at depth is not covered at all** (§6 is CPU-only), and the Linux box has no
  power data (§8f), so the energy comparison that §8f makes interesting cannot be made across
  boxes.
