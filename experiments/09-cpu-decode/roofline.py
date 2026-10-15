#!/usr/bin/env python3
"""Theoretical roofline for this box's CPU vs the A10G, framed for LLM inference.

Pure stdlib. Prints the two ceilings that decide the CPU-vs-GPU gap:
  - peak GEMM FLOPs (prefill / compute-bound work), and
  - peak memory bandwidth (decode / bandwidth-bound work),
plus the predicted gaps, before any measurement.
"""

# ---- CPU: AMD EPYC 7R32 (Zen 2), measured from lscpu/dmidecode on the host ----
CPU_CORES = 8                 # physical cores (16 threads, but FP throughput is per-core)
CPU_CLOCK_GHZ = 2.8           # sustained; 5599.99 BogoMIPS ~ 2.8 GHz base
# Zen 2 SIMD: AVX2 = 256-bit = 8 fp32 lanes; 2 FMA units/core; FMA = 2 flops/element.
AVX2_FP32_LANES = 8
FMA_UNITS = 2
FLOPS_PER_FMA = 2
# Memory: DDR4-3200 = 3200 MT/s * 8 bytes/transfer = 25.6 GB/s per channel.
DDR4_3200_GBPS_PER_CHANNEL = 25.6
# Channel count on this VM slice is measured by bench_mem.py; assume 2 as a nominal.
CPU_MEM_CHANNELS_NOMINAL = 2

# ---- GPU: NVIDIA A10G (blog rig), from spec ----
A10G_FP16_TFLOPS = 31.2       # non-tensor fp16; tensor-core fp16 is ~125 but LLM decode GEMV doesn't use it well
A10G_TENSOR_FP16_TFLOPS = 125.0
A10G_BW_GBPS = 600.0          # GDDR6, ~600 GB/s


def cpu_peak_gemm_gflops(fp32=True):
    # cores * lanes * FMA_units * flops_per_fma * clock
    lanes = AVX2_FP32_LANES if fp32 else AVX2_FP32_LANES * 2  # fp16 packs 2x, but Zen2 has no native fp16 FMA -> upconverts
    g = CPU_CORES * lanes * FMA_UNITS * FLOPS_PER_FMA * CPU_CLOCK_GHZ
    return g  # GFLOP/s


def main():
    cpu_gflops = cpu_peak_gemm_gflops(fp32=True)
    cpu_bw = DDR4_3200_GBPS_PER_CHANNEL * CPU_MEM_CHANNELS_NOMINAL

    print("=== ROOFLINE: CPU (EPYC 7R32, Zen2/AVX2) vs GPU (A10G) ===\n")

    print("COMPUTE ceiling (prefill / GEMM-bound):")
    print(f"  CPU  fp32 GEMM peak : {cpu_gflops/1000:6.2f} TFLOP/s "
          f"({CPU_CORES} cores x {AVX2_FP32_LANES} AVX2 lanes x {FMA_UNITS} FMA x2 x {CPU_CLOCK_GHZ}GHz)")
    print(f"  GPU  fp16 (CUDA)    : {A10G_FP16_TFLOPS:6.2f} TFLOP/s")
    print(f"  GPU  fp16 (tensor)  : {A10G_TENSOR_FP16_TFLOPS:6.2f} TFLOP/s")
    print(f"  --> prefill gap (GPU cuda / CPU): {A10G_FP16_TFLOPS/(cpu_gflops/1000):5.1f}x")
    print(f"  --> prefill gap (GPU tensor / CPU): {A10G_TENSOR_FP16_TFLOPS/(cpu_gflops/1000):5.1f}x\n")

    print("BANDWIDTH ceiling (decode / weight-streaming-bound):")
    print(f"  CPU  DDR4-3200 x{CPU_MEM_CHANNELS_NOMINAL}ch : {cpu_bw:6.1f} GB/s  (nominal; measured by bench_mem.py)")
    print(f"  GPU  A10G GDDR6      : {A10G_BW_GBPS:6.1f} GB/s")
    print(f"  --> decode gap (GPU BW / CPU BW): {A10G_BW_GBPS/cpu_bw:5.1f}x\n")

    print("PREDICTION (the study's thesis):")
    print("  If decode is bandwidth-bound, CPU decode tok/s should be ~1/{:.0f} the GPU's,".format(A10G_BW_GBPS/cpu_bw))
    print("  i.e. roughly the bandwidth ratio, NOT the compute ratio.")
    print("  Prefill should show the much larger compute gap above.")
    print("  Quantization (fp16->Q4) cuts bytes/token ~4x, so it should lift CPU decode ~linearly")
    print("  until the CPU stops being bandwidth-bound and hits its compute/overhead floor.")

    # Weight-streaming decode estimate for Qwen3.5-4B
    print("\nDECODE back-of-envelope (Qwen3.5-4B):")
    for name, gb in [("fp16 (8.6 GB weights)", 8.6), ("Q8 (~4.3 GB)", 4.3), ("Q4 (~2.4 GB)", 2.4)]:
        cpu_tok = cpu_bw / gb
        gpu_tok = A10G_BW_GBPS / gb
        print(f"  {name:26s}: CPU ~{cpu_tok:6.1f} tok/s   GPU ~{gpu_tok:7.1f} tok/s   (pure BW ceiling, batch=1)")


if __name__ == "__main__":
    main()
