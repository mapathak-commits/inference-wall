#!/usr/bin/env python3
"""OpenBLAS GEMM throughput at LLM-shaped matrices, with a thread sweep externally.

Two regimes matter for an LLM:
  - PREFILL: big matmuls (M = prompt_len tokens, K,N = hidden dims). Compute-bound; this is
    where the CPU's GFLOP ceiling bites.
  - DECODE: GEMV (M=1, one token). Bandwidth-bound; throughput is limited by streaming the
    weight matrix, so GFLOP/s looks terrible but that's the point.

Qwen3.5-4B rough dims: hidden ~2560, intermediate ~9728, layers ~36. We use representative
shapes. Reports GFLOP/s per shape.
"""
import os
import time
import json

import numpy as np


def gemm_gflops(m, k, n, dtype=np.float32, iters=20):
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.rand(k, n).astype(dtype)
    c = a @ b  # warm
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        c = a @ b
        dt = time.perf_counter() - t0
        flops = 2.0 * m * k * n
        best = max(best, flops / dt / 1e9)
    return round(best, 1)


def main():
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    H = 2560          # hidden
    I = 9728          # intermediate (FFN)
    shapes = {
        # prefill-like: many tokens through the FFN up-projection
        "prefill_ffn_up_M512":  (512, H, I),
        "prefill_attn_qkv_M512": (512, H, 3 * H),
        "prefill_ffn_up_M128":  (128, H, I),
        # decode-like: single token (GEMV), the bandwidth-bound case
        "decode_ffn_up_M1":     (1, H, I),
        "decode_attn_qkv_M1":   (1, H, 3 * H),
        # a big square for peak GFLOP reference
        "square_4096":          (4096, 4096, 4096),
    }
    out = {"threads_env": threads, "dtype": "fp32", "gflops": {}}
    for name, (m, k, n) in shapes.items():
        out["gflops"][name] = gemm_gflops(m, k, n)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
