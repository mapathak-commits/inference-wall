#!/usr/bin/env python3
"""STREAM-style sustained memory bandwidth on the CPU, plus a thread-scaling sweep.

Decode streams weights through memory once per token, so *sustained* bandwidth (not the
DDR4 spec number) is the real decode ceiling. We measure the triad a = b + s*c over arrays
far larger than L3 (32 MiB) so every access misses cache and hits DRAM.

numpy vector ops are memory-bound and already multi-threaded through OpenBLAS/OMP, so we
sweep OMP_NUM_THREADS externally and read the peak. Reports GB/s.
"""
import os
import time
import json
import sys

import numpy as np


def triad_gbps(n_elems, iters=30):
    # float64 arrays; triad touches 3 arrays (read b, read c, write a) per element.
    a = np.zeros(n_elems, dtype=np.float64)
    b = np.random.rand(n_elems)
    c = np.random.rand(n_elems)
    s = 3.0
    # warm
    a[:] = b + s * c
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        np.add(b, c * s, out=a)   # read b, read c, write a
        dt = time.perf_counter() - t0
        bytes_moved = 3 * n_elems * 8  # 2 reads + 1 write, 8 bytes each
        best = max(best, bytes_moved / dt / 1e9)
    return best


def copy_gbps(n_elems, iters=30):
    a = np.random.rand(n_elems)
    b = np.empty_like(a)
    np.copyto(b, a)
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        np.copyto(b, a)  # read a, write b
        dt = time.perf_counter() - t0
        best = max(best, 2 * n_elems * 8 / dt / 1e9)
    return best


def main():
    # ~512 MiB per array => 3x = 1.5 GiB working set, far past 32 MiB L3.
    n = 64 * 1024 * 1024
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    res = {
        "threads_env": threads,
        "n_elems": n,
        "array_mib": n * 8 / 1024 / 1024,
        "triad_gbps": round(triad_gbps(n), 2),
        "copy_gbps": round(copy_gbps(n), 2),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()
