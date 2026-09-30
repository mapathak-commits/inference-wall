#!/usr/bin/env python3
"""Kernel-family CUDA-time census over a torch/chrome trace (json or json.gz).

Sums device (GPU) kernel durations, buckets by family from the kernel name:
  attention : flash / splitkv / flex_attention / fmha / attn / softmax-in-attn
  gemm      : gemm / cutlass / matmul / *_mm / linear / moe
  other     : everything else (elementwise, norm, rope, copy, reduce, ...)

Prints family shares and the top kernels by total time. Also flags the
signature attention kernels so the FLASH-vs-FLEX contrast is visible by name.

Args: TRACE [TRACE ...]
"""
import gzip, json, sys, collections, os

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

ATTN = ("flash", "splitkv", "flex_attention", "fmha", "attention", "attn",
        "single_query", "paged")
GEMM = ("gemm", "cutlass", "matmul", "_mm", "wgmma", "s16816", "linear",
        "moe", "grouped")

def family(name):
    n = name.lower()
    if any(k in n for k in ATTN):
        return "attention"
    if any(k in n for k in GEMM):
        return "gemm"
    return "other"

def census(path):
    data = load(path)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    # device (GPU) kernels: torch marks cat == "kernel" with a device tid;
    # 0.11/torch2.8 uses cat "kernel" for CUDA kernels.
    by_family = collections.Counter()
    by_kernel = collections.Counter()
    total = 0.0
    for e in events:
        if e.get("cat") != "kernel":
            continue
        dur = e.get("dur", 0) or 0
        name = e.get("name", "")
        by_family[family(name)] += dur
        by_kernel[name] += dur
        total += dur
    print(f"\n=== {os.path.basename(path)} ===")
    if total == 0:
        print("  (no kernel events found)")
        return
    for fam in ("attention", "gemm", "other"):
        print(f"  {fam:>10}: {100*by_family[fam]/total:5.1f}%  "
              f"({by_family[fam]/1000:.1f} ms)")
    print(f"  {'total':>10}: {total/1000:.1f} ms")
    print("  top kernels:")
    for name, dur in by_kernel.most_common(6):
        tag = family(name)
        print(f"    [{tag:>9}] {100*dur/total:5.1f}%  {name[:70]}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
