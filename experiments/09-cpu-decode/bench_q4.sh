#!/usr/bin/env bash
# Q4_K_M tier only, informative thread counts (1,2,4,8). The 16-thread decode case is
# skipped on purpose: bf16 already showed it collapses to ~0.2 tok/s (HT oversubscription
# on 8 physical cores), and that single cell takes ~30 min to run for no new insight.
set -u
ROOT=/home/mpathak/code/research/vllm/cpu-inference-study
cd "$ROOT"
BIN=llama.cpp/build-cpu/bin/llama-bench
OUT=results/infer_bench.md
echo "" >> "$OUT"
echo "## Qwen_Qwen3.5-4B-Q4_K_M.gguf" >> "$OUT"
"$BIN" -m models/Qwen_Qwen3.5-4B-Q4_K_M.gguf -t 1,2,4,8 -p 512 -n 128 -r 3 -o md >> "$OUT" 2>> results/infer_bench.err
echo "INFER_BENCH_DONE" >> "$OUT"
