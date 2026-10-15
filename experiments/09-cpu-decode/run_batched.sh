#!/bin/bash
# Part 9 gap closer: batched decode throughput on the EPYC slice, physical cores 0-7.
# llama-batched-bench: npp=128 shared prompt, ntg=128 generated per sequence,
# npl = number of parallel sequences (the batch). Reports per-cell generation tok/s.
set -u
cd /home/mpathak/code/research/vllm/cpu-inference-study
B=llama.cpp/build-cpu/bin/llama-batched-bench
OUT=results/batched_bench.md
echo "# Batched decode on EPYC slice (g5.4xlarge, 8 phys cores pinned) $(date -u +%FT%TZ)" > $OUT
echo "build: $(cd llama.cpp && git rev-parse --short HEAD)" >> $OUT
for M in Qwen_Qwen3.5-4B-bf16 Qwen_Qwen3.5-4B-Q4_K_M; do
  echo "" >> $OUT; echo "## $M" >> $OUT
  taskset -c 0-7 $B -m models/$M.gguf -c 8192 -b 2048 -ub 512 -npp 128 -ntg 128 \
    -npl 1,2,4,8,16,32 -np 32 -t 8 -tb 8 2>results/batched_bench_$M.stderr | tee -a $OUT
done
echo "BATCHED_DONE $(date -u +%FT%TZ)" >> $OUT
