#!/bin/bash
# Part 9 gap closer, GPU side: same llama-batched-bench on the A10G via CUDA, so the
# batched CPU-vs-GPU comparison is same-engine. Waits for the CPU run to finish first.
set -u
cd /home/mpathak/code/research/vllm/cpu-inference-study
export PATH=/usr/local/cuda/bin:$PATH
for i in $(seq 1 60); do grep -q BATCHED_DONE results/batched_bench.md 2>/dev/null && break; sleep 10; done
cmake --build llama.cpp/build-cuda --target llama-batched-bench -j8 > results/batched_gpu_build.log 2>&1 || { echo BUILD_FAILED >> results/batched_bench_gpu.md; exit 1; }
B=llama.cpp/build-cuda/bin/llama-batched-bench
OUT=results/batched_bench_gpu.md
echo "# Batched decode on A10G via llama.cpp CUDA, -ngl 99 $(date -u +%FT%TZ)" > $OUT
echo "build: $(cd llama.cpp && git rev-parse --short HEAD)" >> $OUT
for M in Qwen_Qwen3.5-4B-F16 Qwen_Qwen3.5-4B-Q4_K_M; do
  echo "" >> $OUT; echo "## $M" >> $OUT
  LD_LIBRARY_PATH=llama.cpp/build-cuda/bin $B -m models/$M.gguf -ngl 99 -c 32768 -b 2048 -ub 512 \
    -npp 128 -ntg 128 -npl 1,2,4,8,16,32,64,128 -np 128 -t 8 2>results/batched_bench_gpu_$M.stderr | tee -a $OUT
done
echo "BATCHED_GPU_DONE $(date -u +%FT%TZ)" >> $OUT
