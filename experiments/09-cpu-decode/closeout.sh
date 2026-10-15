#!/bin/bash
# Closeout experiments:
#  A. AMD CPU byte-identical F16 vs bf16 A/B (settles the §8c bf16-kernel-vs-memory question)
#  B. A10G Q4_K_M decode (replaces the ~150 tok/s estimate with a real number)
set -u
cd /home/mpathak/code/research/vllm/cpu-inference-study
CPU=llama.cpp/build-cpu/bin/llama-bench
CUDA=llama.cpp/build-cuda/bin/llama-bench
OUT=results/closeout_ab.md
echo "# Closeout A/B — $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"

echo "" >> "$OUT"; echo "## A. AMD CPU: F16 vs bf16, byte-identical (8,665,620,192 B each), -t 8" >> "$OUT"
echo "### bf16 (round 1)" >> "$OUT"
$CPU -m models/Qwen_Qwen3.5-4B-bf16.gguf -t 8 -p 0 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err
echo "### F16 (round 1)" >> "$OUT"
$CPU -m models/Qwen_Qwen3.5-4B-F16.gguf  -t 8 -p 0 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err
echo "### bf16 (round 2)" >> "$OUT"
$CPU -m models/Qwen_Qwen3.5-4B-bf16.gguf -t 8 -p 0 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err
echo "### F16 (round 2)" >> "$OUT"
$CPU -m models/Qwen_Qwen3.5-4B-F16.gguf  -t 8 -p 0 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err

echo "" >> "$OUT"; echo "## B. A10G GPU: Q4_K_M and F16 decode+prefill, -ngl 99" >> "$OUT"
LD_LIBRARY_PATH=llama.cpp/build-cuda/bin $CUDA -m models/Qwen_Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err
LD_LIBRARY_PATH=llama.cpp/build-cuda/bin $CUDA -m models/Qwen_Qwen3.5-4B-F16.gguf   -ngl 99 -p 512 -n 128 -r 3 -o md >> "$OUT" 2>> results/closeout.err

echo "" >> "$OUT"; echo "CLOSEOUT_DONE" >> "$OUT"
