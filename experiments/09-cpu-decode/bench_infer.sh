#!/bin/bash
# CPU inference benchmark: prefill (pp) and decode (tg) tok/s across the quant ladder
# and a thread sweep, using llama-bench (the canonical llama.cpp benchmark tool).
#
# pp512 = prefill throughput (process 512 prompt tokens) -> compute-bound
# tg128 = decode throughput (generate 128 tokens, batch=1) -> bandwidth-bound
#
# Output is llama-bench's own markdown table, appended to results/infer_bench.md.
set -e
cd /home/mpathak/code/research/vllm/cpu-inference-study
BIN=llama.cpp/build-cpu/bin/llama-bench
OUT=results/infer_bench.md

echo "# llama.cpp CPU inference bench — $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"
echo "" >> "$OUT"

for MODEL in models/Qwen_Qwen3.5-4B-bf16.gguf models/Qwen_Qwen3.5-4B-Q8_0.gguf models/Qwen_Qwen3.5-4B-Q4_K_M.gguf; do
  [ -f "$MODEL" ] || { echo "MISSING $MODEL, skipping" >> "$OUT"; continue; }
  echo "" >> "$OUT"
  echo "## $(basename "$MODEL")" >> "$OUT"
  # Thread sweep for decode; single thread-count for prefill+decode at the best thread count.
  # -t = threads, -p = prefill batch, -n = decode tokens, -r = repetitions
  "$BIN" -m "$MODEL" -t 1,2,4,8,16 -p 512 -n 128 -r 3 -o md >> "$OUT" 2>> results/infer_bench.err
done
echo "INFER_BENCH_DONE" >> "$OUT"
