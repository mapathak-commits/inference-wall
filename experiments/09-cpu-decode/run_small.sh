#!/bin/bash
# Part 9 follow-up: small models on the EPYC slice (CPU, cores 0-7) and the A10G (CUDA),
# same llama.cpp build as the 4B runs. Tests whether the CPU-to-GPU decode gap changes
# with model size and gives the per-engine fit short-lever points to pin the intercept.
set -u
cd /home/mpathak/code/research/vllm/cpu-inference-study
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
mkdir -p models/small
OUT=results/small_model_bench.md
dl () { hf download "$1" "$2" --local-dir models/small >> results/small_model_dl.log 2>&1 || huggingface-cli download "$1" "$2" --local-dir models/small >> results/small_model_dl.log 2>&1; }
dl Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf
dl Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf
dl Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf
dl Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf
dl Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf
echo "# Small-model decode: EPYC slice CPU vs A10G, llama.cpp $(cd llama.cpp && git rev-parse --short HEAD), $(date -u +%FT%TZ)" > $OUT
echo "" >> $OUT; echo "## File sizes (bytes)" >> $OUT
ls -l models/small/*.gguf | awk '{print "- " $9 ": " $5}' >> $OUT
CPU=llama.cpp/build-cpu/bin/llama-bench
GPU=llama.cpp/build-cuda/bin/llama-bench
for F in models/small/*.gguf; do
  echo "" >> $OUT; echo "## $(basename $F)" >> $OUT
  echo "### CPU, -ngl 0, 8 physical cores pinned" >> $OUT
  taskset -c 0-7 $CPU -m $F -ngl 0 -t 8 -C 0xFF --cpu-strict 1 -p 128 -n 128 -r 3 -o md >> $OUT 2>>results/small_model_bench.err
  echo "### A10G, -ngl 99" >> $OUT
  LD_LIBRARY_PATH=llama.cpp/build-cuda/bin $GPU -m $F -ngl 99 -p 128 -n 128 -r 3 -o md >> $OUT 2>>results/small_model_bench.err
done
echo "" >> $OUT; echo "## 4B F16 anchor, same session" >> $OUT
echo "### CPU" >> $OUT
taskset -c 0-7 $CPU -m models/Qwen_Qwen3.5-4B-F16.gguf -ngl 0 -t 8 -C 0xFF --cpu-strict 1 -p 128 -n 128 -r 3 -o md >> $OUT 2>>results/small_model_bench.err
echo "### A10G" >> $OUT
LD_LIBRARY_PATH=llama.cpp/build-cuda/bin $GPU -m models/Qwen_Qwen3.5-4B-F16.gguf -ngl 99 -p 128 -n 128 -r 3 -o md >> $OUT 2>>results/small_model_bench.err
echo "SMALL_DONE $(date -u +%FT%TZ)" >> $OUT
