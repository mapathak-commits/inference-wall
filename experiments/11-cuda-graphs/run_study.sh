#!/bin/bash
# Part 11 study: CUDA graphs on (vLLM default) vs off (--enforce-eager), vLLM 0.18, A10G.
# Phase 1 measures speed with no profiler; phase 2 captures traces with the profiler armed.
set -u
cd /home/mpathak/code/research/vllm/cudagraph-study
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:/usr/local/cuda/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
PY=/var/tmp/vllm-study/venv-qwen35/bin/python
OUT=results/study.log; mkdir -p results traces; echo "=== CUDA GRAPH STUDY START $(date -u +%FT%TZ) ===" > $OUT

start_server () {  # $1 model $2 arm $3 tag $4 extra
  pkill -f "vllm serve" 2>/dev/null; sleep 6
  local extra="$4"; [ "$2" = eager ] && extra="$extra --enforce-eager"
  local t0=$(date +%s)
  $BIN serve $1 --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 --trust-remote-code --port 8000 $extra > results/server_$3.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s -o /dev/null -w "%{http_code}" localhost:8000/health 2>/dev/null | grep -q 200 && {
      echo "[$3] ready after $(( $(date +%s) - t0 )) s" | tee -a $OUT; return 0; }
    sleep 5
  done
  echo "[$3] SERVER FAILED TO START" | tee -a $OUT; return 1
}

bench () {  # $1 model $2 tag $3 concurrency $4 prompts $5 in $6 out [$7 rate]
  echo "########## [$2] conc=$3 prompts=$4 in=$5 out=$6 rate=${7:-inf} ##########" >> $OUT
  local extra=""; [ "$3" != "none" ] && extra="--max-concurrency $3"
  $BIN bench serve --backend vllm --model $1 --host localhost --port 8000 \
    --dataset-name random --random-input-len $5 --random-output-len $6 --num-prompts $4 \
    --request-rate ${7:-inf} $extra --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 2>&1 \
    | grep -iE "Successful|output token throughput|Median TTFT|Median TPOT|P99 TPOT|Median ITL|P99 ITL" >> $OUT
}

# ---------- phase 1: speed ----------
for MODEL in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen3.5-4B; do
  short=$(basename $MODEL)
  for ARM in graph eager; do
    tag=${short}_$ARM
    start_server $MODEL $ARM $tag "" || continue
    grep -hE "Graph capturing finished|init engine|torch.compile takes|Capturing CUDA graph" results/server_$tag.log | tail -4 | cut -c1-200 >> $OUT
    bench $MODEL WARMUP_$tag 64 128 128 128   # warm pass; WARMUP_ lines are ignored in analysis
    bench $MODEL $tag 1 16 128 256
    bench $MODEL $tag 4 64 128 256
    bench $MODEL $tag 16 128 128 256
    bench $MODEL $tag 64 256 128 256
    [ "$short" = "Qwen3.5-4B" ] && bench $MODEL $tag none 200 256 128 inf
  done
done
pkill -f "vllm serve" 2>/dev/null; sleep 6

# ---------- phase 2: traces, profiler armed ----------
for MODEL in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3.5-4B; do
  short=$(basename $MODEL)
  for ARM in graph eager; do
    tag=trace_${short}_$ARM; d=$(pwd)/traces/${short}_$ARM; mkdir -p $d
    start_server $MODEL $ARM $tag "--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$d" || continue
    $PY capture_trace.py $MODEL >> $OUT 2>&1
    sleep 15; ls -la $d | tail -3 >> $OUT
  done
done
pkill -f "vllm serve" 2>/dev/null
echo "=== CUDA GRAPH STUDY DONE $(date -u +%FT%TZ) ===" >> $OUT
