#!/bin/bash
# Post 6 follow-up session: the three gaps before drafting.
#   A. stability re-run of the headline flood point, off vs ngram-k5 (2x each)
#   B. knee-locating points for ngram-k5 (rate 3, plus rate 4 repeat)
#   C. torch-profiler trace of steady decode WITH speculation armed (ngram-k5),
#      predictable-text decoders so verify steps carry accepted tokens.
# Writes to draft2/: spec_followup.log, server_fu_*.log, spec_verify_trace dir listing.
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1

DIR=/home/mpathak/code/research/vllm/draft2
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
PY=/var/tmp/vllm-study/venv-qwen35/bin/python
MODEL=Qwen/Qwen3.5-4B
TDIR=/var/tmp/vllm-study/traces_spec
mkdir -p $TDIR

SPEC5='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'

start_server () {  # $1 tag, $2 extra args
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "=== [$1] starting server $(date -u +%H:%M:%S) ==="
  $BIN serve $MODEL \
    --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 --trust-remote-code --port 8000 \
    $2 > $DIR/server_fu_$1.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" localhost:8000/health 2>/dev/null \
      | grep -q 200 && { echo "=== [$1] server ready ==="; return 0; }
    sleep 10
  done
  echo "=== [$1] SERVER FAILED ==="; return 1
}

bench () {  # $1 tag, $2 rate
  echo "########## [$1] request_rate=$2 ##########"
  $BIN bench serve \
    --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate $2 --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    2>&1 | grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median E2EL|P99 E2EL"
}

warm () {
  $BIN bench serve --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf --ignore-eos --seed 0 >/dev/null 2>&1
}

# ---- A+B on the ngram-k5 server (also C afterwards, same server) ----
start_server k5 "$SPEC5 --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$TDIR" || exit 1
echo "=== [k5] warm ==="; warm
bench k5-stability inf
bench k5-stability inf
bench k5-knee 3
bench k5-knee 4
echo "=== [k5] trace capture (predictable decoders, spec armed) ==="
$PY $DIR/capture_spec_trace.py
sleep 5
ls -la $TDIR | tail -5

# ---- A on the off server ----
start_server off "" || exit 1
echo "=== [off] warm ==="; warm
bench off-stability inf
bench off-stability inf

pkill -f "vllm serve" 2>/dev/null
echo "=== FOLLOWUP DONE $(date -u +%H:%M:%S) ==="
