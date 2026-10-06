#!/bin/bash
# Post 6 measurement session: speculative decoding on/off, single-stream + load sweep.
# Runs the full A/B unattended inside one tmux pane; writes everything to draft2/.
#
# Arms (one server per arm, serially):
#   off       -- canonical Part-1 config, no speculation (fresh baseline)
#   ngram-k5  -- + ngram speculation, num_speculative_tokens=5
#   ngram-k3  -- + ngram speculation, num_speculative_tokens=3 (depth knob)
#
# Per arm: warm-up pass, spec_probes.py (single-stream + 32-way concurrent),
# then the series' standard rate sweep (256/128, rates 1..inf, 200 prompts).
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1

DIR=/home/mpathak/code/research/vllm/draft2
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
PY=/var/tmp/vllm-study/venv-qwen35/bin/python
MODEL=Qwen/Qwen3.5-4B
RATES="1 2 4 6 8 16 inf"

start_server () {  # $1 tag, $2 extra args
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "=== [$1] starting server $(date -u +%H:%M:%S) ==="
  $BIN serve $MODEL \
    --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
    --gpu-memory-utilization 0.9 --trust-remote-code --port 8000 \
    $2 > $DIR/server_$1.log 2>&1 &
  for i in $(seq 1 90); do
    curl -s -o /dev/null -w "%{http_code}" localhost:8000/health 2>/dev/null \
      | grep -q 200 && { echo "=== [$1] server ready ==="; return 0; }
    sleep 10
  done
  echo "=== [$1] SERVER FAILED TO START ==="; return 1
}

sweep () {  # $1 tag
  for RATE in $RATES; do
    echo "########## [$1] request_rate=$RATE ##########"
    $BIN bench serve \
      --backend vllm --model $MODEL --host localhost --port 8000 \
      --dataset-name random --random-input-len 256 --random-output-len 128 \
      --num-prompts 200 --request-rate $RATE --ignore-eos --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      2>&1 | grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median ITL|P99 ITL|Median E2EL|P99 E2EL"
  done
}

run_arm () {  # $1 tag, $2 extra server args
  start_server "$1" "$2" || return 1
  # warm pass: touch every batch size the sweep will hit, then discard
  echo "=== [$1] warm pass ==="
  $BIN bench serve --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate inf --ignore-eos --seed 0 >/dev/null 2>&1
  echo "=== [$1] probes ==="
  $PY $DIR/spec_probes.py "$1"
  echo "=== [$1] sweep ==="
  sweep "$1"
  # capture the arm's spec metrics + memory report before the server dies
  grep -E "SpecDecoding metrics|KV cache size|Maximum concurrency|Available KV cache" \
    $DIR/server_$1.log | tail -8
}

run_arm off      ""
run_arm ngram-k5 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
run_arm ngram-k3 "--speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":3,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"

pkill -f "vllm serve" 2>/dev/null
echo "=== SPEC STUDY DONE $(date -u +%H:%M:%S) ==="
