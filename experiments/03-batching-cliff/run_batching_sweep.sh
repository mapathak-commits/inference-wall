#!/bin/bash
# Post 3 — the batching cliff. Sweep max_num_seqs (the running-batch cap) and, for
# each value, flood the server (--request-rate inf) with the fixed 256/128 workload
# and record sustained throughput. max_num_seqs is a server LAUNCH param, so each
# value needs a server restart; this driver does the whole sweep in one host run.
#
# Usage: bash run_batching_sweep.sh   (writes batching_sweep.log in this dir)
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
MODEL=Qwen/Qwen3.5-4B
DIR=/home/mpathak/code/research/serving/qwen35
OUT=$DIR/batching_sweep.log
: > "$OUT"

CAPS=${1:-"1 4 16 64 256"}
# Prompt count scales with the cap: at low caps throughput saturates almost
# immediately and draining 300 serial prompts wastes ~15 min for no extra signal.
# NP = max(48, 4*cap) keeps every point fast while still filling the batch at high caps.

wait_ready () {
  for i in $(seq 1 40); do
    sleep 5
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen; then
      return 0
    fi
  done
  return 1
}

for CAP in $CAPS; do
  echo "########## max_num_seqs=$CAP ##########" | tee -a "$OUT"
  tmux kill-session -t vllmserve 2>/dev/null; sleep 3
  tmux new-session -d -s vllmserve \
    "$BIN serve $MODEL --dtype float16 --max-model-len 2048 --max-num-seqs $CAP \
       --gpu-memory-utilization 0.9 --trust-remote-code --port 8000 \
       > $DIR/batch_server_$CAP.log 2>&1"
  if ! wait_ready; then echo "SERVER_FAILED cap=$CAP" | tee -a "$OUT"; continue; fi
  NP=$(( CAP*4 > 48 ? CAP*4 : 48 ))
  WARM=$(( CAP*2 > 16 ? CAP*2 : 16 ))
  # warmup (unmeasured) then a measured flood
  $BIN bench serve --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $WARM --request-rate inf --ignore-eos --seed 0 >/dev/null 2>&1
  $BIN bench serve --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate inf --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 2>&1 \
    | grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median ITL|P99 ITL|Median E2EL|P99 E2EL|Maximum request" \
    | tee -a "$OUT"
done
tmux kill-session -t vllmserve 2>/dev/null
echo "=== BATCHING SWEEP DONE ===" | tee -a "$OUT"
