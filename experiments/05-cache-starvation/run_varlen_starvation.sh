#!/bin/bash
# E2 (Part 4 follow-up) — can a VARIABLE-LENGTH workload force preemption?
# Part 4 / E1 both used fixed 1024/512 requests, which let vLLM size the running
# batch conservatively at admission, so it never over-committed and never preempted.
# Preemption is the recovery path for when already-admitted requests grow their KV
# faster than the batch was sized for. A high-variance output-length workload is the
# case most likely to create that: many requests admitted while short, some then
# running long, collectively out-growing the cache mid-flight.
#
# We run BOTH models (hybrid 4B, dense 7B) under a high-variance flood at a MODERATE
# cache cap (tight enough to pressure, not so tight admission collapses to a handful),
# and count preemptions. Same measured columns as run_starvation.sh / run_dense_natural.sh.
#
# Usage: bash run_varlen_starvation.sh [nprompts]  (writes varlen.log here)
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
DIR=/home/mpathak/code/research/serving/qwen35
OUT=$DIR/varlen.log
: > "$OUT"

# range-ratio must be in [0,1); it spreads each length over [len*(1-r), len*(1+r)].
# With r=0.9 and len=512, both input and output span ~51..973 tokens, so even the
# worst-case request (973+973) stays under --max-model-len 2048. --ignore-eos keeps
# each request running to its full sampled output length so KV actually grows
# (without it the model emits EOS early and KV never builds up — no preemption test).
IN_LEN=512
OUT_LEN=512
NP=${1:-200}
RANGE=0.9

wait_ready () {
  for i in $(seq 1 60); do
    sleep 5
    curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && return 0
  done
  return 1
}

bench () {  # $1 = tag, $2 = model
  echo "########## $1 ##########" | tee -a "$OUT"
  local BLOG="$DIR/bench_${1// /_}.log"
  $BIN bench serve --backend vllm --model "$2" --host localhost --port 8000 \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --random-range-ratio $RANGE --ignore-eos --num-prompts 16 --request-rate inf --seed 0 \
    > "$BLOG" 2>&1
  # Full bench output to a per-arm log so a crash/argparse error is never swallowed.
  $BIN bench serve --backend vllm --model "$2" --host localhost --port 8000 \
    --dataset-name random --random-input-len $IN_LEN --random-output-len $OUT_LEN \
    --random-range-ratio $RANGE --ignore-eos --num-prompts $NP --request-rate inf --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    > "$BLOG" 2>&1
  grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median ITL|P99 ITL|Median E2EL|P99 E2EL|Maximum request" \
    "$BLOG" | tee -a "$OUT"
  # If nothing matched, surface the tail so the failure is visible, not silent.
  grep -qiE "request throughput" "$BLOG" || { echo "!! BENCH PRODUCED NO METRICS — tail of $BLOG:" | tee -a "$OUT"; tail -15 "$BLOG" | tee -a "$OUT"; }
}

run_one () {  # $1 = model, $2 = blocks-override ("" = natural), $3 = server-log, $4 = tag
  local MODEL="$1" BLOCKS="$2" SLOG="$DIR/$3" TAG="$4"
  tmux kill-session -t vllmserve 2>/dev/null; sleep 3
  local OVR=""; [ -n "$BLOCKS" ] && OVR="--num-gpu-blocks-override $BLOCKS"
  tmux new-session -d -s vllmserve \
    "$BIN serve $MODEL --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
       --gpu-memory-utilization 0.9 $OVR --trust-remote-code --port 8000 \
       > $SLOG 2>&1"
  wait_ready || { echo "${TAG}_FAILED" | tee -a "$OUT"; tail -30 "$SLOG" | tee -a "$OUT"; return; }
  echo "-- $TAG memory split --" | tee -a "$OUT"
  grep -iE "Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency" "$SLOG" | tee -a "$OUT"
  bench "$TAG" "$MODEL"
  echo "-- $TAG preemption evidence --" | tee -a "$OUT"
  grep -icE "preempt" "$SLOG" | sed 's/^/PREEMPTION_LOG_LINES /' | tee -a "$OUT"
  grep -iE "preempt" "$SLOG" | head -3 | tee -a "$OUT"
  echo "-- $TAG admission evidence (last Running/Waiting) --" | tee -a "$OUT"
  grep -iE "Running:.*Waiting:.*GPU KV cache usage" "$SLOG" | tail -8 | tee -a "$OUT"
}

# Hybrid 4B: 40 blocks was the Part 4 severe cap; use it so varlen is the only change.
run_one Qwen/Qwen3.5-4B 40 varlen_server_4b40.log "VARLEN 4B hybrid (40-block cap)"
# Dense 7B: cap KV to ~300 blocks (~a few thousand tokens) to pressure it like the 4B severe arm.
run_one Qwen/Qwen2.5-7B-Instruct 300 varlen_server_7b300.log "VARLEN 7B dense (300-block cap)"
# Dense 7B: natural pressure (no override) with variance, to compare against E1's fixed-length.
run_one Qwen/Qwen2.5-7B-Instruct "" varlen_server_7bnat.log "VARLEN 7B dense (natural KV)"

tmux kill-session -t vllmserve 2>/dev/null
echo "=== VARLEN DONE ===" | tee -a "$OUT"
