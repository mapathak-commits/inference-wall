#!/bin/bash
# REMEASURE — re-run every arm that appears in the E1/E2 results tables, counting
# preemptions with the AUTHORITATIVE Prometheus counter (vllm:num_preemptions_total)
# instead of the server-log "Preemptions:" clause.
#
# Why: the log clause is DEAD CODE in vLLM 0.18.0. loggers.py log() calls
# _update_stats() -> _reset() which zeroes self.num_preemptions BEFORE the
# `if self.num_preemptions > 0: append("Preemptions: %d")` check, and (unlike
# throughput) keeps no last_* copy. So the clause can never print, and every
# "0 preemptions" E1/E2 reported by grepping the log is a false negative.
# The positive control already proved the counter fires (POS=205, NEG=19, BAL=36)
# on the tight 300-block cap. This script gets counter numbers for ALL arms,
# especially the natural-cache ones whose true preemption count is still unknown.
#
# Arms (match the original workloads exactly):
#   R4B_NAT  Qwen3.5-4B      natural      in=1024 out=512  fixed   (Part 4 roomy)
#   E1_7B_NAT Qwen2.5-7B     natural      in=1024 out=512  fixed   (E1)
#   E2_4B_40  Qwen3.5-4B     40-block     in=512  out=512  r=0.9   (E2 tight hybrid)
#   E2_7B_300 Qwen2.5-7B     300-block    in=512  out=512  r=0.9   (E2 tight dense)
#   E2_7B_NAT Qwen2.5-7B     natural      in=512  out=512  r=0.9   (E2 natural dense)
#
# Usage: bash run_counter_remeasure.sh [nprompts]   (writes remeasure.log here)
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
DIR=/home/mpathak/code/research/serving/qwen35
OUT=$DIR/remeasure.log
: > "$OUT"
NP=${1:-200}

wait_ready () {
  for i in $(seq 1 72); do
    sleep 5
    curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && return 0
  done
  return 1
}

scrape_preempt () {  # authoritative counter value (0 if absent)
  curl -s http://localhost:8000/metrics 2>/dev/null \
    | grep -E "^vllm:num_preemptions(_total)?\{" \
    | awk '{s+=$2} END{printf "%d\n", s+0}'
}

# $1 tag  $2 model  $3 blocks("" natural)  $4 maxlen  $5 in  $6 out  $7 range
run_arm () {
  local TAG="$1" MODEL="$2" BLOCKS="$3" MAXLEN="$4" IN="$5" O="$6" R="$7"
  local SLOG="$DIR/rm_server_${TAG}.log" BLOG="$DIR/rm_bench_${TAG}.log"
  local OVR=""; [ -n "$BLOCKS" ] && OVR="--num-gpu-blocks-override $BLOCKS"
  tmux kill-session -t vllmserve 2>/dev/null; sleep 3
  tmux new-session -d -s vllmserve \
    "$BIN serve $MODEL --dtype float16 --max-model-len $MAXLEN --max-num-seqs 256 \
       --gpu-memory-utilization 0.9 $OVR --trust-remote-code --port 8000 > $SLOG 2>&1"
  wait_ready || { echo "${TAG}_FAILED" | tee -a "$OUT"; tail -30 "$SLOG" | tee -a "$OUT"; return; }
  echo "########## $TAG ($MODEL blocks=${BLOCKS:-natural} in=$IN out=$O r=$R) ##########" | tee -a "$OUT"
  grep -iE "GPU KV cache size|Maximum concurrency" "$SLOG" | tee -a "$OUT"
  local P0; P0=$(scrape_preempt)
  echo "PREEMPT_COUNTER_BEFORE $P0" | tee -a "$OUT"
  $BIN bench serve --backend vllm --model "$MODEL" --host localhost --port 8000 \
    --dataset-name random --random-input-len "$IN" --random-output-len "$O" \
    --random-range-ratio "$R" --ignore-eos --num-prompts $NP --request-rate inf --seed 0 \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 > "$BLOG" 2>&1
  grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median E2EL|P99 E2EL" \
    "$BLOG" | tee -a "$OUT"
  grep -qiE "request throughput" "$BLOG" || { echo "!! NO METRICS — tail:" | tee -a "$OUT"; tail -15 "$BLOG" | tee -a "$OUT"; }
  local P1; P1=$(scrape_preempt)
  echo "PREEMPT_COUNTER_AFTER  $P1" | tee -a "$OUT"
  echo "PREEMPT_COUNTER_DELTA  $((P1 - P0))" | tee -a "$OUT"
  echo "-- $TAG last Running/Waiting/KV --" | tee -a "$OUT"
  grep -iE "Running:.*Waiting:.*GPU KV cache usage" "$SLOG" | tail -4 | tee -a "$OUT"
}

run_arm R4B_NAT   Qwen/Qwen3.5-4B          ""  2048 1024 512 0.0
run_arm E1_7B_NAT Qwen/Qwen2.5-7B-Instruct ""  2048 1024 512 0.0
run_arm E2_4B_40  Qwen/Qwen3.5-4B          40  2048 512  512 0.9
run_arm E2_7B_300 Qwen/Qwen2.5-7B-Instruct 300 2048 512  512 0.9
run_arm E2_7B_NAT Qwen/Qwen2.5-7B-Instruct ""  2048 512  512 0.9

tmux kill-session -t vllmserve 2>/dev/null
echo "=== REMEASURE DONE ===" | tee -a "$OUT"
