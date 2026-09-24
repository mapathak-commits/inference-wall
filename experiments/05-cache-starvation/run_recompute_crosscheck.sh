#!/bin/bash
# CROSS-CHECK — corroborate the preemption counter with an INDEPENDENT signal, and
# resolve the "why does a roomy cache preempt at all?" question with a timeline.
#
# Independent signal: RECOMPUTE preemption re-runs a request's prefill. So the total
# prompt tokens vLLM actually processes must EXCEED the sum of the requests' input
# lengths by roughly (preempted tokens re-prefilled). We scrape vllm:prompt_tokens_total
# (a different counter from vllm:num_preemptions_total) and compute the excess over the
# known input budget. If recompute is real, excess > 0 and grows with preemption count.
#
# Timeline: a background sampler curls /metrics every 3s into a per-arm timeline file
# (t preempt prompt_tok gen_tok running waiting), so we can see WHEN preemptions happen
# -- clustered at flood onset (transient over-commit) or spread across steady state.
#
# Two arms:
#   ROOMY_4B : 4B hybrid, natural (83.7x), 1024/512 fixed  -> the 17-preempt mystery
#   TIGHT_7B : 7B dense, 300-block (2.34x), 512/512 var    -> strongest recompute signal
#
# Usage: bash run_recompute_crosscheck.sh [nprompts]   (writes crosscheck.log here)
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
DIR=/home/mpathak/code/research/serving/qwen35
OUT=$DIR/crosscheck.log
: > "$OUT"
NP=${1:-200}

wait_ready () {
  for i in $(seq 1 72); do
    sleep 5
    curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && return 0
  done
  return 1
}

# scrape a single counter value by exact metric prefix (sums across engine labels)
scrape () {  # $1 = metric name (e.g. vllm:num_preemptions_total)
  curl -s http://localhost:8000/metrics 2>/dev/null \
    | grep -E "^$1(_total)?\{" | awk '{s+=$2} END{printf "%.0f\n", s+0}'
}

# background sampler: append "t preempt prompt gen running waiting" every 3s
sampler () {  # $1 = timeline file
  local TL="$1" t=0
  while true; do
    local P G R W PR
    PR=$(scrape vllm:num_preemptions)
    P=$(scrape vllm:prompt_tokens)
    G=$(scrape vllm:generation_tokens)
    R=$(curl -s http://localhost:8000/metrics 2>/dev/null | grep -E "^vllm:num_requests_running\{" | awk '{s+=$2} END{printf "%.0f\n", s+0}')
    W=$(curl -s http://localhost:8000/metrics 2>/dev/null | grep -E "^vllm:num_requests_waiting\{" | awk '{s+=$2} END{printf "%.0f\n", s+0}')
    echo "$t preempt=$PR prompt_tok=$P gen_tok=$G running=$R waiting=$W" >> "$TL"
    t=$((t+3)); sleep 3
  done
}

# $1 tag  $2 model  $3 blocks("" natural)  $4 in  $5 out  $6 range
run_arm () {
  local TAG="$1" MODEL="$2" BLOCKS="$3" IN="$4" O="$5" R="$6"
  local SLOG="$DIR/cc_server_${TAG}.log" BLOG="$DIR/cc_bench_${TAG}.log" TL="$DIR/cc_timeline_${TAG}.log"
  : > "$TL"
  local OVR=""; [ -n "$BLOCKS" ] && OVR="--num-gpu-blocks-override $BLOCKS"
  tmux kill-session -t vllmserve 2>/dev/null; sleep 3
  tmux new-session -d -s vllmserve \
    "$BIN serve $MODEL --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
       --gpu-memory-utilization 0.9 $OVR --trust-remote-code --port 8000 > $SLOG 2>&1"
  wait_ready || { echo "${TAG}_FAILED" | tee -a "$OUT"; tail -30 "$SLOG" | tee -a "$OUT"; return; }
  echo "########## $TAG ($MODEL blocks=${BLOCKS:-natural} in=$IN out=$O r=$R np=$NP) ##########" | tee -a "$OUT"
  grep -iE "GPU KV cache size|Maximum concurrency" "$SLOG" | tee -a "$OUT"

  local PRE0 PT0 GT0
  PRE0=$(scrape vllm:num_preemptions); PT0=$(scrape vllm:prompt_tokens); GT0=$(scrape vllm:generation_tokens)
  echo "BEFORE preempt=$PRE0 prompt_tok=$PT0 gen_tok=$GT0" | tee -a "$OUT"

  sampler "$TL" & local SPID=$!
  $BIN bench serve --backend vllm --model "$MODEL" --host localhost --port 8000 \
    --dataset-name random --random-input-len "$IN" --random-output-len "$O" \
    --random-range-ratio "$R" --ignore-eos --num-prompts $NP --request-rate inf --seed 0 \
    --percentile-metrics ttft,e2el --metric-percentiles 50,99 > "$BLOG" 2>&1
  kill $SPID 2>/dev/null

  local PRE1 PT1 GT1
  PRE1=$(scrape vllm:num_preemptions); PT1=$(scrape vllm:prompt_tokens); GT1=$(scrape vllm:generation_tokens)
  echo "AFTER  preempt=$PRE1 prompt_tok=$PT1 gen_tok=$GT1" | tee -a "$OUT"
  grep -iE "Successful|request throughput|Total input tokens|Total generated tokens|Median TTFT|P99 E2EL" "$BLOG" | tee -a "$OUT"

  # Independent cross-check: prompt tokens actually processed vs. the input budget.
  # Excess prefill (processed - budget) should be > 0 and track preemptions if recompute is real.
  local DPRE=$((PRE1 - PRE0)) DPT=$((PT1 - PT0))
  local BUDGET; BUDGET=$(grep -iE "Total input tokens" "$BLOG" | grep -oE "[0-9]+" | tail -1)
  echo "DELTA  preemptions=$DPRE prompt_tok_processed=$DPT input_budget=${BUDGET:-NA}" | tee -a "$OUT"
  if [ -n "${BUDGET:-}" ]; then
    echo "CROSSCHECK excess_prefill_tokens=$((DPT - BUDGET))  (should be >0 and grow with preemptions if RECOMPUTE)" | tee -a "$OUT"
  fi
  echo "-- $TAG timeline (preempt count over time; onset-clustered vs steady-state) --" | tee -a "$OUT"
  cat "$TL" | tee -a "$OUT"
}

run_arm ROOMY_4B Qwen/Qwen3.5-4B          ""  1024 512 0.0
run_arm TIGHT_7B Qwen/Qwen2.5-7B-Instruct 300 512  512 0.9

tmux kill-session -t vllmserve 2>/dev/null
echo "=== CROSSCHECK DONE ===" | tee -a "$OUT"
