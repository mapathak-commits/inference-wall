#!/bin/bash
# E3 — POSITIVE CONTROL: prove the harness CAN detect preemption, and map the boundary.
#
# Source fact (v1/core/sched/scheduler.py): admission allocates blocks for the PROMPT only
# (num_new_tokens = request.num_tokens - num_computed_tokens); it does NOT reserve the eventual
# output length. Preemption (line ~462) fires when an ALREADY-RUNNING request needs a block
# mid-decode and allocate_slots() returns None (pool empty).
#
# => The case that forces preemption is CHEAP ADMISSION + LARGE DECODE GROWTH:
#    short prompt (many requests admitted before max_num_seqs / KV gate bites) + long output
#    (they all grow during decode and collectively outrun the pool). E1/E2 used balanced
#    512/512, so admission cost ~= final footprint and the running set was gated small at entry
#    -- which is WHY they never preempted. This script tests the opposite corner.
#
# Three arms on the SAME dense 7B server config, cache capped tight, varying only the workload:
#   POS : in=64,  out=2000  -> cheap admission, huge growth   -> EXPECT preemption > 0
#   NEG : in=2000, out=64   -> expensive admission, tiny growth-> EXPECT preemption == 0
#   BAL : in=512,  out=512  -> E2's balanced case, same cap    -> EXPECT preemption == 0 (repro)
# If POS fires and NEG/BAL don't, the detector is proven AND the mechanism claim is demonstrated,
# not just reasoned. If POS does NOT fire, our understanding is wrong -> do not publish.
#
# Preemption is read TWO ways for cross-checking:
#   (1) the server-log "Preemptions: N" clause (loggers.py only appends it when N>0), and
#   (2) the Prometheus counter vllm:num_preemptions scraped from /metrics (authoritative).
#
# Usage: bash run_preempt_positive_control.sh [nprompts]   (writes preempt_pc.log here)
set -u
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
DIR=/home/mpathak/code/research/serving/qwen35
OUT=$DIR/preempt_pc.log
: > "$OUT"

MODEL=Qwen/Qwen2.5-7B-Instruct
BLOCKS=300          # same tight cap as E2's 7b300 arm: 4,800 KV tokens, 2.34x @ 2048
MAXLEN=2100         # must exceed the 2000-token POS output
NP=${1:-120}

wait_ready () {
  for i in $(seq 1 60); do
    sleep 5
    curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && return 0
  done
  return 1
}

# Authoritative preemption counter from Prometheus /metrics (not a log grep).
scrape_preempt () {  # prints the integer value of vllm:num_preemptions (0 if absent)
  curl -s http://localhost:8000/metrics 2>/dev/null \
    | grep -E "^vllm:num_preemptions(_total)?\{" \
    | awk '{s+=$2} END{printf "%d\n", s+0}'
}

bench () {  # $1 = tag, in=$2 out=$3
  local TAG="$1" IN="$2" O="$3" BLOG="$DIR/bench_pc_${1}.log"
  echo "########## $TAG (in=$IN out=$O) ##########" | tee -a "$OUT"
  $BIN bench serve --backend vllm --model "$MODEL" --host localhost --port 8000 \
    --dataset-name random --random-input-len "$IN" --random-output-len "$O" \
    --random-range-ratio 0.0 --ignore-eos --num-prompts $NP --request-rate inf --seed 0 \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,99 \
    > "$BLOG" 2>&1
  grep -iE "Successful|request throughput|output token throughput|Median TTFT|P99 TTFT|Median E2EL|P99 E2EL" \
    "$BLOG" | tee -a "$OUT"
  grep -qiE "request throughput" "$BLOG" || { echo "!! NO METRICS — tail:" | tee -a "$OUT"; tail -15 "$BLOG" | tee -a "$OUT"; }
}

run_arm () {  # $1=tag  $2=in  $3=out
  local TAG="$1" IN="$2" O="$3" SLOG="$DIR/pc_server_${1}.log"
  tmux kill-session -t vllmserve 2>/dev/null; sleep 3
  tmux new-session -d -s vllmserve \
    "$BIN serve $MODEL --dtype float16 --max-model-len $MAXLEN --max-num-seqs 256 \
       --gpu-memory-utilization 0.9 --num-gpu-blocks-override $BLOCKS --trust-remote-code \
       --port 8000 > $SLOG 2>&1"
  wait_ready || { echo "${TAG}_FAILED" | tee -a "$OUT"; tail -30 "$SLOG" | tee -a "$OUT"; return; }
  echo "-- $TAG memory split --" | tee -a "$OUT"
  grep -iE "GPU KV cache size|Maximum concurrency" "$SLOG" | tee -a "$OUT"
  local P0; P0=$(scrape_preempt)
  echo "PREEMPT_COUNTER_BEFORE $P0" | tee -a "$OUT"
  bench "$TAG" "$IN" "$O"
  local P1; P1=$(scrape_preempt)
  echo "PREEMPT_COUNTER_AFTER  $P1" | tee -a "$OUT"
  echo "PREEMPT_COUNTER_DELTA  $((P1 - P0))" | tee -a "$OUT"
  echo "-- $TAG server-log Preemptions clause (should appear iff >0) --" | tee -a "$OUT"
  grep -iE "Preemptions:" "$SLOG" | tail -3 | tee -a "$OUT"
  grep -icE "Preemptions:" "$SLOG" | sed 's/^/PREEMPT_LOG_LINES /' | tee -a "$OUT"
  echo "-- $TAG last Running/Waiting/KV --" | tee -a "$OUT"
  grep -iE "Running:.*Waiting:.*GPU KV cache usage" "$SLOG" | tail -6 | tee -a "$OUT"
}

run_arm POS 64   2000    # expect preemption > 0
run_arm NEG 2000 64      # expect preemption == 0
run_arm BAL 512  512     # expect preemption == 0 (E2 repro on same server)

tmux kill-session -t vllmserve 2>/dev/null
echo "=== PC DONE ===" | tee -a "$OUT"
