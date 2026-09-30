#!/usr/bin/env bash
# S3 driver: capture bounded profiler windows for FLASH and FLEX at 2k/8k/28k
# prefill and 2k/28k decode. Traces land in $TRACEDIR, gzipped after each arm.
set -uo pipefail
DIR=/home/mpathak/code/research/vllm/fa-study
PY=/var/tmp/vllm-study/venv/bin/python
LOG="$DIR/s3.log"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8001

run_arm () {
  local backend="$1" label="$2"
  local tdir="/var/tmp/vllm-study/traces_fa/$label"
  rm -rf "$tdir"; mkdir -p "$tdir"
  local srvlog="$DIR/server_s3_${label}.log"
  echo "[run_s3] $(date -u +%FT%TZ) ARM $label backend=$backend tracedir=$tdir" | tee -a "$LOG"
  # export profiler dir so vLLM writes chrome traces there
  VLLM_TORCH_PROFILER_DIR="$tdir" bash "$DIR/start_server.sh" "$MODEL" "$backend" 32768 "$PORT" > "$srvlog" 2>&1 &
  local srv=$!
  local ok=0
  for i in $(seq 1 108); do
    if curl -s -m 3 "http://localhost:$PORT/v1/models" | grep -q "$MODEL"; then ok=1; break; fi
    if ! kill -0 $srv 2>/dev/null; then echo "[run_s3] $label died early" | tee -a "$LOG"; break; fi
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    grep -iE "Using .* backend on V1" "$srvlog" | head -1 | tee -a "$LOG"
    # prefill windows
    for L in 2048 8192 28672; do
      $PY "$DIR/s3_capture.py" "$PORT" "$MODEL" "$label" prefill "$L" 2>&1 | tee -a "$LOG"
    done
    # decode windows (short generation atop context)
    for L in 2048 28672; do
      $PY "$DIR/s3_capture.py" "$PORT" "$MODEL" "$label" decode "$L" 30 2>&1 | tee -a "$LOG"
    done
  else
    echo "[run_s3] $label NOT READY - tail:" | tee -a "$LOG"; tail -30 "$srvlog" | tee -a "$LOG"
  fi
  kill $srv 2>/dev/null; sleep 6
  pkill -f "vllm serve $MODEL" 2>/dev/null; sleep 4
  # gzip traces to save disk, list sizes
  echo "[run_s3] $label traces:" | tee -a "$LOG"
  for f in "$tdir"/*.json; do [ -f "$f" ] && gzip -f "$f"; done
  ls -la "$tdir" 2>&1 | tee -a "$LOG"
}

echo "[run_s3] START $(date -u +%FT%TZ)" > "$LOG"
run_arm FLASH_ATTN     flash
run_arm FLEX_ATTENTION flex
echo "[run_s3] ALL DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
