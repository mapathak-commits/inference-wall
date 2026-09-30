#!/usr/bin/env bash
# S1 driver: run three arms in sequence, one server each, sweep, stop, next.
set -uo pipefail
DIR=/home/mpathak/code/research/vllm/fa-study
PY=/var/tmp/vllm-study/venv/bin/python
LOG="$DIR/s1.log"

run_arm () {
  local model="$1" backend="$2" maxlen="$3" label="$4" port="$5"
  local srvlog="$DIR/server_s1_${label}.log"
  echo "[run_s1] $(date -u +%FT%TZ) ARM $label model=$model backend=$backend" | tee -a "$LOG"
  bash "$DIR/start_server.sh" "$model" "$backend" "$maxlen" "$port" > "$srvlog" 2>&1 &
  local srv=$!
  local ok=0
  for i in $(seq 1 108); do
    if curl -s -m 3 "http://localhost:$port/v1/models" | grep -q "$model"; then ok=1; break; fi
    if ! kill -0 $srv 2>/dev/null; then echo "[run_s1] $label server died early" | tee -a "$LOG"; break; fi
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    grep -iE "Using .* backend on V1" "$srvlog" | head -1 | tee -a "$LOG"
    $PY "$DIR/s1_probe.py" "$port" "$model" "$label" 2>&1 | tee -a "$LOG"
  else
    echo "[run_s1] $label NOT READY - tail:" | tee -a "$LOG"
    tail -30 "$srvlog" | tee -a "$LOG"
  fi
  kill $srv 2>/dev/null; sleep 6
  pkill -f "vllm serve $model" 2>/dev/null; sleep 4
  echo "[run_s1] $label done" | tee -a "$LOG"
}

echo "[run_s1] START $(date -u +%FT%TZ)" > "$LOG"
run_arm "Qwen/Qwen2.5-7B-Instruct" FLASH_ATTN     32768 "7b_flash" 8001
run_arm "Qwen/Qwen2.5-7B-Instruct" FLEX_ATTENTION 32768 "7b_flex"  8001
run_arm "Qwen/Qwen3.5-4B"          FLASH_ATTN     32768 "hy4b_flash" 8001
echo "[run_s1] ALL DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
