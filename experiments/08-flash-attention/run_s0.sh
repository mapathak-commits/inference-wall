#!/usr/bin/env bash
# Self-contained S0 driver: start 7B server (FLASH_ATTN, 32k), wait warm, sweep, stop.
set -uo pipefail
cd /home/mpathak/code/research/vllm/fa-study
DIR=/home/mpathak/code/research/vllm/fa-study
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8001
PY=/var/tmp/vllm-study/venv/bin/python

echo "[run_s0] $(date -u +%FT%TZ) launching server" | tee "$DIR/s0.log"
bash "$DIR/start_server.sh" "$MODEL" FLASH_ATTN 32768 $PORT > "$DIR/server_s0_7b.log" 2>&1 &
SRV=$!
echo "[run_s0] server pid=$SRV" | tee -a "$DIR/s0.log"

# wait up to 8 min for readiness
ok=0
for i in $(seq 1 96); do
  if curl -s -m 3 "http://localhost:$PORT/v1/models" | grep -q "$MODEL"; then ok=1; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[run_s0] server died early" | tee -a "$DIR/s0.log"; break; fi
  sleep 5
done

if [ "$ok" = 1 ]; then
  echo "[run_s0] server ready, verifying backend line" | tee -a "$DIR/s0.log"
  grep -iE "Using .* backend|attention backend" "$DIR/server_s0_7b.log" | head -3 | tee -a "$DIR/s0.log"
  echo "[run_s0] running sweep" | tee -a "$DIR/s0.log"
  $PY "$DIR/s0_prefill_sweep.py" $PORT "$MODEL" 2>&1 | tee -a "$DIR/s0.log"
else
  echo "[run_s0] NOT READY - dumping server tail" | tee -a "$DIR/s0.log"
  tail -40 "$DIR/server_s0_7b.log" | tee -a "$DIR/s0.log"
fi

echo "[run_s0] stopping server pid=$SRV" | tee -a "$DIR/s0.log"
kill $SRV 2>/dev/null
sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null
echo "[run_s0] $(date -u +%FT%TZ) DONE" | tee -a "$DIR/s0.log"
