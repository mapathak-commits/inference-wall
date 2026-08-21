#!/bin/bash
# Drive the batch-size decode sweep against the running profiler server.
# For each N: clear the trace dir, run the capture, save the rank0 trace as batch<N>.json.gz.
set -u
VENV=/var/tmp/vllm-study/venv-qwen35/bin/python
TDIR=/var/tmp/vllm-study/traces_steady
DIR=/home/mpathak/code/research/serving/qwen35
cd "$DIR"

for N in 1 8 64 150; do
  echo "########## batch N=$N ##########"
  rm -f $TDIR/rank0.*.json.gz $TDIR/*.async_llm.*.json.gz
  timeout 60 $VENV capture_batch_sweep.py $N 2>&1
  sleep 2
  f=$(ls -t $TDIR/rank0.*.json.gz 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp "$f" "$DIR/batch${N}.json.gz"
    echo "saved batch${N}.json.gz"
  else
    echo "NO TRACE for N=$N"
  fi
  # capture the actual running batch the server saw, from its log
  grep -oE "Running: [0-9]+ reqs" "$DIR/steady_prof_server.log" | tail -3 | tr '\n' ' '
  echo
done
echo "=== SWEEP DONE ==="
