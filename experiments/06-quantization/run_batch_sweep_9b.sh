#!/bin/bash
# Batch-size decode sweep against the running 9B-AWQ profiler server.
set -u
VENV=/var/tmp/vllm-study/venv-qwen35/bin/python
TDIR=/var/tmp/vllm-study/traces_9b
DIR=/home/mpathak/code/research/serving/qwen35
cd "$DIR"

for N in 1 8 64 150; do
  echo "########## 9B batch N=$N ##########"
  rm -f $TDIR/rank0.*.json.gz $TDIR/*.async_llm.*.json.gz
  timeout 60 $VENV capture_batch_sweep_9b.py $N 2>&1
  sleep 2
  f=$(ls -t $TDIR/rank0.*.json.gz 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    cp "$f" "$DIR/batch9b${N}.json.gz"
    echo "saved batch9b${N}.json.gz"
  else
    echo "NO TRACE for N=$N"
  fi
  grep -oE "Running: [0-9]+ reqs" "$DIR/steady_9b_server.log" | tail -3 | tr '\n' ' '
  echo
done
echo "=== 9B SWEEP DONE ==="
