#!/usr/bin/env bash
# Launch the llama.cpp CPU inference sweep detached. Run via sidecar target=host.
set -u
ROOT=/home/mpathak/code/research/vllm/cpu-inference-study
cd "$ROOT"
pkill -9 -f llama-bench 2>/dev/null || true
rm -f infer_bench.log
setsid nohup bash src/bench_infer.sh > infer_bench.log 2>&1 &
echo "LAUNCHED pid=$!"
