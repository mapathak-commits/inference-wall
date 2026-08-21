#!/bin/bash
# Launch Qwen3.5-4B with the torch profiler enabled, for steady-state decode capture.
# Mirrors start_server.sh but adds VLLM_TORCH_PROFILER_DIR so /start_profile works.
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=INFO
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_TORCH_PROFILER_DIR=/var/tmp/vllm-study/traces_steady
DIR=/home/mpathak/code/research/serving/qwen35
mkdir -p /var/tmp/vllm-study/traces_steady

tmux kill-session -t vllmprof 2>/dev/null
sleep 2
# NB: env vars must be set INSIDE the tmux command string; a new tmux session does
# not inherit this script's exported env. VLLM_TORCH_PROFILER_DIR is what registers
# the /start_profile route.
tmux new-session -d -s vllmprof \
  "export HF_HOME=/var/tmp/vllm-study/hf-cache VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1 PATH=/var/tmp/vllm-study/venv-qwen35/bin:\$PATH; \
   /var/tmp/vllm-study/venv-qwen35/bin/vllm serve Qwen/Qwen3.5-4B \
     --dtype float16 --max-model-len 8192 --max-num-seqs 256 \
     --gpu-memory-utilization 0.9 --trust-remote-code --port 8000 \
     --profiler-config.profiler=torch \
     --profiler-config.torch_profiler_dir=/var/tmp/vllm-study/traces_steady \
     > $DIR/steady_prof_server.log 2>&1"
echo "launched tmux session vllmprof"
sleep 2
tmux ls
