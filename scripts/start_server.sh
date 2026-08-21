#!/bin/bash
# Launch vLLM's OpenAI-compatible server for Qwen3.5-4B in a detached tmux session.
# Uses the SEPARATE venv-qwen35 (vLLM 0.24.0, which registers Qwen3.5); the 0.11.0
# study env is left untouched so the two studies stay reproducible side by side.
# fp16, CUDA graphs on, max_num_seqs=256 -- matched to the 0.5B serving study so the
# only variable that changed is the model (bigger + hybrid attention).
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
# vLLM 0.18.0 on this box needs: spawn (forked EngineCore can't re-init CUDA) and
# the flashinfer version-check bypass (downgrade left cubin 0.6.12 vs python 0.6.6).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
MODEL=${1:-Qwen/Qwen3.5-4B}
LOG=${2:-/home/mpathak/code/research/serving/qwen35/server.log}

tmux kill-session -t vllmserve 2>/dev/null
sleep 2
tmux new-session -d -s vllmserve \
  "/var/tmp/vllm-study/venv-qwen35/bin/vllm serve $MODEL \
     --dtype float16 --max-model-len 2048 --max-num-seqs 256 \
     --gpu-memory-utilization 0.9 --trust-remote-code \
     --port 8000 \
     > $LOG 2>&1"
echo "launched tmux session vllmserve model=$MODEL"
sleep 2
tmux ls
