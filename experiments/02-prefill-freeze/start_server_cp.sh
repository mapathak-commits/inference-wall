#!/bin/bash
# Start the Qwen3.5-4B server for the chunked-prefill probe. Arg1 = on|off.
# Larger max-model-len (8192) so the long-prompt injection is a genuinely heavy
# prefill. vLLM enables chunked prefill by default; --no-enable-chunked-prefill
# turns it off. This re-tests, on a hybrid-attention model, the 0.5B null result:
# is chunked prefill still a V1-default no-op when 3/4 of the layers are linear?
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FLASHINFER_DISABLE_VERSION_CHECK=1
MODE=$1
MODEL=${2:-Qwen/Qwen3.5-4B}

FLAG="--enable-chunked-prefill"
[ "$MODE" = "off" ] && FLAG="--no-enable-chunked-prefill"

tmux kill-session -t vllmserve 2>/dev/null
sleep 2
tmux new-session -d -s vllmserve \
  "/var/tmp/vllm-study/venv-qwen35/bin/vllm serve $MODEL \
     --dtype float16 --max-model-len 8192 --max-num-seqs 256 $FLAG \
     --gpu-memory-utilization 0.9 --trust-remote-code \
     --port 8000 \
     > /home/mpathak/code/research/serving/qwen35/server_cp_$MODE.log 2>&1"
echo "launched vllmserve chunked_prefill=$MODE model=$MODEL"
