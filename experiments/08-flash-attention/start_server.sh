#!/usr/bin/env bash
# Launch a vLLM server for the FA long-context study. Args: MODEL BACKEND MAXLEN PORT
set -uo pipefail
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
BACKEND="${2:-FLASH_ATTN}"
MAXLEN="${3:-32768}"
PORT="${4:-8001}"

export HF_HOME=/var/tmp/vllm-study/hf-cache
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="$BACKEND"   # 0.11.0: backend selected by env var, not a CLI flag
PY=/var/tmp/vllm-study/venv/bin
# flashinfer JITs a sampler kernel at startup; it needs ninja + nvcc on PATH
export PATH="$PY:/usr/local/cuda/bin:$PATH"

echo "[start_server] model=$MODEL backend=$BACKEND maxlen=$MAXLEN port=$PORT"
exec "$PY/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
