#!/bin/bash
# Build vLLM 0.18.0 CPU backend from source into venv-vllm-cpu.
# vLLM's stock wheel is CUDA-linked; the CPU backend must be compiled with
# VLLM_TARGET_DEVICE=cpu against a CPU torch.
set -e
cd /home/mpathak/code/research/vllm/cpu-inference-study
ROOT=/home/mpathak/code/research/vllm/cpu-inference-study
VENV=$ROOT/venv-vllm-cpu
PY=$VENV/bin/python
UV=~/.local/bin/uv

echo "=== [1/4] fresh CPU venv ==="
rm -rf "$VENV"
$UV venv "$VENV" --python 3.12

echo "=== [2/4] CPU torch + build deps ==="
$UV pip install --python "$PY" --link-mode=copy \
  torch==2.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
$UV pip install --python "$PY" --link-mode=copy \
  "cmake>=3.26" ninja packaging setuptools setuptools-scm wheel numpy

echo "=== [3/4] CPU requirements ==="
$UV pip install --python "$PY" --link-mode=copy -r vllm-src/requirements/cpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match || true

echo "=== [4/4] build vLLM CPU (VLLM_TARGET_DEVICE=cpu) ==="
cd vllm-src
VLLM_TARGET_DEVICE=cpu MAX_JOBS=16 "$PY" -m pip install -e . --no-build-isolation -v 2>&1 | tail -40
echo "VLLM_CPU_BUILD_DONE"
