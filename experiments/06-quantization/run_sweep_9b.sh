#!/bin/bash
# Request-rate sweep for the Qwen3.5-9B-AWQ companion (4-bit AWQ, awq_marlin kernel).
# Same fixed 256-in/128-out workload as the 4B and 0.5B studies so all three compose.
# 9B is bigger than 4B and adds AWQ dequant overhead, so the knee is expected even
# lower; rates start at 1.
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
export FLASHINFER_DISABLE_VERSION_CHECK=1
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
MODEL=QuantTrio/Qwen3.5-9B-AWQ
RATES=${1:-"1 2 4 6 8 16 inf"}

for RATE in $RATES; do
  echo "########## request_rate=$RATE ##########"
  $BIN bench serve --backend vllm --model $MODEL --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate $RATE --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    2>&1 | grep -iE "Successful|Request throughput|Output token throughput|Median TTFT|P99 TTFT|P99 ITL|Median E2EL|P99 E2EL"
done
echo "=== 9B SWEEP DONE ==="
