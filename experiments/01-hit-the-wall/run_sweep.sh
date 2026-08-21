#!/bin/bash
# Request-rate sweep against the running Qwen3.5-4B server. Same fixed workload as
# the 0.5B study (256-token input, 128-token output, ignore-eos) so the two curves
# compose -- only the model changed. A 4B model saturates at a far LOWER request
# rate than 0.5B, so the rate list is passed in (arg1, space-separated) to let us
# place points around wherever the knee actually lands.
#   ./run_sweep.sh "1 2 4 8 16 inf"   [num_prompts]  [tag]
export HF_HOME=/var/tmp/vllm-study/hf-cache
export PATH=/var/tmp/vllm-study/venv-qwen35/bin:$PATH
BIN=/var/tmp/vllm-study/venv-qwen35/bin/vllm
MODEL=Qwen/Qwen3.5-4B
RATES=${1:-"1 2 4 8 16 inf"}
NP=${2:-200}
TAG=${3:-main}

for RATE in $RATES; do
  echo "########## [$TAG] request_rate=$RATE ##########"
  $BIN bench serve \
    --backend vllm --model $MODEL \
    --host localhost --port 8000 \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts $NP --request-rate $RATE --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    2>&1 | grep -iE "Successful|request throughput|output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|P99 TPOT|Mean ITL|P99 ITL|Mean E2EL|Median E2EL|P99 E2EL|Maximum request"
done
echo "=== SWEEP [$TAG] DONE ==="
