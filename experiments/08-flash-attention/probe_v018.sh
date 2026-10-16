#!/bin/bash
# Does vLLM 0.18 run the FlashAttention-vs-FlexAttention comparison on Qwen2.5-7B at 32k?
cd /home/mpathak/code/research/vllm/fa-study
V=/var/tmp/vllm-study/venv-qwen35/bin
export HF_HOME=/var/tmp/vllm-study/hf-cache PATH=$V:/usr/local/cuda/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn FLASHINFER_DISABLE_VERSION_CHECK=1
OUT=probe_v018.log; : > $OUT
PROMPT=$(python3 -c 'print("word "*16000)')
for B in FLEX_ATTENTION FLASH_ATTN; do
  echo "===== vLLM 0.18, $B =====" >> $OUT
  timeout 900 $V/vllm serve Qwen/Qwen2.5-7B-Instruct --attention-backend $B --max-model-len 32768 \
    --gpu-memory-utilization 0.9 --port 8011 > server_v018_$B.log 2>&1 &
  P=$!; ok=0
  for i in $(seq 1 60); do
    curl -s -o /dev/null -w "%{http_code}" localhost:8011/health 2>/dev/null | grep -q 200 && { ok=1; break; }
    kill -0 $P 2>/dev/null || break; sleep 10
  done
  grep -hE "version 0\.|Using .*[Bb]ackend|AttentionBackendEnum|not support|Error" server_v018_$B.log | grep -v FutureWarning | cut -c1-220 | head -8 >> $OUT
  if [ $ok = 1 ]; then
    for r in 1 2; do
      T0=$(date +%s.%N)
      curl -s localhost:8011/v1/completions -H "Content-Type: application/json" \
        -d "{\"model\":\"Qwen/Qwen2.5-7B-Instruct\",\"prompt\":\"$PROMPT\",\"max_tokens\":1}" | head -c 120 >> $OUT; echo >> $OUT
      T1=$(date +%s.%N); echo "16k-token prefill, run $r: $(python3 -c "print(round($T1-$T0,2))") s" >> $OUT
    done
  else
    echo "SERVER DID NOT BECOME READY" >> $OUT; tail -8 server_v018_$B.log | cut -c1-220 >> $OUT
  fi
  kill $P 2>/dev/null; sleep 8; pkill -f "port 8011" 2>/dev/null; sleep 5
done
echo PROBE_DONE >> $OUT
