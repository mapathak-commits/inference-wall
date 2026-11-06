#!/bin/bash
# Third arm: torch.compile ON, CUDA graphs OFF (cudagraph_mode NONE). Isolates graphs from compilation.
set -u
cd /home/mpathak/code/research/vllm/cudagraph-study
source <(sed -n '/^export HF_HOME/,/^OUT=/p' run_study.sh | sed 's|^OUT=.*|OUT=results/study_nograph.log; mkdir -p results traces; echo "=== NOGRAPH ARM START $(date -u +%FT%TZ) ===" > $OUT|')
eval "$(sed -n '/^start_server () {/,/^}/p' run_study.sh)"
eval "$(sed -n '/^bench () {/,/^}/p' run_study.sh)"
NG='--compilation-config {"cudagraph_mode":"NONE"}'
for MODEL in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen3.5-4B; do
  short=$(basename $MODEL); tag=${short}_compileonly
  start_server $MODEL graph $tag "$NG" || continue
  grep -ohE "cudagraph_mode': [^,]*|'mode': <CompilationMode[^>]*>|torch.compile took [0-9.]+ s" results/server_$tag.log | head -3 >> $OUT
  bench $MODEL WARMUP_$tag 64 128 128 128
  bench $MODEL $tag 1 16 128 256
  bench $MODEL $tag 4 64 128 256
  bench $MODEL $tag 16 128 128 256
  bench $MODEL $tag 64 256 128 256
  [ "$short" = "Qwen3.5-4B" ] && bench $MODEL $tag none 200 256 128 inf
done
for MODEL in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3.5-4B; do
  short=$(basename $MODEL); tag=trace_${short}_compileonly; d=$(pwd)/traces/${short}_compileonly; mkdir -p $d
  start_server $MODEL graph $tag "$NG --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=$d" || continue
  $PY capture_trace.py $MODEL >> $OUT 2>&1; sleep 15
done
pkill -f "vllm serve" 2>/dev/null
echo "=== NOGRAPH ARM DONE $(date -u +%FT%TZ) ===" >> $OUT
