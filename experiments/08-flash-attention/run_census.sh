#!/usr/bin/env bash
# Census the engine-process traces in capture order for both arms.
set -uo pipefail
DIR=/home/mpathak/code/research/vllm/fa-study
PY=/var/tmp/vllm-study/venv/bin/python
LABELS=(prefill_2k prefill_8k prefill_28k decode_2k decode_28k)
for arm in flash flex; do
  echo "############## ARM $arm ##############"
  tdir="/var/tmp/vllm-study/traces_fa/$arm"
  # engine-process traces = those NOT containing 'async_llm', sorted by name (timestamp order)
  mapfile -t files < <(ls -1 "$tdir"/*.json.gz | grep -v async_llm | sort)
  i=0
  for f in "${files[@]}"; do
    lbl="${LABELS[$i]:-extra$i}"
    echo "===================== $arm / $lbl ====================="
    $PY "$DIR/tp_census.py" "$f"
    i=$((i+1))
  done
done
