#!/usr/bin/env bash
# CPU optimization sweep on Q4_K_M (best model). Everything measured in tok/s — the ground
# truth, since this virtualized host exposes no hardware PMU (perf counters unavailable).
# Topology: 2 CCX / 8 physical cores (0-3 share L3#0, 4-7 share L3#1), HT siblings 8-15,
# single memory controller. All configs pinned via -C cpu-mask + --cpu-strict 1.
set -u
ROOT=/home/mpathak/code/research/vllm/cpu-inference-study
cd "$ROOT"
BIN=llama.cpp/build-cpu/bin/llama-bench
M=models/Qwen_Qwen3.5-4B-Q4_K_M.gguf
OUT=results/opt_sweep.md
echo "# CPU optimization sweep — Q4_K_M — $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT"

run() { # label, extra-args
  echo "" >> "$OUT"; echo "### $1" >> "$OUT"
  "$BIN" -m "$M" -p 512 -n 128 -r 3 -o md $2 >> "$OUT" 2>> results/opt_sweep.err
}

# A. Affinity: 4 physical cores, all in ONE CCX (share 16MiB L3) vs SPLIT across both CCX.
run "4 cores, 1 CCX (0-3), strict"      "-t 4 -C 0x0F --cpu-strict 1"
run "4 cores, split CCX (0,1,4,5)"      "-t 4 -C 0x33 --cpu-strict 1"
# B. 8 physical cores, no HT (mask 0-7) vs default (lets scheduler use HT).
run "8 physical cores (0-7), strict"    "-t 8 -C 0xFF --cpu-strict 1"
run "8 threads, default placement"      "-t 8"
# C. KV-cache dtype — cut KV bytes streamed (q8_0 K+V) vs default f16.
run "8 phys, KV q8_0"                    "-t 8 -C 0xFF --cpu-strict 1 -ctk q8_0 -ctv q8_0"
# D. flash-attn explicit on (default is auto).
run "8 phys, flash-attn on"             "-t 8 -C 0xFF --cpu-strict 1 -fa on"
# E. poll=100 (busy-wait, lower wakeup latency) vs default 50.
run "8 phys, poll 100"                  "-t 8 -C 0xFF --cpu-strict 1 --poll 100"

echo "" >> "$OUT"; echo "OPT_SWEEP_DONE" >> "$OUT"
