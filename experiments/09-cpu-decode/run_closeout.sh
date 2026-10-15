#!/bin/bash
cd /home/mpathak/code/research/vllm/cpu-inference-study
pkill -9 -f llama-bench 2>/dev/null
rm -f results/closeout.err
setsid nohup bash src/closeout.sh > /tmp/closeout_launch.log 2>&1 &
echo "LAUNCHED pid=$!"
exit 0
