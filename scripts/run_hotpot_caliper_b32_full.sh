#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/caliper/code/CALIPER
PY=/home/gpt/anaconda3/envs/caliper2/bin/python
mkdir -p full_runs/hotpotqa_caliper_B32 logs
pids=()
for shard in 0 1 2 3; do
  port=$((6010 + shard))
  "$PY" scripts/run_hotpot_caliper_b32_shard.py "$shard" 4 "$port" > "logs/hotpotqa_caliper_B32_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
"$PY" scripts/finalize_hotpot_caliper_b32.py > logs/hotpotqa_caliper_B32_finalize.log 2>&1
echo DONE
