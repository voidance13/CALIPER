#!/usr/bin/env bash
set -euo pipefail
BASE=/mnt/data/caliper/code/CALIPER
RUN=$BASE/router_e2e_runs/v2v4_text_grouped_20260919
PY=/home/gpt/anaconda3/envs/caliper_new/bin/python
VLLM_PY=/home/gpt/anaconda3/envs/qwen35/bin/python
MODEL=/mnt/data/caliper/models/Qwen3.5-9B
mkdir -p "$RUN" "$BASE/logs/v2v4_20260919"
for g in 0 1 2 3; do
  port=$((6010+g))
  CUDA_VISIBLE_DEVICES=$g nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$port" --trust-remote-code --max-model-len 8192 > "$BASE/logs/v2v4_20260919/vllm_${g}.log" 2>&1 &
done
for port in 6010 6011 6012 6013; do
  for i in $(seq 1 120); do
    if curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then break; fi
    sleep 5
  done
  curl -fsS "http://localhost:${port}/v1/models" >/dev/null
done
for g in 0 1 2 3; do
  port=$((6010+g))
  nohup "$PY" -u "$BASE/work/router_e2e_v2v4_rerun_20260919.py" --port "$port" --shard "$g" --nshards 4 > "$BASE/logs/v2v4_20260919/shard_${g}.log" 2>&1 &
done
echo "STARTED v2v4_20260919 4 shards"
