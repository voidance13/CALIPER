#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/data/caliper/code/CALIPER
RUN=$BASE/router_e2e_runs/text_grouped_20260918
PY=/home/gpt/anaconda3/envs/caliper_new/bin/python
VLLM_PY=/home/gpt/anaconda3/envs/qwen35/bin/python
MODEL=/mnt/data/caliper/models/Qwen3.5-9B
mkdir -p "$RUN" "$BASE/logs/router_e2e_20260918"

# Immutable input snapshot for this run.
cp -n /tmp/true_routed_text_grouped_20260918/final_records/all_records.jsonl "$RUN/routed_records.jsonl"
sha256sum "$RUN/routed_records.jsonl" > "$RUN/input_sha256.txt"

for g in 0 1 2 3; do
  port=$((6010+g))
  log="$BASE/logs/router_e2e_20260918/vllm_${g}.log"
  CUDA_VISIBLE_DEVICES=$g nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$port" --trust-remote-code --max-model-len 8192 \
    > "$log" 2>&1 &
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
  nohup "$PY" -u "$BASE/work/router_e2e_rerun_20260918.py" \
    --port "$port" --shard "$g" --nshards 4 \
    > "$BASE/logs/router_e2e_20260918/shard_${g}.log" 2>&1 &
done
echo "STARTED router_e2e_20260918 4 shards"
