#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/caliper/code/CALIPER
OUT=query_clf/heldout_source_router_20260917/routed
ROUTER=query_clf/heldout_source_router_20260917/domain_clf.bin
mkdir -p "$OUT/logs"

run_job() {
  local gpu="$1" dataset="$2" route="$3"
  local output="$OUT/${dataset}_to_${route}.jsonl"
  echo "START gpu=$gpu dataset=$dataset route=$route" >&2
  CUDA_VISIBLE_DEVICES="$gpu" /home/gpt/anaconda3/envs/caliper_new/bin/python \
    work/repair/true_routed_recall_heldout.py \
    --mode run --dataset "$dataset" --predicted-route "$route" \
    --shards 1 --output "$output" --router "$ROUTER" --overwrite \
    > "$OUT/logs/${dataset}_to_${route}.log" 2>&1
}

run_group() {
  local group="$1"
  shift
  local pids=()
  while [ "$#" -gt 0 ]; do
    local gpu="$1" dataset="$2" route="$3"
    shift 3
    run_job "$gpu" "$dataset" "$route" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "GROUP_FAILED $group" >&2
    exit 1
  fi
  echo "GROUP_DONE $group" >&2
}

run_group 1 0 hp hp 1 ms ms 2 tw tw 3 hp ms
run_group 2 0 hp tw 1 ms hp 2 ms tw 3 tw hp
run_group 3 0 tw ms

/home/gpt/anaconda3/envs/caliper_new/bin/python \
  work/repair/true_routed_recall_heldout.py \
  --mode aggregate --input-dir "$OUT" \
  --output "$OUT/summary.json" --router "$ROUTER"
echo "RUN_COMPLETE $OUT/summary.json" >&2
