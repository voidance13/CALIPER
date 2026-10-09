#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/caliper/code/CALIPER
files=(
  always_multi_runs/always_multi_ircot_hotpotqa_full_seed42.json
  always_multi_runs/always_multi_ircot_musique_full_seed42.json
  always_multi_runs/always_multi_ircot_2wikimultihopqa_full_seed42.s0.json
  always_multi_runs/always_multi_ircot_2wikimultihopqa_full_seed42.s1.json
)
for _ in $(seq 1 480); do
  ready=1
  for file in "${files[@]}"; do [[ -s "$file" ]] || ready=0; done
  [[ "$ready" == 1 ]] && break
  sleep 30
done
for file in "${files[@]}"; do [[ -s "$file" ]] || { echo "timeout: $file missing"; exit 1; }; done
/home/gpt/anaconda3/envs/qwen35/bin/python scripts/finalize_always_multi.py
/home/gpt/anaconda3/envs/qwen35/bin/python scripts/eval_adaptive_rag.py \
  --run-dir always_multi_runs \
  --pattern 'always_multi_ircot_{dataset}_full_seed42.json' \
  --output always_multi_runs/always_multi_full_metrics.json
sha256sum \
  always_multi_runs/always_multi_ircot_*_full_seed42.json \
  always_multi_runs/always_multi_integrity.json \
  always_multi_runs/always_multi_paired_bootstrap.json \
  always_multi_runs/always_multi_full_metrics.json \
  > always_multi_runs/SHA256SUMS.always_multi.txt
echo DONE
