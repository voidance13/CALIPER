#!/usr/bin/env python3
"""Run only the frozen 2,403-question MuSiQue E2E protocol.

This reuses the original generation implementation and changes only the
versioned routed-record snapshot and MuSiQue BASE context cache.
"""
import importlib.util
import json
from pathlib import Path

SRC = Path("/mnt/data/caliper/code/CALIPER/work/router_e2e_rerun_20260918.py")
spec = importlib.util.spec_from_file_location("router_e2e_frozen", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.RUN = mod.BASE / "router_e2e_runs/text_grouped_20260920_unified_bgebase"
mod.DATASETS = ("musique",)


def load_items(records_path):
    routed = {}
    for line in records_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            dataset = mod.DATASET_ALIASES.get(row["dataset"], row["dataset"])
            if dataset == "musique":
                routed[row["question"]] = row
    mu_keys = mod.load_mu_keys()
    tests = mod.load_json(mod.BASE / "data/musique/test.json")
    items = []
    for i, t in enumerate(tests):
        q = t["question"]
        if mod.canonical(q) not in mu_keys:
            continue
        if q not in routed:
            raise RuntimeError(f"missing new MuSiQue routed record: row={i}")
        items.append({"dataset": "musique", "row_id": i, "question": q, "routed": routed[q]})
    if len(items) != 2403:
        raise RuntimeError(f"frozen MuSiQue intersection changed: {len(items)}")
    return items


def load_contexts(items):
    return {
        "musique": {
            r["question"]: mod.text_list(r)
            for r in mod.load_json(mod.BASE / "data/musique/test_retrieved_unified_bgebase_base.json")
        }
    }


mod.load_items = load_items
mod.load_contexts = load_contexts

if __name__ == "__main__":
    mod.main()
