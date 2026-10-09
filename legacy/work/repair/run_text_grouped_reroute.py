#!/usr/bin/env python3
"""Rerun only questions whose route changes under the strict grouped router."""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


OUT = Path("/tmp/true_routed_text_grouped_20260918")
ROOT = Path("/mnt/data/caliper/code/CALIPER")
SCRIPT = ROOT / "work/repair/true_routed_recall_heldout.py"
ROUTER = ROOT / "query_clf/text_grouped_source_router_20260917/domain_clf.bin"
PLAN = OUT / "reroute_plan.json"
PYTHON = "/home/gpt/anaconda3/envs/caliper_new/bin/python"


def main():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    jobs = []
    for key, row_ids in sorted(plan["changed_by_target"].items()):
        dataset, route = key.split("->")
        if not row_ids:
            continue
        output = OUT / f"{dataset}_to_{route}_new.jsonl"
        log = OUT / "logs" / f"{dataset}_to_{route}_new.log"
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")
        jobs.append({"dataset": dataset, "route": route, "ids": row_ids,
                     "output": output, "log": log})
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    def start(job, gpu):
        cmd = [PYTHON, str(SCRIPT), "--mode", "run", "--dataset", job["dataset"],
               "--predicted-route", job["route"], "--output", str(job["output"]),
               "--router", str(ROUTER), "--normalize-router", "--overwrite", "--row-ids",
               ",".join(map(str, job["ids"]))]
        log_handle = job["log"].open("w", encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_handle,
                                stderr=subprocess.STDOUT)
        job["proc"] = proc
        job["log_handle"] = log_handle
        job["gpu"] = gpu
        print(f"START gpu={gpu} {job['dataset']}->{job['route']} n={len(job['ids'])}", flush=True)

    pending = list(jobs)
    running = []
    available_gpus = [0, 1, 2, 3]
    while pending or running:
        while pending and available_gpus:
            job = pending.pop(0)
            gpu = available_gpus.pop(0)
            start(job, gpu)
            running.append(job)
        time.sleep(5)
        still = []
        for job in running:
            code = job["proc"].poll()
            if code is None:
                still.append(job)
                continue
            job["log_handle"].close()
            print(f"DONE gpu={job['gpu']} {job['dataset']}->{job['route']} exit={code}", flush=True)
            if code != 0:
                raise SystemExit(f"job failed; inspect {job['log']}")
            available_gpus.append(job["gpu"])
        running = still

    # Merge old unchanged records with new rerun records and aggregate once.
    merge = OUT / "merged_records.jsonl"
    old_dir = Path("/tmp/true_routed_final_20260917")
    new_by_key = {}
    for path in OUT.glob("*_to_*_new.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                new_by_key[(row["dataset"], row["row_id"])] = row
    changed_keys = {
        (key.split("->")[0], row_id)
        for key, row_ids in plan["changed_by_target"].items()
        for row_id in row_ids
    }
    old_keys = set()
    with merge.open("w", encoding="utf-8") as handle:
        for path in sorted(old_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["dataset"], row["row_id"])
                old_keys.add(key)
                if key not in changed_keys:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        # New rows replace the old rows for changed assignments.
        for key, row in sorted(new_by_key.items()):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_dir = OUT / "final_records"
    final_dir.mkdir(exist_ok=True)
    shutil.copyfile(merge, final_dir / "all_records.jsonl")
    agg = [PYTHON, str(SCRIPT), "--mode", "aggregate", "--input-dir", str(final_dir),
           "--output", str(OUT / "summary.json"), "--router", str(ROUTER)]
    subprocess.run(agg, cwd=ROOT, check=True)
    print(f"RUN_COMPLETE {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
