"""Merge and score the complete V2/V4 Scheme-A rerun."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from finalize_router_e2e_20260918 import BASE, DATASETS, EXPECTED, canonical, answer, vals, bootstrap

RUN = BASE / "router_e2e_runs/v2v4_text_grouped_20260919"
OLD_RUN = BASE / "router_e2e_runs/text_grouped_20260918"
VARIANTS = ("v2", "v4")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def gold_for(d):
    rows = json.loads((BASE / f"data/{d}/test.json").read_text(encoding="utf-8"))
    return {canonical(r["question"]): (r.get("answer") if isinstance(r.get("answer"), str) else (r.get("answers") or [""])[0]) for r in rows}


def main():
    merged = {v: {} for v in VARIANTS}
    for path in sorted(RUN.glob("shard_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for v in VARIANTS:
                x = row[v]
                q = x["question"]
                if q in merged[v]:
                    raise RuntimeError(f"duplicate {v}: {q}")
                merged[v][q] = x
    total = sum(EXPECTED.values())
    for v in VARIANTS:
        if len(merged[v]) != total:
            raise RuntimeError(f"{v}: {len(merged[v])} rows, expected {total}")

    pred_dir = RUN / "predictions"
    pred_dir.mkdir(exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEVICE", device, flush=True)
    encoder = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device=device)
    metrics, arrays = {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}
    for d in DATASETS:
        gold = gold_for(d)
        for v in VARIANTS:
            rows = sorted((x for x in merged[v].values() if x["dataset"] == d), key=lambda x: x["row_id"])
            if len(rows) != EXPECTED[d]:
                raise RuntimeError(f"{v}/{d}: {len(rows)} rows, expected {EXPECTED[d]}")
            out_rows = [{"question": x["question"], "pred": x["pred"]} for x in rows]
            (pred_dir / f"{d}_{v}_router_textgrouped_20260919.json").write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
            answers = [gold[canonical(x["question"])] for x in rows]
            predictions = [answer(x["pred"]) for x in rows]
            arr = np.asarray([vals(x["pred"], g) for x, g in zip(rows, answers)], dtype=float)
            with torch.no_grad():
                pv = encoder.encode(predictions, convert_to_tensor=True, batch_size=1024, show_progress_bar=False)
                gv = encoder.encode(answers, convert_to_tensor=True, batch_size=1024, show_progress_bar=False)
                sbert = float(util.pytorch_cos_sim(pv, gv).diag().mean().item()) * 100
            mean = arr.mean(axis=0) * 100
            metrics[v][d] = {"n": len(rows), "SBERT": round(sbert, 2), "Hit": round(float(mean[0]), 2), "P": round(float(mean[1]), 2), "R": round(float(mean[2]), 2), "F1": round(float(mean[3]), 2)}
            arrays[v][d] = arr[:, 3]
            print(v, d, metrics[v][d], flush=True)

    comparisons = {"V1_vs_V2": {}, "V1_vs_V4": {}}
    for d in DATASETS:
        v1_rows = json.loads((OLD_RUN / "predictions" / f"{d}_v1_router_textgrouped_20260918.json").read_text(encoding="utf-8"))
        gold = gold_for(d)
        v1 = {canonical(x["question"]): x for x in v1_rows}
        v1_arr = np.asarray([vals(v1[canonical(x["question"])] ["pred"], gold[canonical(x["question"])])[3] for x in sorted(merged["v2"].values(), key=lambda x: x["row_id"]) if x["dataset"] == d], dtype=float)
        for v, key in (("v2", "V1_vs_V2"), ("v4", "V1_vs_V4")):
            rows = sorted((x for x in merged[v].values() if x["dataset"] == d), key=lambda x: x["row_id"])
            comparisons[key][d] = bootstrap(v1_arr, arrays[v][d], seed=42)
    result = {"metrics": metrics, "comparisons": comparisons, "counts": {v: len(merged[v]) for v in VARIANTS}, "sha256": {p.name: sha256(p) for p in sorted(RUN.glob("shard_*.jsonl"))}}
    target = RUN / "metrics_and_bootstrap.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target, flush=True)


if __name__ == "__main__":
    main()
