"""Official evaluator cross-check for Scheme-A outputs; uses one free GPU."""
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from finalize_router_e2e_20260918 import BASE, RUN, DATASETS, canonical, answer, vals

EXPECTED = {"hotpotqa": 7405, "musique": 2403, "2wikimultihopqa": 12576}
PRED_DIR = RUN / "predictions"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("DEVICE", device, flush=True)
    model = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device=device)
    out = {}
    for v in ("v0", "v1", "v3", "clf"):
        out[v] = {}
        for d in DATASETS:
            rows = json.loads((PRED_DIR / f"{d}_{v}_router_textgrouped_20260918.json").read_text(encoding="utf-8"))
            gold_rows = json.loads((BASE / f"data/{d}/test.json").read_text(encoding="utf-8"))
            gold = {canonical(r["question"]): (r.get("answer") if isinstance(r.get("answer"), str) else (r.get("answers") or [""])[0]) for r in gold_rows}
            assert len(rows) == EXPECTED[d]
            predictions = [answer(r["pred"]) for r in rows]
            answers = [gold[canonical(r["question"])] for r in rows]
            arr = np.asarray([vals(r["pred"], g) for r, g in zip(rows, answers)], dtype=np.float32)
            with torch.no_grad():
                pv = model.encode(predictions, convert_to_tensor=True, batch_size=1024, show_progress_bar=False)
                gv = model.encode(answers, convert_to_tensor=True, batch_size=1024, show_progress_bar=False)
                sbert = float(util.pytorch_cos_sim(pv, gv).diag().mean().item()) * 100
            mean = arr.mean(axis=0) * 100
            out[v][d] = {"n": len(rows), "SBERT": round(sbert, 2), "Hit": round(float(mean[0]), 2), "P": round(float(mean[1]), 2), "R": round(float(mean[2]), 2), "F1": round(float(mean[3]), 2)}
            print(v, d, out[v][d], flush=True)
    target = RUN / "official_metric_check.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target, flush=True)


if __name__ == "__main__":
    main()
