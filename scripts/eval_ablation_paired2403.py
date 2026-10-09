"""Re-evaluate V0--V4 on the same 2,403-question MuSiQue set as Table 2."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

BASE = Path("/mnt/data/caliper/code/CALIPER")
TEST = BASE / "data/musique/test.json"
SOURCES = {
    "V0": "data/musique/musique_v0_2stream_steps_reflect.json",
    "V1": "data/musique/musique_v1_ubh_2stream.json",
    "V2": ["data/musique/musique_v2_cal_base_reflect.json.s0", "data/musique/musique_v2_cal_base_reflect.json.s1"],
    "V3": "data/musique/musique_v3_baseline_2stream.json",
    "V4": "data/musique/musique_v4_baseline_base.json",
}
PUNCT = set(chr(i) for i in range(33, 48)) | set(chr(i) for i in range(58, 65)) | set(chr(i) for i in range(91, 97)) | set(chr(i) for i in range(123, 127))

def canonical(text):
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())

def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(c for c in text if c not in PUNCT).split())

def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        m = re.search(re.escape(marker), text or "")
        if m:
            return text[m.end():]
    line = (text or "").split("\n")[-1]
    m = re.search(": ", line)
    return line[m.end():] if m else line

def f1_score(prediction, gold):
    pt, gt = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(pt) & Counter(gt)).values())
    p = common / len(pt) if pt else 0.0
    r = common / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def answer(row):
    value = row.get("answer")
    return value if isinstance(value, str) else (row.get("answers") or [""])[0]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_rows(spec):
    paths = spec if isinstance(spec, list) else [spec]
    rows = []
    for rel in paths:
        rows.extend(json.load((BASE / rel).open(encoding="utf-8")))
    return rows, paths

def main():
    tests = json.load(TEST.open(encoding="utf-8"))
    test_counts = Counter(canonical(row["question"]) for row in tests)
    loaded = {v: load_rows(spec) for v, spec in SOURCES.items()}
    source_counts = {v: Counter(canonical(row.get("question", "")) for row in rows) for v, (rows, _) in loaded.items()}
    keys = [canonical(row["question"]) for row in tests if test_counts[canonical(row["question"])] == 1 and all(source_counts[v][canonical(row["question"])] == 1 for v in SOURCES)]
    assert len(keys) == 2403, f"expected 2403, got {len(keys)}"
    included = set(keys)
    gold = {canonical(row["question"]): answer(row) for row in tests if canonical(row["question"]) in included}
    answers = [gold[k] for k in keys]
    model = SentenceTransformer(str(BASE.parents[1] / "models/paraphrase-mpnet-base-v2"), device="cpu")
    with torch.no_grad():
        answer_vectors = model.encode(answers, convert_to_tensor=True, batch_size=256)
    results = {}
    for variant, (rows, paths) in loaded.items():
        pred_by_key = {canonical(row["question"]): parse_answer(row.get("pred") or row.get("res", "")) for row in rows if source_counts[variant][canonical(row.get("question", ""))] == 1}
        predictions = [pred_by_key[k] for k in keys]
        hit = precision = recall = f1 = 0.0
        for pred, gold_answer in zip(predictions, answers):
            pn, gn = normalize(pred), normalize(gold_answer)
            hit += int(gn in pn)
            pt, gt = pn.split(), gn.split()
            common = sum((Counter(pt) & Counter(gt)).values())
            p = common / len(pt) if pt else 0.0
            r = common / len(gt) if gt else 0.0
            precision += p; recall += r; f1 += 2 * p * r / (p + r) if p + r else 0.0
        with torch.no_grad():
            pred_vectors = model.encode(predictions, convert_to_tensor=True, batch_size=256)
            sbert = float(util.pytorch_cos_sim(pred_vectors, answer_vectors).diag().mean()) * 100
        n = len(answers)
        results[variant] = {
            "SBERT": round(sbert, 2), "Hit": round(hit / n * 100, 2),
            "P": round(precision / n * 100, 2), "R": round(recall / n * 100, 2),
            "F1": round(f1 / n * 100, 2), "n": n,
            "sources": paths, "source_sha256": [sha256(BASE / p) for p in paths],
        }
        print(variant, results[variant], flush=True)
    payload = {
        "evaluator": "scripts/eval_ablation_paired2403.py",
        "selection": "canonical question unique in test and exactly once in every V0--V4 prediction file",
        "test_n": len(tests), "paired_n": len(keys), "excluded_n": len(tests) - len(keys),
        "results": results,
    }
    target = BASE / "table-result/ablation_paired2403_metrics.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target, flush=True)

if __name__ == "__main__":
    main()
