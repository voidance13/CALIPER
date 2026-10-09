"""Uniform final evaluation for every Table 2 row."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

BASE = Path("/mnt/data/caliper/code/CALIPER")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
SOURCES = {
    "Qwen3.5-9B": {
        "hotpotqa": "data/hotpotqa/hotpotqa_baseline_2stage_soft.json",
        "musique": "data/musique/musique_baseline_2stage_soft.json",
        "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_baseline_2stage_soft.json",
    },
    "Naive RAG (strict)": {d: f"baseline_runs/naive_rag_{d}_full_seed42_w1.json" for d in DATASETS},
    "Adaptive-RAG (adapted)": {d: f"adaptive_runs/adaptive_{d}_full_seed42_v3.json" for d in DATASETS},
    "Always-Multi (adapted C)": {d: f"always_multi_runs/always_multi_ircot_{d}_full_seed42.json" for d in DATASETS},
    "CALIPER+clf": {d: f"data/{d}/{d}_clf_t2_th0997.json" for d in DATASETS},
    "CALIPER": {
        "hotpotqa": "data/hotpotqa/hotpotqa_caliper_planA_top5.json",
        "musique": "data/musique/musique_v1_ubh_2stream.json",
        "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json",
    },
    "Qwen3.7-plus": {
        "hotpotqa": "data/hotpotqa/direct_pred_t2_qwen37.json",
        "musique": "data/musique/direct_pred_t2_qwen37_A.json",
        "2wikimultihopqa": "data/2wikimultihopqa/direct_pred_t2_qwen37.json",
    },
}
PUNCT = set(chr(i) for i in range(33, 48)) | set(chr(i) for i in range(58, 65)) | set(chr(i) for i in range(91, 97)) | set(chr(i) for i in range(123, 127))


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in PUNCT).split())


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(model, dataset, source_name):
    tests = json.load((BASE / f"data/{dataset}/test.json").open(encoding="utf-8"))
    rows = json.load((BASE / source_name).open(encoding="utf-8"))
    aligned = len(rows) == len(tests) and all(rows[i].get("question") == tests[i]["question"] for i in range(len(tests)))
    gold_by_question = {row["question"]: (row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0]) for row in tests}
    pairs = []
    for index, row in enumerate(rows):
        gold = (tests[index].get("answer") if isinstance(tests[index].get("answer"), str) else (tests[index].get("answers") or [""])[0]) if aligned else gold_by_question.get(row.get("question"), "")
        if gold:
            pairs.append((parse_answer(row.get("pred") or row.get("res", "")), gold))
    hit = precision = recall = f1 = 0.0
    for prediction, gold in pairs:
        predicted, expected = normalize(prediction), normalize(gold)
        hit += int(expected in predicted)
        predicted_tokens, expected_tokens = predicted.split(), expected.split()
        common = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
        p = common / len(predicted_tokens) if predicted_tokens else 0.0
        r = common / len(expected_tokens) if expected_tokens else 0.0
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if p + r else 0.0
    with torch.no_grad():
        predictions = model.encode([pair[0] for pair in pairs], convert_to_tensor=True, batch_size=256)
        answers = model.encode([pair[1] for pair in pairs], convert_to_tensor=True, batch_size=256)
        sbert = float(util.pytorch_cos_sim(predictions, answers).diag().mean()) * 100
    n = len(pairs)
    return {
        "SBERT": round(sbert, 2), "Hit": round(hit / n * 100, 2),
        "P": round(precision / n * 100, 2), "R": round(recall / n * 100, 2),
        "F1": round(f1 / n * 100, 2), "n": n, "aligned_by_index": aligned,
        "source": source_name, "source_sha256": sha256(BASE / source_name),
    }


model = SentenceTransformer(str(BASE.parents[1] / "models/paraphrase-mpnet-base-v2"), device="cpu")
results = {}
for method, datasets in SOURCES.items():
    results[method] = {}
    for dataset, source in datasets.items():
        results[method][dataset] = evaluate(model, dataset, source)
        print(method, dataset, results[method][dataset], flush=True)

assert len(results) == 7 and all(len(row) == 3 for row in results.values())
target = BASE / "table-result/table2_final_metrics.json"
target.write_text(json.dumps({"evaluator": "scripts/eval_table2_final.py", "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print("SAVED", target)
