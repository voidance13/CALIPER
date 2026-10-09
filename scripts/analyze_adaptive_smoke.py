"""Compare Adaptive-RAG smoke predictions with Naive RAG on identical indices."""

import json
import re
import sys
from collections import Counter
from pathlib import Path


PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join("".join(x for x in text if x not in PUNCT).split())


def f1(prediction, answer):
    predicted = normalize(prediction).split()
    gold = normalize(answer).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    precision = common / len(predicted) if predicted else 0
    recall = common / len(gold) if gold else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


base = Path.cwd()
suffix = sys.argv[1] if len(sys.argv) > 1 else ""
datasets = (sys.argv[2],) if len(sys.argv) > 2 else ("hotpotqa", "musique", "2wikimultihopqa")
for dataset in datasets:
    adaptive = json.loads((base / "adaptive_runs" / f"adaptive_{dataset}_smoke100_seed42{suffix}.json").read_text())
    naive_rows = json.loads((base / "baseline_runs" / f"naive_rag_{dataset}_full_seed42_w1.json").read_text())
    naive = {row["index"]: row for row in naive_rows}
    output = {}
    for route in ("all", "A", "B", "C"):
        rows = adaptive if route == "all" else [row for row in adaptive if row["route"] == route]
        if not rows:
            continue
        adaptive_f1 = sum(f1(row["pred"], row["answer"]) for row in rows) / len(rows) * 100
        naive_f1 = sum(f1(naive[row["index"]]["pred"], row["answer"]) for row in rows) / len(rows) * 100
        output[route] = {
            "n": len(rows), "adaptive_f1": round(adaptive_f1, 2),
            "naive_f1": round(naive_f1, 2), "delta": round(adaptive_f1 - naive_f1, 2),
        }
    print(json.dumps({dataset: output}, ensure_ascii=False))
    if dataset == "2wikimultihopqa":
        failures = sorted(
            (
                (f1(naive[row["index"]]["pred"], row["answer"]) - f1(row["pred"], row["answer"]), row)
                for row in adaptive if row["route"] == "C"
            ),
            reverse=True,
            key=lambda item: item[0],
        )[:5]
        for delta, row in failures:
            print(json.dumps({
                "delta": round(delta, 3), "question": row["question"], "gold": row["answer"],
                "naive": naive[row["index"]]["pred"], "adaptive": row["pred"], "trace": row["trace"],
            }, ensure_ascii=False))
