"""Paired bootstrap test of Adaptive-RAG against Naive RAG."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join("".join(x for x in text if x not in PUNCT).split())


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def f1(prediction, answer):
    predicted = normalize(parse_answer(prediction)).split()
    gold = normalize(answer).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    args = parse_args()
    generator = np.random.default_rng(args.seed)
    results = {}
    for dataset in DATASETS:
        adaptive = json.loads(
            (args.base / "adaptive_runs" / f"adaptive_{dataset}_full_seed42_v3.json").read_text()
        )
        naive = json.loads(
            (args.base / "baseline_runs" / f"naive_rag_{dataset}_full_seed42_w1.json").read_text()
        )
        if len(adaptive) != len(naive) or any(a["index"] != n["index"] for a, n in zip(adaptive, naive)):
            raise ValueError(f"{dataset}: prediction alignment mismatch")
        adaptive_f1 = np.asarray([f1(row["pred"], row["answer"]) for row in adaptive])
        naive_f1 = np.asarray([f1(row["pred"], row["answer"]) for row in naive])
        differences = adaptive_f1 - naive_f1
        bootstrap = np.empty(args.samples)
        for start in range(0, args.samples, 50):
            size = min(50, args.samples - start)
            indices = generator.integers(0, len(differences), size=(size, len(differences)))
            bootstrap[start : start + size] = differences[indices].mean(axis=1)
        below = (np.count_nonzero(bootstrap <= 0) + 1) / (args.samples + 1)
        above = (np.count_nonzero(bootstrap >= 0) + 1) / (args.samples + 1)
        result = {
            "n": len(differences),
            "adaptive_F1": round(float(adaptive_f1.mean() * 100), 2),
            "naive_F1": round(float(naive_f1.mean() * 100), 2),
            "delta_F1": round(float(differences.mean() * 100), 2),
            "delta_CI95": [round(float(x * 100), 2) for x in np.percentile(bootstrap, (2.5, 97.5))],
            "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
            "bootstrap_samples": args.samples,
            "seed": args.seed,
        }
        results[dataset] = result
        print(json.dumps({dataset: result}), flush=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
