"""Paired bootstrap of CALIPER core against Adaptive-RAG and Naive RAG."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
SOURCES = {
    "hotpotqa": Path("data/hotpotqa/hotpotqa_caliper_planA_top5.json"),
    "musique": Path("data/musique/musique_v1_ubh_2stream.json"),
    "2wikimultihopqa": Path("data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json"),
}


def canonical(text):
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(text).lower()).split())


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(ch for ch in text if ch not in PUNCT).split())


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def f1(prediction, answer):
    predicted, gold = normalize(parse_answer(prediction)).split(), normalize(answer).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def test_difference(left, right, generator, samples=10000):
    differences = left - right
    bootstrap = np.empty(samples)
    for start in range(0, samples, 50):
        size = min(50, samples - start)
        indices = generator.integers(0, len(differences), size=(size, len(differences)))
        bootstrap[start:start + size] = differences[indices].mean(axis=1)
    below = (np.count_nonzero(bootstrap <= 0) + 1) / (samples + 1)
    above = (np.count_nonzero(bootstrap >= 0) + 1) / (samples + 1)
    return {
        "n": len(differences),
        "left_F1": round(float(left.mean() * 100), 2),
        "right_F1": round(float(right.mean() * 100), 2),
        "delta_F1": round(float(differences.mean() * 100), 2),
        "delta_CI95": [round(float(x * 100), 2) for x in np.percentile(bootstrap, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
        "bootstrap_samples": samples,
        "seed": 42,
    }


def main():
    output = {}
    for dataset, source in SOURCES.items():
        tests = json.load(open(f"data/{dataset}/test.json"))
        adaptive = {row["index"]: row for row in json.load(open(f"adaptive_runs/adaptive_{dataset}_full_seed42_v3.json"))}
        naive = {row["index"]: row for row in json.load(open(f"baseline_runs/naive_rag_{dataset}_full_seed42_w1.json"))}
        always = {row["index"]: row for row in json.load(open(f"always_multi_runs/always_multi_ircot_{dataset}_full_seed42.json"))}
        caliper_rows = json.load(source.open())
        test_counts = Counter(canonical(row["question"]) for row in tests)
        caliper_counts = Counter(canonical(row["question"]) for row in caliper_rows)
        caliper = {canonical(row["question"]): row["pred"] for row in caliper_rows}
        indices = [i for i, row in enumerate(tests) if test_counts[canonical(row["question"])] == 1 and caliper_counts[canonical(row["question"])] == 1]
        answers = [adaptive[i]["answer"] for i in indices]
        scores = {
            "caliper": np.asarray([f1(caliper[canonical(tests[i]["question"])], answer) for i, answer in zip(indices, answers)]),
            "adaptive": np.asarray([f1(adaptive[i]["pred"], answer) for i, answer in zip(indices, answers)]),
            "naive": np.asarray([f1(naive[i]["pred"], answer) for i, answer in zip(indices, answers)]),
            "always_multi": np.asarray([f1(always[i]["pred"], answer) for i, answer in zip(indices, answers)]),
        }
        generator = np.random.default_rng(42)
        output[dataset] = {
            "caliper_source": str(source),
            "excluded_missing_or_ambiguous": len(tests) - len(indices),
            "caliper_vs_adaptive": test_difference(scores["caliper"], scores["adaptive"], generator),
            "caliper_vs_naive": test_difference(scores["caliper"], scores["naive"], generator),
            "caliper_vs_always_multi": test_difference(scores["caliper"], scores["always_multi"], generator),
        }
        print(dataset, json.dumps(output[dataset]), flush=True)
    target = Path("significance_analysis")
    target.mkdir(exist_ok=True)
    (target / "caliper_paired_bootstrap.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
