import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("always_multi_runs")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(ch for ch in text if ch not in PUNCT).split())


def f1(prediction, answer):
    predicted, gold = normalize(prediction).split(), normalize(answer).split()
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
        "n": len(differences), "left_F1": round(float(left.mean() * 100), 2),
        "right_F1": round(float(right.mean() * 100), 2),
        "delta_F1": round(float(differences.mean() * 100), 2),
        "delta_CI95": [round(float(x * 100), 2) for x in np.percentile(bootstrap, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
        "bootstrap_samples": samples, "seed": 42,
    }


def main():
    shards = [json.load(open(ROOT / f"always_multi_ircot_2wikimultihopqa_full_seed42.s{i}.json")) for i in (0, 1)]
    merged = sorted(shards[0] + shards[1], key=lambda row: row["index"])
    merged_path = ROOT / "always_multi_ircot_2wikimultihopqa_full_seed42.json"
    temporary = merged_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    temporary.replace(merged_path)

    integrity, significance = {}, {}
    for dataset in DATASETS:
        path = ROOT / f"always_multi_ircot_{dataset}_full_seed42.json"
        rows = json.load(path.open())
        expected = len(json.load(open(f"data/{dataset}/test.json")))
        indices = [row["index"] for row in rows]
        if len(rows) != expected or indices != list(range(expected)):
            raise ValueError(f"{dataset}: incomplete or misordered indices")
        integrity[dataset] = {
            "n": len(rows), "all_route_C": all(row["route"] == "C" for row in rows),
            "empty": sum(not row["pred"].strip() for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if not integrity[dataset]["all_route_C"] or integrity[dataset]["empty"] or integrity[dataset]["errors"]:
            raise ValueError(f"{dataset}: integrity failure")
        always = np.asarray([f1(row["pred"], row["answer"]) for row in rows])
        significance[dataset] = {}
        for name, other_path in {
            "adaptive": Path(f"adaptive_runs/adaptive_{dataset}_full_seed42_v3.json"),
            "naive": Path(f"baseline_runs/naive_rag_{dataset}_full_seed42_w1.json"),
        }.items():
            other = json.load(other_path.open())
            other_scores = np.asarray([f1(row["pred"], row["answer"]) for row in other])
            significance[dataset][f"always_multi_vs_{name}"] = test_difference(always, other_scores, np.random.default_rng(42))
    (ROOT / "always_multi_integrity.json").write_text(json.dumps(integrity, indent=2))
    (ROOT / "always_multi_paired_bootstrap.json").write_text(json.dumps(significance, indent=2))
    print(json.dumps({"integrity": integrity, "significance": significance}), flush=True)


if __name__ == "__main__":
    main()
