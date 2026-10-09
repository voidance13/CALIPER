"""Evaluate cloud-deep quality and bootstrap Table 4 latency."""
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sentence_transformers import SentenceTransformer


BASE = Path("/mnt/data/caliper/code/CALIPER")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
PUNCTUATION = set(string.punctuation)


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in PUNCTUATION).split())


def score(prediction, gold):
    predicted, expected = normalize(prediction), normalize(gold)
    p, g = Counter(predicted.split()), Counter(expected.split())
    common = sum((p & g).values())
    precision = common / sum(p.values()) if p else 0.0
    recall = common / sum(g.values()) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return int(bool(expected) and expected in predicted), precision, recall, f1


def bootstrap_mean(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = np.empty(samples)
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        draws[start:start + size] = values[indices].mean(axis=1)
    return [round(float(x), 3) for x in np.percentile(draws, (2.5, 97.5))]


def paired_latency(deep, other, seed=42, samples=10000):
    questions = sorted(set(deep) & set(other))
    deep_values = np.asarray([deep[q] for q in questions])
    other_values = np.asarray([other[q] for q in questions])
    generator = np.random.default_rng(seed)
    differences, reductions = np.empty(samples), np.empty(samples)
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        indices = generator.integers(0, len(questions), size=(size, len(questions)))
        deep_means = deep_values[indices].mean(axis=1)
        other_means = other_values[indices].mean(axis=1)
        differences[start:start + size] = deep_means - other_means
        reductions[start:start + size] = (deep_means - other_means) / deep_means * 100
    return {
        "n": len(questions), "deep_mean": round(float(deep_values.mean()), 3),
        "other_mean": round(float(other_values.mean()), 3),
        "delta_seconds": round(float((deep_values - other_values).mean()), 3),
        "delta_seconds_ci95": [round(float(x), 3) for x in np.percentile(differences, (2.5, 97.5))],
        "reduction_percent": round(float((deep_values.mean() - other_values.mean()) / deep_values.mean() * 100), 2),
        "reduction_percent_ci95": [round(float(x), 2) for x in np.percentile(reductions, (2.5, 97.5))],
        "bootstrap_samples": samples, "seed": seed,
    }


def timing_values(rows, key):
    values = {}
    for row in rows:
        value = row[key]
        if isinstance(value, dict):
            if value.get("error") is None:
                values[row["question"]] = value["seconds"]
        else:
            values[row["question"]] = value
    return values


def main():
    model = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cuda:0")
    output = {"bootstrap_samples": 10000, "seed": 42, "datasets": {}}
    for dataset in DATASETS:
        test = json.load((BASE / f"data/{dataset}/test.json").open(encoding="utf-8"))
        gold = {row["question"]: (row.get("answer") if isinstance(row.get("answer"), str)
                                   else (row.get("answers") or [""])[0]) for row in test}
        deep_path = BASE / f"table-result/table4_cloud_deep_{dataset}_full_b256_seed42.json"
        deep = json.load(deep_path.open(encoding="utf-8"))
        pairs = [(row["prediction"], gold[row["question"]]) for row in deep["rows"] if row.get("error") is None]
        lexical = np.asarray([score(*pair) for pair in pairs])
        with torch.no_grad():
            predictions = model.encode([x[0] for x in pairs], convert_to_tensor=True, batch_size=200)
            answers = model.encode([x[1] for x in pairs], convert_to_tensor=True, batch_size=200)
            sbert = float(functional.cosine_similarity(predictions, answers).mean()) * 100
        quality = dict(zip(("Hit", "P", "R", "F1"), (lexical.mean(axis=0) * 100).round(2)))
        quality = {"SBERT": round(sbert, 2), **quality, "n": len(pairs)}

        timing = json.load((BASE / f"table-result/table4_timing_{dataset}_seed42.json").open(encoding="utf-8"))
        classifier = json.load((BASE / f"table-result/table4_clf_actual_{dataset}_seed42.json").open(encoding="utf-8"))
        values = {
            "baseline": timing_values(timing["rows"], "baseline"),
            "caliper": timing_values(timing["rows"], "caliper"),
            "cloud_short": timing_values(timing["rows"], "cloud"),
            "caliper_clf": {row["question"]: row["seconds"] for row in classifier["rows"] if row["error"] is None},
            "cloud_deep": {row["question"]: row["seconds"] for row in deep["rows"] if row["error"] is None},
        }
        latency = {}
        for method, question_values in values.items():
            seconds = list(question_values.values())
            latency[method] = {"n": len(seconds), "mean": round(float(np.mean(seconds)), 3),
                               "mean_ci95": bootstrap_mean(seconds)}
        output["datasets"][dataset] = {
            "cloud_deep_quality": quality, "latency_seconds": latency,
            "caliper_vs_cloud_deep": paired_latency(values["cloud_deep"], values["caliper"], 42),
            "caliper_clf_vs_cloud_deep": paired_latency(values["cloud_deep"], values["caliper_clf"], 43),
        }
        print(dataset, json.dumps(output["datasets"][dataset], ensure_ascii=False), flush=True)
    target = BASE / "table-result/table4_quality_latency_ci.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(output["datasets"][d]["cloud_deep_quality"]["n"] in (199, 200) for d in DATASETS)
    print("SAVED", target)


if __name__ == "__main__":
    main()
