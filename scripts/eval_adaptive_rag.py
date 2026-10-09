"""Evaluate Adaptive-RAG predictions with the frozen CALIPER metric definitions."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True, help="Filename containing {dataset}")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = "".join(character for character in text if character not in PUNCT)
    return " ".join(text.split())


def bootstrap_ci(values, samples=2000, seed=42):
    values = np.asarray(values, dtype=np.float32)
    generator = np.random.default_rng(seed)
    means = [generator.choice(values, size=len(values)).mean() for _ in range(samples)]
    return [round(float(x * 100), 2) for x in np.percentile(means, (2.5, 97.5))]


def main():
    args = parse_args()
    model = SentenceTransformer(
        "/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu"
    )
    results = {}
    for dataset in DATASETS:
        rows = json.loads(
            (args.run_dir / args.pattern.format(dataset=dataset)).read_text(encoding="utf-8")
        )
        hits, precisions, recalls, f1s, predictions, answers = [], [], [], [], [], []
        for row in rows:
            predicted_tokens = normalize(row["pred"]).split()
            answer_tokens = normalize(row["answer"]).split()
            common = sum((Counter(predicted_tokens) & Counter(answer_tokens)).values())
            precision = common / len(predicted_tokens) if predicted_tokens else 0.0
            recall = common / len(answer_tokens) if answer_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            hits.append(float(normalize(row["answer"]) in normalize(row["pred"])))
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            predictions.append(row["pred"])
            answers.append(row["answer"])
        with torch.no_grad():
            prediction_embeddings = model.encode(predictions, convert_to_tensor=True, batch_size=256)
            answer_embeddings = model.encode(answers, convert_to_tensor=True, batch_size=256)
            sbert = float(util.pytorch_cos_sim(prediction_embeddings, answer_embeddings).diag().mean())
        latency = np.asarray([row["latency_seconds"] for row in rows])
        result = {
            "SBERT": round(sbert * 100, 2),
            "Hit": round(float(np.mean(hits)) * 100, 2),
            "P": round(float(np.mean(precisions)) * 100, 2),
            "R": round(float(np.mean(recalls)) * 100, 2),
            "F1": round(float(np.mean(f1s)) * 100, 2),
            "F1_CI95": bootstrap_ci(f1s),
            "n": len(rows),
            "routes": {label: sum(row["route"] == label for row in rows) for label in "ABC"},
            "latency_mean_seconds": round(float(latency.mean()), 3),
            "latency_p95_seconds": round(float(np.percentile(latency, 95)), 3),
            "retrievals_mean": round(float(np.mean([row["retrieval_count"] for row in rows])), 3),
            "llm_calls_mean": round(float(np.mean([row["llm_calls"] for row in rows])), 3),
            "empty": sum(not row["pred"].strip() for row in rows),
            "errors": sum(bool(row["error"]) for row in rows),
        }
        results[dataset] = result
        print(json.dumps({dataset: result}, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
