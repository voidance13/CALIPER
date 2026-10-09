import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


BASE = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
MODEL = SentenceTransformer(
    "/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu"
)
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = "".join(character for character in text if character not in PUNCT)
    return " ".join(text.split())


def bootstrap_ci(values, samples=2000, seed=42):
    values = np.asarray(values, dtype=np.float32)
    generator = np.random.default_rng(seed)
    means = []
    for _ in range(0, samples, 50):
        resampled = generator.choice(values, size=(min(50, samples - len(means)), len(values)))
        means.extend(resampled.mean(axis=1))
    lower, upper = np.percentile(means, (2.5, 97.5))
    return round(float(lower * 100), 2), round(float(upper * 100), 2)


results = {}
for dataset in ("hotpotqa", "musique", "2wikimultihopqa"):
    rows = json.load(
        (
            BASE
            / "baseline_runs"
            / f"naive_rag_{dataset}_full_seed42_w1.json"
        ).open(encoding="utf-8")
    )
    hit_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    predictions = []
    answers = []
    for row in rows:
        predicted = row["pred"]
        answer = row["answer"]
        predicted_tokens = normalize(predicted).split()
        answer_tokens = normalize(answer).split()
        common = sum((Counter(predicted_tokens) & Counter(answer_tokens)).values())
        precision = common / len(predicted_tokens) if predicted_tokens else 0.0
        recall = common / len(answer_tokens) if answer_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        hit_values.append(float(normalize(answer) in normalize(predicted)))
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        predictions.append(predicted)
        answers.append(answer)

    with torch.no_grad():
        prediction_embeddings = MODEL.encode(
            predictions, convert_to_tensor=True, batch_size=256
        )
        answer_embeddings = MODEL.encode(answers, convert_to_tensor=True, batch_size=256)
        sbert = float(
            util.pytorch_cos_sim(prediction_embeddings, answer_embeddings).diag().mean()
        )
    f1_lower, f1_upper = bootstrap_ci(f1_values)
    results[dataset] = {
        "SBERT": round(sbert * 100, 2),
        "Hit": round(float(np.mean(hit_values)) * 100, 2),
        "P": round(float(np.mean(precision_values)) * 100, 2),
        "R": round(float(np.mean(recall_values)) * 100, 2),
        "F1": round(float(np.mean(f1_values)) * 100, 2),
        "F1_CI95": [f1_lower, f1_upper],
        "n": len(rows),
    }
    print(json.dumps({dataset: results[dataset]}, ensure_ascii=False), flush=True)

with OUTPUT.open("w", encoding="utf-8") as stream:
    json.dump(results, stream, ensure_ascii=False, indent=2)
