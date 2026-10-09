"""Evaluate all Table 2 MuSiQue methods on one identical 2,403-question set."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util


BASE = Path("/mnt/data/caliper/code/CALIPER")
TEST = BASE / "data/musique/test.json"
SOURCES = {
    "Qwen3.5-9B": "data/musique/musique_baseline_2stage_soft.json",
    "Naive RAG (strict)": "baseline_runs/naive_rag_musique_full_seed42_w1.json",
    "Adaptive-RAG (adapted)": "adaptive_runs/adaptive_musique_full_seed42_v3.json",
    "Always-Multi (adapted C)": "always_multi_runs/always_multi_ircot_musique_full_seed42.json",
    "CALIPER+clf": "data/musique/musique_clf_t2_th0997.json",
    "CALIPER": "data/musique/musique_v1_ubh_2stream.json",
    "Qwen3.7-plus": "data/musique/direct_pred_t2_qwen37_A.json",
}
PUNCT = set(chr(i) for i in range(33, 48)) | set(chr(i) for i in range(58, 65)) | set(chr(i) for i in range(91, 97)) | set(chr(i) for i in range(123, 127))


def canonical(text):
    return " ".join("".join(char if char.isalnum() else " " for char in str(text).lower()).split())


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


def answer(row):
    value = row.get("answer")
    return value if isinstance(value, str) else (row.get("answers") or [""])[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f1_score(prediction, gold):
    predicted_tokens, expected_tokens = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    precision = common / len(predicted_tokens) if predicted_tokens else 0.0
    recall = common / len(expected_tokens) if expected_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def paired_bootstrap(left, right, samples=10000):
    differences = left - right
    generator = np.random.default_rng(42)
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
        "delta_CI95": [round(float(value * 100), 2) for value in np.percentile(bootstrap, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
        "bootstrap_samples": samples,
        "seed": 42,
    }


def metrics(model, predictions, answers, answer_vectors):
    hit = precision = recall = f1 = 0.0
    for prediction, gold in zip(predictions, answers):
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
        predicted_vectors = model.encode(predictions, convert_to_tensor=True, batch_size=256)
        sbert = float(util.pytorch_cos_sim(predicted_vectors, answer_vectors).diag().mean()) * 100
    n = len(answers)
    return {
        "SBERT": round(sbert, 2),
        "Hit": round(hit / n * 100, 2),
        "P": round(precision / n * 100, 2),
        "R": round(recall / n * 100, 2),
        "F1": round(f1 / n * 100, 2),
        "n": n,
    }


def main():
    tests = json.load(TEST.open(encoding="utf-8"))
    test_counts = Counter(canonical(row["question"]) for row in tests)
    source_rows = {method: json.load((BASE / source).open(encoding="utf-8")) for method, source in SOURCES.items()}
    source_counts = {
        method: Counter(canonical(row.get("question", "")) for row in rows)
        for method, rows in source_rows.items()
    }
    keys = [
        canonical(row["question"])
        for row in tests
        if test_counts[canonical(row["question"])] == 1
        and all(source_counts[method][canonical(row["question"])] == 1 for method in SOURCES)
    ]
    assert len(keys) == 2403, f"expected 2403 common unique questions, got {len(keys)}"
    included = set(keys)
    gold_by_key = {canonical(row["question"]): answer(row) for row in tests if canonical(row["question"]) in included}
    answers = [gold_by_key[key] for key in keys]
    model = SentenceTransformer(str(BASE.parents[1] / "models/paraphrase-mpnet-base-v2"), device="cpu")
    with torch.no_grad():
        answer_vectors = model.encode(answers, convert_to_tensor=True, batch_size=256)
    results = {}
    f1_vectors = {}
    for method, source in SOURCES.items():
        prediction_by_key = {
            canonical(row["question"]): parse_answer(row.get("pred") or row.get("res", ""))
            for row in source_rows[method]
            if source_counts[method][canonical(row.get("question", ""))] == 1
        }
        predictions = [prediction_by_key[key] for key in keys]
        result = metrics(model, predictions, answers, answer_vectors)
        result.update({"source": source, "source_sha256": sha256(BASE / source)})
        results[method] = result
        f1_vectors[method] = np.asarray([f1_score(prediction, gold) for prediction, gold in zip(predictions, answers)])
        print(method, result, flush=True)

    excluded = [
        {"index": index, "id": row.get("id") or row.get("_id"), "question": row["question"]}
        for index, row in enumerate(tests)
        if canonical(row["question"]) not in included
    ]
    payload = {
        "evaluator": "scripts/eval_table2_musique_paired.py",
        "selection": "canonical question unique in test and exactly once in every method prediction file",
        "test_n": len(tests),
        "paired_n": len(keys),
        "excluded_n": len(excluded),
        "excluded": excluded,
        "results": results,
    }
    target = BASE / "table-result/table2_musique_paired2403_metrics.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    significance = {
        "selection": payload["selection"],
        "paired_n": len(keys),
        "caliper_vs": {
            method: paired_bootstrap(f1_vectors["CALIPER"], values)
            for method, values in f1_vectors.items() if method != "CALIPER"
        },
        "adaptive_vs_naive": paired_bootstrap(
            f1_vectors["Adaptive-RAG (adapted)"], f1_vectors["Naive RAG (strict)"]
        ),
    }
    significance_target = BASE / "significance_analysis/table2_musique_paired2403_bootstrap.json"
    significance_target.write_text(json.dumps(significance, ensure_ascii=False, indent=2), encoding="utf-8")

    full = json.load((BASE / "table-result/table2_final_metrics.json").open(encoding="utf-8"))
    for method, result in results.items():
        full["results"][method]["musique_full_available"] = full["results"][method]["musique"]
        full["results"][method]["musique"] = result
    full.update({
        "evaluator": "scripts/eval_table2_final.py + scripts/eval_table2_musique_paired.py",
        "paper_policy": "MuSiQue uses the common paired n=2403 set; other datasets use their full test sets.",
    })
    paper_target = BASE / "table-result/table2_paper_final_metrics.json"
    paper_target.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target, paper_target, significance_target, flush=True)


if __name__ == "__main__":
    main()
