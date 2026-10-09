"""Rebuild Table 3 on full, question-deduplicated test sets."""
import hashlib
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fasttext
import jieba
import torch
import torch.nn.functional as functional
from sentence_transformers import SentenceTransformer


BASE = Path("/mnt/data/caliper/code/CALIPER")
THRESHOLD = 0.997
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
SOURCES = {
    "Qwen3.5-9B": {
        "hotpotqa": "data/hotpotqa/normal_pred_7b_sep_use_best_hit.json",
        "musique": "data/musique/normal_pred_7b_sep_use_best_hit_single_full.json",
        "2wikimultihopqa": "data/2wikimultihopqa/normal_pred_7b_sep_use_best_hit.json",
    },
    "CALIPER": {
        "hotpotqa": "data/hotpotqa/hybrid_pred_7b_72b_sep_use_best_hit_instr_extract_full.json",
        "musique": "data/musique/hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
        "2wikimultihopqa": "data/2wikimultihopqa/hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
    },
    "Qwen3.7-plus": {
        "hotpotqa": "data/hotpotqa/normal_pred_72b_sep_use_best_hit.json",
        "musique": "data/musique/normal_pred_72b_sep_use_best_hit_merge.json",
        "2wikimultihopqa": "data/2wikimultihopqa/normal_pred_72b_sep_use_best_hit_merge.json",
    },
}
PUNCTUATION = set(string.punctuation)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in PUNCTUATION).split())


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def deduplicate(rows):
    unique = {}
    for row in rows:
        unique.setdefault(row["question"], row)
    return unique


def lexical_metrics(pairs):
    hit = precision = recall = f1 = 0.0
    for prediction, gold in pairs:
        predicted, expected = normalize(prediction), normalize(gold)
        hit += int(bool(expected) and expected in predicted)
        predicted_tokens, expected_tokens = predicted.split(), expected.split()
        common = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
        p = common / len(predicted_tokens) if predicted_tokens else 0.0
        r = common / len(expected_tokens) if expected_tokens else 0.0
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if p + r else 0.0
    n = len(pairs)
    return {"Hit": hit / n * 100, "P": precision / n * 100,
            "R": recall / n * 100, "F1": f1 / n * 100}


def evaluate(model, pairs):
    metrics = lexical_metrics(pairs)
    with torch.no_grad():
        predictions = model.encode([x[0] for x in pairs], convert_to_tensor=True, batch_size=256)
        answers = model.encode([x[1] for x in pairs], convert_to_tensor=True, batch_size=256)
        metrics["SBERT"] = float(functional.cosine_similarity(predictions, answers).mean()) * 100
    metrics = {key: round(metrics[key], 2) for key in ("SBERT", "Hit", "P", "R", "F1")}
    metrics["n"] = len(pairs)
    assert metrics["F1"] <= (metrics["P"] + metrics["R"]) / 2 + 0.02
    return metrics


def main():
    classifier_path = BASE / "query_clf/train_new/9b_clf.bin"
    stopwords_path = BASE / "query_clf/stopwords.txt"
    classifier = fasttext.load_model(str(classifier_path))
    stopwords = {line.strip() for line in stopwords_path.open(encoding="utf-8")}
    model = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cuda:0")
    result = {
        "status": "full-set unified evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "scripts/eval_table3_full.py",
        "deduplication": "first occurrence per exact question; all methods use the same questions",
        "classifier": {"model": str(classifier_path.relative_to(BASE)), "threshold": THRESHOLD,
                       "model_sha256": sha256(classifier_path), "stopwords_sha256": sha256(stopwords_path)},
        "datasets": {},
    }
    for dataset in DATASETS:
        test_path = BASE / f"data/{dataset}/test.json"
        gold_rows = json.load(test_path.open(encoding="utf-8"))
        gold = deduplicate(gold_rows)
        questions = list(gold)
        source_rows = {}
        sources = {}
        for method, files in SOURCES.items():
            path = BASE / files[dataset]
            rows = json.load(path.open(encoding="utf-8"))
            unique = deduplicate(rows)
            assert set(unique) == set(questions), (dataset, method, len(unique), len(questions))
            source_rows[method] = unique
            sources[method] = {"path": files[dataset], "sha256": sha256(path),
                               "raw_n": len(rows), "unique_n": len(unique)}

        local_questions = set()
        for question in questions:
            cleaned = " ".join(token for token in jieba.lcut(question)
                               if len(token) > 1 and token not in stopwords)
            labels, probabilities = classifier.predict(cleaned, k=2)
            easy_probability = dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0)
            if easy_probability >= THRESHOLD:
                local_questions.add(question)

        methods = {}
        for method in SOURCES:
            pairs = [(parse_answer(source_rows[method][q].get("pred") or source_rows[method][q].get("res", "")),
                      gold[q].get("answer") if isinstance(gold[q].get("answer"), str) else (gold[q].get("answers") or [""])[0])
                     for q in questions]
            methods[method] = evaluate(model, pairs)
        routed_pairs = []
        for question in questions:
            method = "Qwen3.5-9B" if question in local_questions else "CALIPER"
            row = source_rows[method][question]
            target = gold[question].get("answer") if isinstance(gold[question].get("answer"), str) else (gold[question].get("answers") or [""])[0]
            routed_pairs.append((parse_answer(row.get("pred") or row.get("res", "")), target))
        methods["CALIPER+clf"] = evaluate(model, routed_pairs) | {
            "local_n": len(local_questions), "collaborative_n": len(questions) - len(local_questions),
            "local_rate": round(len(local_questions) / len(questions) * 100, 2),
        }
        result["datasets"][dataset] = {
            "raw_test_n": len(gold_rows), "unique_test_n": len(questions),
            "duplicate_n": len(gold_rows) - len(questions), "sources": sources, "methods": methods,
        }
        print(dataset, json.dumps(methods, ensure_ascii=False), flush=True)

    assert [result["datasets"][d]["unique_test_n"] for d in DATASETS] == [7405, 2412, 12576]
    target = BASE / "table-result/table3_full_unified_metrics.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target)


if __name__ == "__main__":
    main()
