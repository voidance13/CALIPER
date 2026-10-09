"""Recompute Figure 3 threshold sensitivity with the overlap-free classifier.

This is an offline audit: it reads frozen prediction caches and does not call
models for generation or any cloud endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba


BASE = Path("/mnt/data/caliper/code/CALIPER")
CLASSIFIER = BASE / "classifier_analysis/overlap_free_retrain_20260927/9b_clf_overlap_free.bin"
STOPWORDS = BASE / "query_clf/stopwords.txt"
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.97, 0.974, 0.977, 0.978, 0.99, 0.995, 0.997, 0.999)
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
PATHS = {
    "hotpotqa": {
        "test": BASE / "data/hotpotqa/test.json",
        "base": BASE / "classifier_analysis/overlap_free_retrain_20260927/frozen_archive_members/predictions/data/hotpotqa/hotpotqa_baseline_2stage_soft.json",
        "core": BASE / "router_e2e_runs/text_grouped_20260918/predictions/hotpotqa_v1_router_textgrouped_20260918.json",
    },
    "musique": {
        "test": BASE / "data/musique/test.json",
        "base": BASE / "classifier_analysis/overlap_free_retrain_20260927/frozen_source_cache/musique_baseline_frozen_table2.json",
        "core": BASE / "classifier_analysis/overlap_free_retrain_20260927/frozen_source_cache/musique_v1_frozen_table2.json",
    },
    "2wikimultihopqa": {
        "test": BASE / "data/2wikimultihopqa/test.json",
        "base": BASE / "classifier_analysis/overlap_free_retrain_20260927/frozen_archive_members/predictions/data/2wikimultihopqa/2wikimultihopqa_baseline_2stage_soft.json",
        "core": BASE / "router_e2e_runs/text_grouped_20260918/predictions/2wikimultihopqa_v1_router_textgrouped_20260918.json",
    },
}
PUNCT = set(string.punctuation)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())


def answer(text: str) -> str:
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text)
        if match:
            return text[match.end():]
    line = text.splitlines()[-1] if text.splitlines() else ""
    match = re.search(": ", line)
    return line[match.end():] if match else line


def normalize(text: str) -> list[str]:
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(c for c in text if c not in PUNCT).split()).split()


def values(prediction: str, gold: str) -> tuple[float, float, float, float]:
    predicted = normalize(answer(prediction))
    expected = normalize(gold)
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hit = float(bool(expected) and " ".join(expected) in " ".join(predicted))
    return hit, precision, recall, f1


def prediction(row: dict) -> str:
    value = row.get("pred")
    if value is None:
        value = row.get("res", "")
    return value or ""


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classifier = fasttext.load_model(str(CLASSIFIER))
    stopwords = {line.strip() for line in STOPWORDS.open(encoding="utf-8")}

    def probability(question: str) -> float:
        cleaned = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
        labels, probabilities = classifier.predict(cleaned, k=2)
        labels = [label.removeprefix("__label__") for label in labels]
        return float(dict(zip(labels, probabilities)).get("easy", 0.0))

    result = {
        "status": "VERIFICATION_CANDIDATE_NOT_FROZEN",
        "model": "9b_clf_overlap_free.bin",
        "model_path": str(CLASSIFIER),
        "thresholds": list(THRESHOLDS),
        "protocol": "For each threshold, select BASE when p_easy >= threshold and CALIPER core otherwise; score fixed cached predictions with the paper's exact-match/token P/R/F1 implementation. Cloud-call rate is 1 - local route rate.",
        "no_generation_or_cloud_calls": True,
        "datasets": {},
        "input_sha256": {str(CLASSIFIER): sha256(CLASSIFIER), str(STOPWORDS): sha256(STOPWORDS)},
    }

    for dataset in DATASETS:
        paths = PATHS[dataset]
        tests = load_rows(paths["test"])
        base_rows = load_rows(paths["base"])
        core_rows = load_rows(paths["core"])
        gold = {
            canonical(row["question"]): (row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0])
            for row in tests
        }
        base = {canonical(row["question"]): prediction(row) for row in base_rows}
        core = {canonical(row["question"]): prediction(row) for row in core_rows}
        questions = [canonical(row["question"]) for row in core_rows]
        question_text = {canonical(row["question"]): row["question"] for row in core_rows}
        if len(questions) != len(set(questions)):
            raise RuntimeError(f"{dataset}: core cache contains duplicate canonical questions")
        if not all(question in gold and question in base for question in questions):
            raise RuntimeError(f"{dataset}: cache/test question mismatch")
        probabilities = {question: probability(question_text[question]) for question in questions}
        curves = []
        for threshold in THRESHOLDS:
            local = [question for question in questions if probabilities[question] >= threshold]
            scores = [values(base[question] if question in local else core[question], gold[question]) for question in questions]
            means = [sum(score[index] for score in scores) / len(scores) * 100 for index in range(4)]
            curves.append({
                "threshold": threshold,
                "n": len(questions),
                "local_n": len(local),
                "local_rate_pct": round(len(local) / len(questions) * 100, 4),
                "cloud_call_rate_pct": round((1 - len(local) / len(questions)) * 100, 4),
                "Hit": round(means[0], 4),
                "P": round(means[1], 4),
                "R": round(means[2], 4),
                "F1": round(means[3], 4),
            })
        result["datasets"][dataset] = {"n": len(questions), "curves": curves}
        for path in paths.values():
            result["input_sha256"][str(path)] = sha256(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
