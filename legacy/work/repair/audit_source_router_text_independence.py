#!/usr/bin/env python3
"""Audit source-router split independence at the actual model-input text level.

This is audit-only: it does not overwrite the existing router files or model.
"""

import hashlib
import json
from pathlib import Path

import jieba


ROOT = Path("/mnt/data/caliper/code/CALIPER")
DATASETS = {"hotpotqa": "hp", "musique": "ms", "2wikimultihopqa": "tw"}
STOPWORDS = {
    line.strip()
    for line in (ROOT / "query_clf/stopwords.txt").open(encoding="utf-8")
    if line.strip()
}


def clean(text: str) -> str:
    return " ".join(token for token in jieba.lcut(text) if len(token) > 1 and token not in STOPWORDS)


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_rows(target, dataset, label, rows, split):
    for row in rows:
        target.append(
            {
                "dataset": dataset,
                "label": label,
                "id": row.get("id"),
                "split": split,
                "text": norm(clean(row["question"])),
            }
        )


def bucket(text: str) -> str:
    value = int(hashlib.sha256(("global\0" + text).encode("utf-8")).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.8 else "valid" if value < 0.9 else "holdout"


def main() -> None:
    train_rows = []
    test_rows = []
    for dataset, label in DATASETS.items():
        add_rows(train_rows, dataset, label, load_json(ROOT / f"data/{dataset}/train.json"), "source_train")
        if dataset == "musique":
            rows = [
                row
                for row in load_json(Path("/tmp/musique_official_dev_full.json"))
                if any(paragraph.get("is_supporting") for paragraph in row["paragraphs"])
            ]
        else:
            rows = load_json(ROOT / f"data/{dataset}/test.json")
        add_rows(test_rows, dataset, label, rows, "official_eval")

    train_by = {}
    test_by = {}
    for row in train_rows:
        train_by.setdefault(row["text"], []).append(row)
    for row in test_rows:
        test_by.setdefault(row["text"], []).append(row)

    train_test_texts = sorted(set(train_by) & set(test_by))
    remaining = [row for row in train_rows if row["text"] not in train_test_texts]
    groups = {}
    for row in remaining:
        groups.setdefault(row["text"], []).append(row)

    candidate_sets = {"train": set(), "valid": set(), "holdout": set(), "test": set(test_by)}
    candidate_rows = {"train": 0, "valid": 0, "holdout": 0}
    candidate_groups = {"train": 0, "valid": 0, "holdout": 0}
    for text, rows in groups.items():
        split = bucket(text)
        candidate_sets[split].add(text)
        candidate_rows[split] += len(rows)
        candidate_groups[split] += 1

    overlap = {}
    for left in candidate_sets:
        for right in candidate_sets:
            if left < right:
                overlap[f"{left}__{right}"] = len(candidate_sets[left] & candidate_sets[right])

    report = {
        "scope": "audit-only; existing router artifacts are untouched",
        "current_source_train_rows": len(train_rows),
        "current_source_train_unique_clean_texts": len(train_by),
        "current_source_train_duplicate_rows": len(train_rows) - len(train_by),
        "official_eval_rows": len(test_rows),
        "official_eval_unique_clean_texts": len(test_by),
        "official_eval_duplicate_rows": len(test_rows) - len(test_by),
        "train_texts_in_official_eval": len(train_test_texts),
        "train_rows_removed_in_candidate_policy": len(train_rows) - len(remaining),
        "candidate_group_counts": candidate_groups,
        "candidate_row_counts": candidate_rows,
        "candidate_text_overlap": overlap,
        "train_test_overlap_examples": train_test_texts[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
