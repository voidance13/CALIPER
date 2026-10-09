#!/usr/bin/env python3
"""Train and audit a held-out three-way benchmark-source router."""

import hashlib
import json
import random
from pathlib import Path

import fasttext
import jieba


ROOT = Path("/mnt/data/caliper/code/CALIPER")
OUT = ROOT / "query_clf/heldout_source_router_20260917"
LABELS = {"hotpotqa": "hp", "musique": "ms", "2wikimultihopqa": "tw"}
STOPWORDS = {line.strip() for line in (ROOT / "query_clf/stopwords.txt").open(encoding="utf-8") if line.strip()}


def clean(text):
    return " ".join(token for token in jieba.lcut(text) if len(token) > 1 and token not in STOPWORDS)


def canonical(text):
    return " ".join(text.lower().split())


def load_rows(dataset, split):
    if split == "official_dev":
        path = Path("/tmp/musique_official_dev_full.json")
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [row for row in rows if any(paragraph.get("is_supporting") for paragraph in row["paragraphs"])]
    return json.loads((ROOT / f"data/{dataset}/{split}.json").read_text(encoding="utf-8"))


def line(label, row):
    return f"__label__{label} {clean(row['question'])}\n"


def stable_bucket(dataset, row):
    key = f"{dataset}\0{row.get('id', row['question'])}"
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.8 else "valid" if value < 0.9 else "holdout"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    random.seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    train_lines, valid_lines, holdout_lines = [], [], []
    test_lines = []
    manifest = {"seed": 42, "split_rule": "sha256(dataset + NUL + id/question): train <0.8, valid <0.9, holdout otherwise", "datasets": {}}

    for dataset, label in LABELS.items():
        train_rows = load_rows(dataset, "train")
        test_rows = load_rows(dataset, "official_dev" if dataset == "musique" else "test")
        train_questions = {canonical(row["question"]) for row in train_rows}
        test_questions = {canonical(row["question"]) for row in test_rows}
        overlap = sorted(train_questions & test_questions)
        buckets = {"train": 0, "valid": 0, "holdout": 0}
        for row in train_rows:
            bucket = stable_bucket(dataset, row)
            buckets[bucket] += 1
            target = {"train": train_lines, "valid": valid_lines, "holdout": holdout_lines}[bucket]
            target.append(line(label, row))
        test_lines.extend(line(label, row) for row in test_rows)
        manifest["datasets"][dataset] = {
            "train_source_rows": len(train_rows),
            "router_train_valid_holdout": buckets,
            "router_test_rows": len(test_rows),
            "train_test_question_overlap": len(overlap),
            "overlap_examples": overlap[:5],
        }

    random.shuffle(train_lines)
    random.shuffle(valid_lines)
    random.shuffle(holdout_lines)
    random.shuffle(test_lines)
    paths = {
        "train": OUT / "router_train.txt",
        "valid": OUT / "router_valid.txt",
        "holdout": OUT / "router_train_holdout.txt",
        "test": OUT / "router_test_official.txt",
    }
    for key, lines in (("train", train_lines), ("valid", valid_lines), ("holdout", holdout_lines), ("test", test_lines)):
        paths[key].write_text("".join(lines), encoding="utf-8")

    model = fasttext.train_supervised(
        input=str(paths["train"]),
        epoch=37,
        lr=0.05,
        dim=160,
        ws=5,
        wordNgrams=2,
        minCount=1,
        minn=0,
        maxn=0,
        bucket=4110692,
        loss="softmax",
        thread=12,
        seed=42,
    )
    model_path = OUT / "domain_clf.bin"
    model.save_model(str(model_path))
    manifest["files"] = {key: {"path": str(path), "sha256": sha256(path), "rows": sum(1 for _ in path.open(encoding="utf-8"))} for key, path in paths.items()}
    manifest["model"] = {"path": str(model_path), "sha256": sha256(model_path), "train_metrics": model.test(str(paths["train"])), "valid_metrics": model.test(str(paths["valid"])), "train_holdout_metrics": model.test(str(paths["holdout"])), "official_test_metrics": model.test(str(paths["test"]))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
