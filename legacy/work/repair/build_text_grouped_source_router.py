#!/usr/bin/env python3
"""Build and train a text-grouped source router without overwriting old artifacts."""

import hashlib
import json
from pathlib import Path

import fasttext
import jieba


ROOT = Path("/mnt/data/caliper/code/CALIPER")
OUT = ROOT / "query_clf/text_grouped_source_router_20260917"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bucket(text: str) -> str:
    value = int(hashlib.sha256(("global\0" + text).encode("utf-8")).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.8 else "valid" if value < 0.9 else "holdout"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_train = []
    official_eval = []
    for dataset, label in DATASETS.items():
        for row in load(ROOT / f"data/{dataset}/train.json"):
            cleaned = norm(clean(row["question"]))
            source_train.append({"dataset": dataset, "label": label, "id": row.get("id"), "text": cleaned})
        if dataset == "musique":
            rows = [
                row
                for row in load(Path("/tmp/musique_official_dev_full.json"))
                if any(paragraph.get("is_supporting") for paragraph in row["paragraphs"])
            ]
        else:
            rows = load(ROOT / f"data/{dataset}/test.json")
        for row in rows:
            cleaned = norm(clean(row["question"]))
            official_eval.append({"dataset": dataset, "label": label, "id": row.get("id"), "text": cleaned})

    test_texts = {row["text"] for row in official_eval}
    groups = {}
    for row in source_train:
        groups.setdefault(row["text"], []).append(row)

    train_lines = {"train": [], "valid": [], "holdout": []}
    assignments = []
    excluded = []
    conflicting_groups = []
    for text, rows in groups.items():
        labels = sorted({row["label"] for row in rows})
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text in test_texts:
            split = "excluded_train_test_text_overlap"
            excluded.extend(rows)
        elif len(labels) > 1:
            split = "excluded_conflicting_source_labels"
            conflicting_groups.append({"text_sha256": text_hash, "labels": labels, "rows": rows})
            excluded.extend(rows)
        else:
            split = bucket(text)
            train_lines[split].extend(f"__label__{rows[0]['label']} {text}\n" for _ in rows)
        for row in rows:
            assignments.append({**row, "text_sha256": text_hash, "split": split})

    test_lines = [f"__label__{row['label']} {row['text']}\n" for row in official_eval]
    for key in train_lines:
        (OUT / f"router_{key}.txt").write_text("".join(train_lines[key]), encoding="utf-8")
    (OUT / "router_test_official.txt").write_text("".join(test_lines), encoding="utf-8")
    with (OUT / "split_assignments.jsonl").open("w", encoding="utf-8") as handle:
        for row in assignments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "official_eval_assignments.jsonl").open("w", encoding="utf-8") as handle:
        for row in official_eval:
            handle.write(json.dumps({**row, "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(), "split": "official_eval"}, ensure_ascii=False) + "\n")

    model = fasttext.train_supervised(
        input=str(OUT / "router_train.txt"), epoch=37, lr=0.05, dim=160, ws=5,
        wordNgrams=2, minCount=1, minn=0, maxn=0, bucket=4110692,
        loss="softmax", thread=12, seed=42,
    )
    model.save_model(str(OUT / "domain_clf.bin"))

    split_sets = {}
    for key in ("train", "valid", "holdout"):
        split_sets[key] = {row["text"] for row in assignments if row["split"] == key}
    split_sets["official_eval"] = {row["text"] for row in official_eval}
    overlap = {}
    for left in split_sets:
        for right in split_sets:
            if left < right:
                overlap[f"{left}__{right}"] = len(split_sets[left] & split_sets[right])

    files = {
        name: OUT / filename
        for name, filename in {
            "train": "router_train.txt", "valid": "router_valid.txt",
            "holdout": "router_holdout.txt", "test": "router_test_official.txt",
            "split_assignments": "split_assignments.jsonl",
            "official_eval_assignments": "official_eval_assignments.jsonl",
            "model": "domain_clf.bin",
        }.items()
    }
    manifest = {
        "policy": "global normalized model-input text grouping; training rows matching official evaluation text are excluded; existing artifacts untouched",
        "seed": 42,
        "split_rule": "sha256(global + NUL + normalized cleaned question): train <0.8, valid <0.9, holdout otherwise",
        "source_train_rows": len(source_train),
        "official_eval_rows": len(official_eval),
        "source_train_rows_excluded": len(excluded),
        "conflicting_source_text_groups_excluded": len(conflicting_groups),
        "split_rows": {key: len(train_lines[key]) for key in train_lines},
        "official_eval_unique_texts": len({row["text"] for row in official_eval}),
        "text_overlap_matrix": overlap,
        "files": {
            name: {
                "path": str(path),
                **({"rows": sum(1 for _ in path.open(encoding="utf-8"))}
                   if name != "model" else {}),
                "sha256": sha256(path),
            }
            for name, path in files.items()
        },
        "metrics": {
            "train": model.test(str(files["train"])),
            "valid": model.test(str(files["valid"])),
            "holdout": model.test(str(files["holdout"])),
            "official_eval": model.test(str(files["test"])),
        },
    }
    manifest["excluded_examples"] = excluded[:10]
    manifest["conflicting_examples"] = conflicting_groups[:3]
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
