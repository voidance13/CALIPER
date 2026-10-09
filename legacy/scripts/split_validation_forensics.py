import json
from collections import Counter
from pathlib import Path

import fasttext
import jieba


ROOT = Path("query_clf/train")
STOPWORDS = {word.strip() for word in Path("query_clf/stopwords.txt").open()}


def clean(text):
    return " ".join(word for word in jieba.lcut(text) if len(word) > 1 and word not in STOPWORDS)


def source_lines(dataset):
    rows = json.load((ROOT / f"{dataset}10000.json").open())
    return {f"__label__{'easy' if row['intent_type'] == 'easy' else 'hard'} {clean(row['question'])}" for row in rows}


def evaluate(model, rows):
    labels, texts = zip(*(row.split(" ", 1) for row in rows))
    predictions = model.predict(list(texts))[0]
    return sum(label in prediction for label, prediction in zip(labels, predictions)) / len(rows)


def main():
    domains = {name: source_lines(name) for name in ("hotpotqa", "musique", "2wikimultihopqa")}
    cases = [
        ("joint", ROOT / "train_bi.bin", ROOT / "valid_bi.txt"),
        ("with_2wiki", ROOT / "train_bi_2wiki.bin", ROOT / "valid_bi_2wiki.txt"),
    ]
    output = {}
    for case, model_path, valid_path in cases:
        model = fasttext.load_model(str(model_path))
        valid = [line.strip() for line in valid_path.open()]
        output[case] = {}
        membership = Counter()
        for dataset, source in domains.items():
            rows = [line for line in valid if line in source]
            membership[dataset] = len(rows)
            if rows:
                output[case][dataset] = {"n": len(rows), "accuracy": evaluate(model, rows)}
        matched = {line for source in domains.values() for line in valid if line in source}
        output[case]["unmatched"] = len(valid) - len(matched)
        print(case, output[case], flush=True)
    model = fasttext.load_model("query_clf/train_new/9b_clf.bin")
    valid = [line.strip() for line in Path("query_clf/train_new/9b_valid.txt").open()]
    test_domains = {}
    for dataset in domains:
        test_domains[dataset] = {clean(row["question"]) for row in json.load(open(f"data/{dataset}/test.json"))}
    output["9b_heldout"] = {}
    matched = set()
    for dataset, questions in test_domains.items():
        rows = [line for line in valid if line.split(" ", 1)[1] in questions]
        matched.update(rows)
        output["9b_heldout"][dataset] = {"n": len(rows), "accuracy": evaluate(model, rows) if rows else None}
    output["9b_heldout"]["unmatched"] = len(valid) - len(matched)
    print("9b_heldout", output["9b_heldout"], flush=True)
    Path("classifier_analysis/split_validation_forensics.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
