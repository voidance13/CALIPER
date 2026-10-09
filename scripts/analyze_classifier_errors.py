import json
import string
from collections import Counter
from pathlib import Path

import fasttext


ROOT = Path("query_clf")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
ROUTED = {
    "hotpotqa": Path("data/hotpotqa/hybrid_pred_7b_72b_sep_clf_t2caliper_top10.json"),
    "musique": Path("data/musique/hybrid_pred_7b_72b_sep_clf_t2caliper_v2.json"),
    "2wikimultihopqa": Path("data/2wikimultihopqa/hybrid_pred_7b_72b_sep_clf_t2caliper_top10.json"),
}


def load_rows(dataset):
    rows = []
    for level in ("easy", "hard"):
        for line in (ROOT / f"train/{dataset}_test2000_{level}_bi.txt").read_text().splitlines():
            label, text = line.split(" ", 1)
            rows.append({"true": label.removeprefix("__label__"), "question": text})
    return rows


def safe_div(a, b):
    return a / b if b else 0.0


def qa_f1(prediction, answer):
    def tokens(text):
        text = "".join(ch for ch in str(text).lower() if ch not in string.punctuation)
        return [word for word in text.split() if word not in {"a", "an", "the"}]
    pred, gold = tokens(prediction), tokens(answer)
    common = sum((Counter(pred) & Counter(gold)).values())
    return 2 * common / (len(pred) + len(gold)) if common else float(pred == gold)


def canonical_question(text):
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(text).lower()).split())


def analyze(model_path, rows, routed_path=None):
    model = fasttext.load_model(str(model_path))
    for row in rows:
        labels, probabilities = model.predict(row["question"], k=2)
        row["pred"] = labels[0].removeprefix("__label__")
        row["confidence"] = float(probabilities[0])
    confusion = Counter((row["true"], row["pred"]) for row in rows)
    classes = ("easy", "hard")
    per_class = {}
    for label in classes:
        tp = confusion[label, label]
        fp = sum(confusion[other, label] for other in classes if other != label)
        fn = sum(confusion[label, other] for other in classes if other != label)
        precision, recall = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "support": sum(confusion[label, pred] for pred in classes),
        }
    thresholds = {}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        kept = [row for row in rows if row["confidence"] >= threshold]
        thresholds[str(threshold)] = {
            "coverage": safe_div(len(kept), len(rows)),
            "accuracy": safe_div(sum(row["true"] == row["pred"] for row in kept), len(kept)),
            "count": len(kept),
        }
    errors = sorted((row for row in rows if row["true"] != row["pred"]), key=lambda x: -x["confidence"])
    prefix_groups = {}
    for prefix in sorted({row["question"].split()[0].lower() for row in rows if row["question"].split()}):
        group = [row for row in rows if row["question"].split() and row["question"].split()[0].lower() == prefix]
        if len(group) >= 20:
            prefix_groups[prefix] = {"n": len(group), "accuracy": safe_div(sum(row["true"] == row["pred"] for row in group), len(group))}
    result = {
        "model": str(model_path),
        "n": len(rows),
        "accuracy": safe_div(sum(row["true"] == row["pred"] for row in rows), len(rows)),
        "predicted_distribution": dict(Counter(row["pred"] for row in rows)),
        "confusion_true_by_pred": {
            true: {pred: confusion[true, pred] for pred in classes} for true in classes
        },
        "per_class": per_class,
        "confidence_thresholds": thresholds,
        "accuracy_by_question_prefix_min20": prefix_groups,
        "high_confidence_errors": errors[:20],
    }
    if routed_path:
        routed = {canonical_question(row["question"]): row for row in json.load(routed_path.open())}
        groups = {}
        for true in classes:
            for pred in classes:
                matched = [row for row in rows if row["true"] == true and row["pred"] == pred and canonical_question(row["question"]) in routed]
                scores = [qa_f1(routed[canonical_question(row["question"])].get("pred", ""), routed[canonical_question(row["question"])].get("answer", "")) for row in matched]
                groups[f"true_{true}__pred_{pred}"] = {"n": len(scores), "qa_f1": safe_div(sum(scores), len(scores))}
        result["downstream_routed_qa_by_confusion_group"] = {
            "prediction_file": str(routed_path), "groups": groups,
            "matched": sum(group["n"] for group in groups.values()),
        }
    return result


def main():
    output = {"label_definition": "easy vs hard (middle merged into hard by original data builder)"}
    for dataset in DATASETS:
        rows = load_rows(dataset)
        deployed = ROOT / ("train/train_bi_2wiki.bin" if dataset == "2wikimultihopqa" else "train/train_bi.bin")
        output[dataset] = {
            "deployed_table2_router": analyze(deployed, [dict(row) for row in rows], ROUTED[dataset]),
            "newer_9b_labeled_router": analyze(ROOT / "train_new/9b_clf.bin", [dict(row) for row in rows]),
        }
    target = Path("classifier_analysis")
    target.mkdir(exist_ok=True)
    (target / "classifier_error_analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    for dataset in DATASETS:
        for name, result in output[dataset].items():
            print(dataset, name, f"n={result['n']} acc={result['accuracy']:.4%}", result["confusion_true_by_pred"])


if __name__ == "__main__":
    main()
