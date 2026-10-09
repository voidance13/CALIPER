"""Offline classifier threshold sensitivity; never selects a new test threshold."""
import json
import re
import statistics
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba


BASE = Path("/mnt/data/caliper/code/CALIPER")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.997, 0.999)
CALIPER = {
    "hotpotqa": "data/hotpotqa/hotpotqa_caliper_planA_top5.json",
    "musique": "data/musique/musique_v1_ubh_2stream.json",
    "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json",
}
ROUTER_MS = {"hotpotqa": 32.48, "musique": 32.05, "2wikimultihopqa": 32.13}
PUNCTUATION = set(string.punctuation)


def canonical(text):
    return " ".join("".join(char if char.isalnum() else " " for char in str(text).lower()).split())


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


def f1(prediction, gold):
    predicted, expected = normalize(parse_answer(prediction)).split(), normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summary(values):
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 3),
            "p50": round(statistics.median(values), 3),
            "p95": round(ordered[int(0.95 * len(ordered)) - 1], 3)}


def main():
    classifier = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
    stopwords = {line.strip() for line in (BASE / "query_clf/stopwords.txt").open(encoding="utf-8")}

    def probability(question):
        cleaned = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
        labels, probabilities = classifier.predict(cleaned, k=2)
        return dict(zip((label.removeprefix("__label__") for label in labels), probabilities)).get("easy", 0.0)

    validation = []
    for line in (BASE / "query_clf/train_new/9b_valid.txt").open(encoding="utf-8"):
        label, text = line.rstrip().split(" ", 1)
        labels, probabilities = classifier.predict(text, k=2)
        validation.append((label.removeprefix("__label__"),
                           dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)))

    output = {
        "status": "descriptive test sensitivity; frozen operating threshold remains 0.997",
        "thresholds": list(THRESHOLDS), "latency_type": "counterfactual from matched measured base/CALIPER rows plus router mean",
        "datasets": {},
    }
    excluded = {row["index"] for row in json.load(
        (BASE / "table-result/table2_musique_paired2403_metrics.json").open(encoding="utf-8"))["excluded"]}
    for dataset in DATASETS:
        test = json.load((BASE / f"data/{dataset}/test.json").open(encoding="utf-8"))
        indices = [index for index in range(len(test)) if dataset != "musique" or index not in excluded]
        questions = [test[index]["question"] for index in indices]
        gold = {row["question"]: (row.get("answer") if isinstance(row.get("answer"), str)
                                   else (row.get("answers") or [""])[0]) for row in test}
        baseline = {row["question"]: row.get("pred", "") for row in json.load(
            (BASE / f"data/{dataset}/{dataset}_baseline_2stage_soft.json").open(encoding="utf-8"))}
        caliper = {row["question"]: row.get("pred", "") for row in json.load((BASE / CALIPER[dataset]).open(encoding="utf-8"))}
        assert len(questions) == len(set(questions)) and all(q in baseline and q in caliper for q in questions)
        probabilities = {q: probability(q) for q in set(questions)}

        timing = json.load((BASE / f"table-result/table4_timing_{dataset}_seed42.json").open(encoding="utf-8"))
        timing_rows = {row["question"]: row for row in timing["rows"]}
        actual = json.load((BASE / f"table-result/table4_clf_actual_{dataset}_seed42.json").open(encoding="utf-8"))
        curve = []
        for threshold in THRESHOLDS:
            local = {q for q in questions if probabilities[q] >= threshold}
            scores = [f1(baseline[q] if q in local else caliper[q], gold[q]) for q in questions]
            latency, errors = [], 0
            for question, row in timing_rows.items():
                if question not in probabilities:
                    continue
                selected = row["baseline"] if probabilities[question] >= threshold else row["caliper"]
                if isinstance(selected, dict):
                    if selected["error"] is None:
                        latency.append(selected["seconds"] + ROUTER_MS[dataset] / 1000)
                    else:
                        errors += 1
                else:
                    latency.append(selected + ROUTER_MS[dataset] / 1000)
            validation_local = sum(prob >= threshold for _, prob in validation)
            hard = [(label, prob) for label, prob in validation if label == "hard"]
            hard_to_local = sum(prob >= threshold for _, prob in hard) / len(hard)
            point = {
                "threshold": threshold, "test_n": len(questions), "F1": round(statistics.mean(scores) * 100, 2),
                "local_rate": round(len(local) / len(questions) * 100, 2),
                "cloud_call_rate": round((1 - len(local) / len(questions)) * 100, 2),
                "counterfactual_latency_seconds": summary(latency) | {"errors": errors},
                "validation_local_rate": round(validation_local / len(validation) * 100, 2),
                "validation_hard_to_local_rate": round(hard_to_local * 100, 2),
            }
            if threshold == 0.997:
                valid = [row["seconds"] for row in actual["rows"] if row["error"] is None]
                point["actual_measured_latency_seconds"] = summary(valid) | {
                    "errors": len(actual["rows"]) - len(valid)}
            curve.append(point)
        output["datasets"][dataset] = {"curve": curve, "caliper_source": CALIPER[dataset]}
        frozen = next(point for point in curve if point["threshold"] == 0.997)
        print(dataset, json.dumps(curve, ensure_ascii=False), flush=True)
        expected = {"hotpotqa": 69.76, "musique": 49.39, "2wikimultihopqa": 73.63}[dataset]
        assert abs(frozen["F1"] - expected) <= 0.02, (dataset, frozen["F1"], expected)

    target = BASE / "classifier_analysis/classifier_threshold_pareto.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED", target)


if __name__ == "__main__":
    main()
