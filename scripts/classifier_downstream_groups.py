"""Current 9b router confusion groups and downstream QA impact."""
import json
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba


ROOT = Path("/mnt/data/caliper/code/CALIPER")
DOMAINS = ("hotpotqa", "musique", "2wikimultihopqa")
CALIPER = {
    "hotpotqa": "data/hotpotqa/hotpotqa_caliper_planA_top5.json",
    "musique": "data/musique/musique_v1_ubh_2stream.json",
    "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json",
}
THRESHOLD = 0.997
model = fasttext.load_model(str(ROOT / "query_clf/train_new/9b_clf.bin"))
stopwords = {line.strip() for line in (ROOT / "query_clf/stopwords.txt").open(encoding="utf-8")}
punctuation = set(string.punctuation)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def clean(text):
    return " ".join(token for token in jieba.lcut(text) if len(token) > 1 and token not in stopwords)


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in punctuation).split())


def answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        if marker in (text or ""):
            return text.split(marker, 1)[1]
    return (text or "").splitlines()[-1] if (text or "").splitlines() else ""


def f1(prediction, gold):
    predicted, expected = normalize(answer(prediction)).split(), normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0
    recall = common / len(expected) if expected else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def probability(question):
    labels, probabilities = model.predict(clean(question), k=2)
    return dict(zip((label.removeprefix("__label__") for label in labels), probabilities)).get("easy", 0)


def interval(values, seed):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    return [round(means[124] * 100, 2), round(means[4874] * 100, 2)]


valid = set((ROOT / "query_clf/train_new/9b_valid.txt").read_text(encoding="utf-8").splitlines())
output = {"classifier": "query_clf/train_new/9b_clf.bin", "threshold": THRESHOLD,
          "label": "easy iff all three Qwen3.5-9B baseline runs contain normalized gold", "datasets": {}}
for domain_index, domain in enumerate(DOMAINS):
    tests = load(f"data/{domain}/test.json")
    gold = {row["question"]: row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0] for row in tests}
    runs = [dict(load(f"query_clf/{domain}_base_sample{seed}.json")) for seed in range(1, 4)]
    labels = {}
    for question, target in list(gold.items())[:2000]:
        if all(question in run for run in runs):
            label = "easy" if all(normalize(target) in normalize(run[question]) for run in runs) else "hard"
            if domain == "2wikimultihopqa" or f"__label__{label} {clean(question)}" in valid:
                labels[question] = label
    baseline = {row["question"]: row.get("pred", "") for row in load(f"data/{domain}/{domain}_baseline_2stage_soft.json")}
    caliper = {row["question"]: row.get("pred", "") for row in load(CALIPER[domain])}
    groups = {}
    for question, true in labels.items():
        route = "local" if probability(question) >= THRESHOLD else "collaborative"
        key = f"true_{true}__route_{route}"
        groups.setdefault(key, []).append({
            "selected": f1(baseline[question] if route == "local" else caliper[question], gold[question]),
            "local": f1(baseline[question], gold[question]),
            "collaborative": f1(caliper[question], gold[question]),
        })
    result = {}
    for group_index, (name, rows) in enumerate(sorted(groups.items())):
        selected = [row["selected"] for row in rows]
        result[name] = {
            "n": len(rows), "rate": round(len(rows) / len(labels) * 100, 2),
            "selected_F1": round(statistics.mean(selected) * 100, 2),
            "selected_F1_CI95": interval(selected, 42 + domain_index * 10 + group_index),
            "local_F1": round(statistics.mean(row["local"] for row in rows) * 100, 2),
            "collaborative_F1": round(statistics.mean(row["collaborative"] for row in rows) * 100, 2),
        }
    output["datasets"][domain] = {"n": len(labels), "groups": result}

target = ROOT / "classifier_analysis/classifier_downstream_groups.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
