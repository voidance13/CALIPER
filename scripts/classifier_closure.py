"""Audit 9b_clf, choose one abstention threshold, and rebuild Table-2 routes."""
import json
import re
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba

BASE = Path("/mnt/data/caliper/code/CALIPER")
CLF = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
STOP = {x.strip() for x in open(BASE / "query_clf/stopwords.txt", encoding="utf-8")}
DOMAINS = ("hotpotqa", "musique", "2wikimultihopqa")
GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        0.97, 0.98, 0.99, 0.995, 0.999, 0.9995]


def clean(text):
    return " ".join(x for x in jieba.lcut(text) if len(x) > 1 and x not in STOP)


def norm(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join("".join(x for x in text if x not in set(string.punctuation)).split())


def answer(text):
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        if marker in text:
            return text.split(marker, 1)[1]
    return text.splitlines()[-1] if text.splitlines() else ""


def token_scores(pred, gold):
    p, g = norm(answer(pred)), norm(gold)
    pc, gc = Counter(p.split()), Counter(g.split())
    common = sum((pc & gc).values())
    if not common:
        return int(bool(g) and g in p), 0.0, 0.0, 0.0
    precision, recall = common / sum(pc.values()), common / sum(gc.values())
    return int(g in p), precision, recall, 2 * precision * recall / (precision + recall)


def probability_easy(question):
    labels, probs = CLF.predict(clean(question), k=2)
    values = {label.removeprefix("__label__"): prob for label, prob in zip(labels, probs)}
    return values.get("easy", 0.0)


def gold_map(domain):
    rows = json.load(open(BASE / f"data/{domain}/test.json", encoding="utf-8"))
    return {x["question"]: (x.get("answer") if isinstance(x.get("answer"), str)
                              else (x.get("answers") or [""])[0]) for x in rows}


def behavior_labels(domain):
    runs = []
    for seed in range(1, 4):
        rows = json.load(open(BASE / f"query_clf/{domain}_base_sample{seed}.json", encoding="utf-8"))
        runs.append(dict(rows))
    gold = gold_map(domain)
    out = {}
    for question, target in list(gold.items())[:2000]:
        if all(question in run for run in runs):
            out[question] = "easy" if all(norm(target) in norm(run[question]) for run in runs) else "hard"
    return out


def confusion(rows, threshold):
    counts = Counter()
    for row in rows:
        routed = "easy" if row["p_easy"] >= threshold else "hard"
        counts[(row["label"], routed)] += 1
    easy, hard = sum(r["label"] == "easy" for r in rows), sum(r["label"] == "hard" for r in rows)
    correct = counts[("easy", "easy")] + counts[("hard", "hard")]
    return {
        "n": len(rows), "threshold": threshold, "accuracy": correct / len(rows),
        "false_negative_rate": counts[("hard", "easy")] / hard,
        "false_positive_rate": counts[("easy", "hard")] / easy,
        "collaborative_rate": (counts[("easy", "hard")] + counts[("hard", "hard")]) / len(rows),
        "matrix": {"easy_easy": counts[("easy", "easy")], "easy_hard": counts[("easy", "hard")],
                   "hard_easy": counts[("hard", "easy")], "hard_hard": counts[("hard", "hard")]},
    }


def mean_metrics(rows):
    totals = [0.0] * 4
    for row in rows:
        scores = token_scores(row["pred"], row["gold"])
        totals = [x + y for x, y in zip(totals, scores)]
    n = len(rows)
    return {k: v / n * 100 for k, v in zip(("hit", "precision", "recall", "f1"), totals)} | {"n": n}


labels = {domain: behavior_labels(domain) for domain in DOMAINS}
valid_lines = set(open(BASE / "query_clf/train_new/9b_valid.txt", encoding="utf-8").read().splitlines())
audit = {}
for domain in DOMAINS:
    rows = []
    for question, label in labels[domain].items():
        line = f"__label__{label} {clean(question)}"
        if domain == "2wikimultihopqa" or line in valid_lines:
            rows.append({"question": question, "label": label, "p_easy": probability_easy(question)})
    audit[domain] = rows

validation = audit["hotpotqa"] + audit["musique"]
curve = [confusion(validation, threshold) for threshold in GRID]
threshold = next((x["threshold"] for x in curve if x["false_negative_rate"] <= 0.10), GRID[-1])

paths = {
    "hotpotqa": ("data/hotpotqa/hotpotqa_baseline_2stage_soft.json", "/tmp/hp_t2cal_B32_200.json"),
    "musique": ("data/musique/musique_baseline_2stage_soft.json", "data/musique/musique_v1_ubh_2stream.json"),
    "2wikimultihopqa": ("data/2wikimultihopqa/2wikimultihopqa_baseline_2stage_soft.json",
                         "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json"),
}
routing = {}
for domain, (base_path, caliper_path) in paths.items():
    load = lambda path: json.load(open(Path(path) if path.startswith("/") else BASE / path, encoding="utf-8"))
    baseline = {x["question"]: x.get("pred", "") for x in load(base_path)}
    caliper = load(caliper_path)
    gold = gold_map(domain)
    routed, local = [], 0
    for row in caliper:
        question = row["question"]
        use_local = probability_easy(question) >= threshold
        pred = baseline.get(question, row.get("pred", "")) if use_local else row.get("pred", "")
        local += use_local
        routed.append({"question": question, "pred": pred, "gold": gold[question], "route": "local" if use_local else "collaborative"})
    destination = BASE / f"data/{domain}/{domain}_clf_t2_fixed_threshold.json"
    json.dump([{"question": x["question"], "pred": x["pred"]} for x in routed], open(destination, "w", encoding="utf-8"), ensure_ascii=False)
    routing[domain] = {"threshold": threshold, "local": local, "collaborative": len(routed) - local,
                       "local_rate": local / len(routed), "metrics": mean_metrics(routed), "file": str(destination)}

result = {
    "classifier": "query_clf/train_new/9b_clf.bin",
    "label_definition": "easy iff all three Qwen3.5-9B baseline runs contain gold answer; otherwise hard",
    "threshold_selection": "smallest fixed grid threshold with validation hard-to-local false-negative rate <= 10%",
    "selected_threshold": threshold,
    "validation_curve": curve,
    "table5": {domain: confusion(rows, 0.5) for domain, rows in audit.items()},
    "selected_threshold_audit": {domain: confusion(rows, threshold) for domain, rows in audit.items()},
    "routing": routing,
}
output = BASE / "classifier_analysis/classifier_closure.json"
output.parent.mkdir(exist_ok=True)
json.dump(result, open(output, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
          default=lambda value: value.item())
print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item()))
print(f"saved {output}")

assert result["table5"]["hotpotqa"]["n"] > 0 and result["routing"]["2wikimultihopqa"]["metrics"]["n"] > 0
