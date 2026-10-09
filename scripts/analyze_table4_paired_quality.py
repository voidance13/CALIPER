"""Same-question F1 comparison for Table 4 latency samples."""
import json
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/data/caliper/code/CALIPER")
DOMAINS = ("hotpotqa", "musique", "2wikimultihopqa")
CALIPER = {
    "hotpotqa": "data/hotpotqa/hotpotqa_caliper_planA_top5.json",
    "musique": "data/musique/musique_v1_ubh_2stream.json",
    "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json",
}
punctuation = set(string.punctuation)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


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


def comparison(left, right, seed):
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    positive = sum(value >= 0 for value in boot)
    return {
        "n": len(differences), "delta_F1": round(statistics.mean(differences) * 100, 2),
        "CI95": [round(boot[249] * 100, 2), round(boot[9749] * 100, 2)],
        "p_two_sided": round(2 * min(positive, len(boot) - positive) / len(boot), 4),
    }


output = {"definition": "same seed-42 Table 4 questions; quality predictions joined from frozen full prediction files", "datasets": {}}
for domain_index, domain in enumerate(DOMAINS):
    tests = load(f"data/{domain}/test.json")
    gold = {row["question"]: row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0] for row in tests}
    baseline = {row["question"]: row.get("pred", "") for row in load(f"data/{domain}/{domain}_baseline_2stage_soft.json")}
    caliper = {row["question"]: row.get("pred", "") for row in load(CALIPER[domain])}
    clf = {row["question"]: row.get("pred", "") for row in load(f"data/{domain}/{domain}_clf_t2_fixed_threshold.json")}
    deep_rows = load(f"table-result/table4_cloud_deep_{domain}_full_b256_seed42.json")["rows"]
    timing = {row["question"]: row for row in load(f"table-result/table4_timing_{domain}_seed42.json")["rows"]}
    rows = []
    for row in deep_rows:
        question = row["question"]
        if row["error"] is not None or question not in caliper or question not in clf or question not in baseline:
            continue
        caliper_timing = timing[question]["caliper"]
        if isinstance(caliper_timing, dict) and caliper_timing["error"] is not None:
            continue
        rows.append({
            "cloud_deep": f1(row["prediction"], gold[question]),
            "caliper": f1(caliper[question], gold[question]),
            "caliper_clf": f1(clf[question], gold[question]),
            "baseline": f1(baseline[question], gold[question]),
        })
    scores = {method: [row[method] for row in rows] for method in rows[0]}
    output["datasets"][domain] = {
        "n": len(rows), "F1": {method: round(statistics.mean(values) * 100, 2) for method, values in scores.items()},
        "caliper_minus_cloud_deep": comparison(scores["caliper"], scores["cloud_deep"], 42 + domain_index),
        "caliper_clf_minus_cloud_deep": comparison(scores["caliper_clf"], scores["cloud_deep"], 52 + domain_index),
    }

target = ROOT / "table-result/table4_paired_quality.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
