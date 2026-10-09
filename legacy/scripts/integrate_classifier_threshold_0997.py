"""Freeze tau=.997, rebuild classifier routes, evaluate, and hash outputs."""
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

BASE = Path("/mnt/data/caliper/code/CALIPER")
THRESHOLD = 0.997
DOMAINS = {
    "hotpotqa": "data/hotpotqa/hotpotqa_caliper_planA_top5.json",
    "musique": "data/musique/musique_v1_ubh_2stream.json",
    "2wikimultihopqa": "data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json",
}
classifier = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
stopwords = {x.strip() for x in open(BASE / "query_clf/stopwords.txt", encoding="utf-8")}
punctuation = set(string.punctuation)


def clean(text):
    return " ".join(x for x in jieba.lcut(text) if len(x) > 1 and x not in stopwords)


def probability_easy(text):
    labels, probabilities = classifier.predict(clean(text), k=2)
    return dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)


def answer(text):
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text)
        if match:
            return text[match.end():]
    line = text.splitlines()[-1] if text.splitlines() else ""
    match = re.search(": ", line)
    return line[match.end():] if match else line


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join("".join(x for x in text if x not in punctuation).split())


def token_metrics(rows, gold):
    values, predictions, targets = [], [], []
    for row in rows:
        prediction, target = answer(row.get("pred")), gold[row["question"]]
        p, g = normalize(prediction), normalize(target)
        pc, gc = Counter(p.split()), Counter(g.split())
        common = sum((pc & gc).values())
        precision = common / sum(pc.values()) if pc else 0.0
        recall = common / sum(gc.values()) if gc else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((int(bool(g) and g in p), precision, recall, f1))
        predictions.append(prediction)
        targets.append(target)
    return np.asarray(values), predictions, targets


def paired(left, right, seed):
    difference = left - right
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10000)
    for index in range(len(bootstrap)):
        bootstrap[index] = difference[rng.integers(0, len(difference), len(difference))].mean() * 100
    return {
        "delta_f1": float(difference.mean() * 100),
        "ci95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "p_two_sided": float(min(1, 2 * min((bootstrap <= 0).mean(), (bootstrap >= 0).mean()))),
        "resamples": 10000,
        "seed": seed,
    }


sentence_model = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu")
results, output_paths = {}, []
hotpot_arrays = {}
for domain, caliper_relative in DOMAINS.items():
    test = json.load(open(BASE / f"data/{domain}/test.json", encoding="utf-8"))
    gold = {x["question"]: (x.get("answer") if isinstance(x.get("answer"), str)
                             else (x.get("answers") or [""])[0]) for x in test}
    baseline = {x["question"]: x.get("pred", "") for x in json.load(
        open(BASE / f"data/{domain}/{domain}_baseline_2stage_soft.json", encoding="utf-8"))}
    caliper_rows = json.load(open(BASE / caliper_relative, encoding="utf-8"))
    routed, local = [], 0
    for row in caliper_rows:
        use_local = probability_easy(row["question"]) >= THRESHOLD
        local += use_local
        routed.append({"question": row["question"], "pred": baseline[row["question"]] if use_local else row.get("pred", "")})
    output_path = BASE / f"data/{domain}/{domain}_clf_t2_th0997.json"
    json.dump(routed, open(output_path, "w", encoding="utf-8"), ensure_ascii=False)
    output_paths.append(output_path)

    values, predictions, targets = token_metrics(routed, gold)
    with torch.no_grad():
        p = sentence_model.encode(predictions, convert_to_tensor=True, batch_size=256)
        g = sentence_model.encode(targets, convert_to_tensor=True, batch_size=256)
        sbert = float(util.pytorch_cos_sim(p, g).diag().mean()) * 100
    means = values.mean(axis=0) * 100
    results[domain] = {
        "n": len(routed), "threshold": THRESHOLD, "local": int(local), "local_rate": local / len(routed),
        "collaborative_rate": 1 - local / len(routed), "metrics": {
            "sbert": sbert, "hit": float(means[0]), "precision": float(means[1]),
            "recall": float(means[2]), "f1": float(means[3]),
        }, "caliper_prediction_file": caliper_relative,
        "prediction_file": str(output_path.relative_to(BASE)),
    }
    if domain == "hotpotqa":
        baseline_rows = [{"question": row["question"], "pred": baseline[row["question"]]} for row in caliper_rows]
        caliper_values, caliper_predictions, caliper_targets = token_metrics(caliper_rows, gold)
        with torch.no_grad():
            cp = sentence_model.encode(caliper_predictions, convert_to_tensor=True, batch_size=256)
            cg = sentence_model.encode(caliper_targets, convert_to_tensor=True, batch_size=256)
            caliper_sbert = float(util.pytorch_cos_sim(cp, cg).diag().mean()) * 100
        caliper_means = caliper_values.mean(axis=0) * 100
        results[domain]["caliper_metrics"] = {
            "sbert": caliper_sbert, "hit": float(caliper_means[0]),
            "precision": float(caliper_means[1]), "recall": float(caliper_means[2]),
            "f1": float(caliper_means[3]),
        }
        hotpot_arrays = {
            "baseline": token_metrics(baseline_rows, gold)[0][:, 3],
            "caliper": caliper_values[:, 3],
            "caliper_clf": values[:, 3],
        }
    print(domain, json.dumps(results[domain], ensure_ascii=False), flush=True)

validation = []
for line in open(BASE / "query_clf/train_new/9b_valid.txt", encoding="utf-8"):
    label, text = line.rstrip().split(" ", 1)
    labels, probabilities = classifier.predict(text, k=2)
    p_easy = dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)
    validation.append((label.removeprefix("__label__"), p_easy))
matrix = Counter((label, "easy" if probability >= THRESHOLD else "hard") for label, probability in validation)
hard = sum(label == "hard" for label, _ in validation)
easy = len(validation) - hard
validation_result = {
    "n": len(validation), "threshold": THRESHOLD,
    "false_negative_rate": matrix[("hard", "easy")] / hard,
    "false_positive_rate": matrix[("easy", "hard")] / easy,
    "collaborative_rate": sum(probability < THRESHOLD for _, probability in validation) / len(validation),
    "matrix": {"easy_easy": matrix[("easy", "easy")], "easy_hard": matrix[("easy", "hard")],
               "hard_easy": matrix[("hard", "easy")], "hard_hard": matrix[("hard", "hard")]},
}
significance = {
    "caliper_minus_baseline": paired(hotpot_arrays["caliper"], hotpot_arrays["baseline"], 42),
    "caliper_clf_minus_baseline": paired(hotpot_arrays["caliper_clf"], hotpot_arrays["baseline"], 43),
    "caliper_minus_caliper_clf": paired(hotpot_arrays["caliper"], hotpot_arrays["caliper_clf"], 44),
}
final = {
    "threshold_selection": "smallest rounded 0.001 threshold with validation hard-to-local false-negative rate <= 10%",
    "validation": validation_result, "datasets": results, "hotpotqa_paired_bootstrap": significance,
}
result_path = BASE / "classifier_analysis/classifier_threshold_0997_results.json"
json.dump(final, open(result_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
output_paths.append(result_path)
manifest = BASE / "classifier_analysis/SHA256SUMS.classifier_threshold_0997.txt"
with open(manifest, "w", encoding="utf-8") as output:
    for path in output_paths:
        output.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(BASE)}\n")
print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
assert results["hotpotqa"]["n"] == 7405 and validation_result["false_negative_rate"] <= 0.10
