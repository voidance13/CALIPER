"""Merge four shards, evaluate full CALIPER and fixed-threshold classifier routes."""
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
RUN = BASE / "full_runs/hotpotqa_caliper_B32"
cache = json.load(open(BASE / "data/hotpotqa/test_retrieved_embedderft_rerankerft_v2.json", encoding="utf-8"))
test = json.load(open(BASE / "data/hotpotqa/test.json", encoding="utf-8"))
gold = {x["question"]: (x.get("answer") if isinstance(x.get("answer"), str) else (x.get("answers") or [""])[0]) for x in test}

merged = {}
for shard in range(4):
    for line in open(RUN / f"part{shard}.jsonl", encoding="utf-8"):
        row = json.loads(line)
        merged[row["question"]] = row
assert len(merged) == len(cache) == 7405, (len(merged), len(cache))
caliper = [{"question": row["question"], "pred": merged[row["question"]]["pred"]} for row in cache]
caliper_path = BASE / "data/hotpotqa/hotpotqa_caliper_t2_B32_full.json"
json.dump(caliper, open(caliper_path, "w", encoding="utf-8"), ensure_ascii=False)

stop = {x.strip() for x in open(BASE / "query_clf/stopwords.txt", encoding="utf-8")}
classifier = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
baseline = {x["question"]: x.get("pred", "") for x in json.load(open(BASE / "data/hotpotqa/hotpotqa_baseline_2stage_soft.json", encoding="utf-8"))}


def clean(text):
    return " ".join(x for x in jieba.lcut(text) if len(x) > 1 and x not in stop)


def p_easy(question):
    labels, probabilities = classifier.predict(clean(question), k=2)
    return dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)


classifier_rows, local = [], 0
for row in caliper:
    use_local = p_easy(row["question"]) >= 0.999
    local += use_local
    classifier_rows.append({"question": row["question"], "pred": baseline[row["question"]] if use_local else row["pred"]})
classifier_path = BASE / "data/hotpotqa/hotpotqa_clf_t2_fixed_threshold_full.json"
json.dump(classifier_rows, open(classifier_path, "w", encoding="utf-8"), ensure_ascii=False)


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
    return " ".join("".join(x for x in text if x not in set(string.punctuation)).split())


def scores(rows):
    values, predictions, targets = [], [], []
    for row in rows:
        prediction, target = answer(row.get("pred")), gold[row["question"]]
        p, g = normalize(prediction), normalize(target)
        common = sum((Counter(p.split()) & Counter(g.split())).values())
        precision = common / len(p.split()) if p.split() else 0.0
        recall = common / len(g.split()) if g.split() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append([int(g in p), precision, recall, f1])
        predictions.append(prediction)
        targets.append(target)
    return np.asarray(values), predictions, targets


rows = {
    "baseline": [{"question": x["question"], "pred": baseline[x["question"]]} for x in cache],
    "caliper": caliper,
    "caliper_clf": classifier_rows,
}
model = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu")
metrics, arrays = {}, {}
for name, data in rows.items():
    values, predictions, targets = scores(data)
    arrays[name] = values[:, 3]
    with torch.no_grad():
        p = model.encode(predictions, convert_to_tensor=True, batch_size=256)
        g = model.encode(targets, convert_to_tensor=True, batch_size=256)
        sbert = float(util.pytorch_cos_sim(p, g).diag().mean()) * 100
    means = values.mean(axis=0) * 100
    metrics[name] = dict(zip(("hit", "precision", "recall", "f1"), means.tolist())) | {"sbert": sbert, "n": len(data)}

rng = np.random.default_rng(42)
paired = {}
for left, right in (("caliper", "baseline"), ("caliper_clf", "baseline"), ("caliper", "caliper_clf")):
    difference = arrays[left] - arrays[right]
    boot = np.empty(10000)
    for index in range(len(boot)):
        boot[index] = difference[rng.integers(0, len(difference), len(difference))].mean() * 100
    observed = difference.mean() * 100
    paired[f"{left}_minus_{right}"] = {"delta_f1": observed, "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                                                   "p_two_sided": float(min(1, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))}

result = {"configuration": "HotpotQA CALIPER B32 full; Qwen3.5-9B; reflection off; threshold 0.999",
          "local": int(local), "local_rate": local / len(cache), "metrics": metrics, "paired_bootstrap": paired}
result_path = RUN / "hotpotqa_caliper_B32_full_metrics.json"
json.dump(result, open(result_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

files = [caliper_path, classifier_path, result_path, *(RUN / f"part{x}.jsonl" for x in range(4))]
with open(RUN / "SHA256SUMS.hotpotqa_caliper_B32_full.txt", "w", encoding="utf-8") as output:
    for path in files:
        output.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(BASE)}\n")
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
assert all(value["n"] == 7405 for value in metrics.values())

