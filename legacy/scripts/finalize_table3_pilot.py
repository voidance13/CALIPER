"""Re-evaluate every Table 3 row on the exact same 200 questions; mark as pilot."""
import json, re, string
from collections import Counter
from pathlib import Path

import fasttext, jieba, torch
from sentence_transformers import SentenceTransformer, util

BASE = Path("/mnt/data/caliper/code/CALIPER")
TAU = 0.997
FILES = {
    "hotpotqa": ("/tmp/hp_t3cal_A_m512b32_200.json", "data/hotpotqa/hotpotqa_baseline_t3_b.json", "data/hotpotqa/normal_pred_72b_sep_use_best_hit.json"),
    "musique": ("/tmp/musique_t3cal_m512_200.json", "data/musique/musique_baseline_t3_soft.json", "data/musique/normal_pred_72b_sep_use_best_hit_merge.json"),
    "2wikimultihopqa": ("/tmp/2wikimultihopqa_t3cal_B32_200.json", "data/2wikimultihopqa/2wikimultihopqa_baseline_t3_b.json", "data/2wikimultihopqa/normal_pred_72b_sep_use_best_hit_merge.json"),
}
clf = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
stopwords = {x.strip() for x in open(BASE / "query_clf/stopwords.txt", encoding="utf-8")}
sbert = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu")

def answer(text):
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text)
        if match: return text[match.end():]
    line = text.splitlines()[-1] if text.splitlines() else ""
    match = re.search(": ", line)
    return line[match.end():] if match else line

def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    return " ".join("".join(x for x in text if x not in set(string.punctuation)).split())

def evaluate(rows, gold):
    values, predictions, targets = [], [], []
    for row in rows:
        prediction, target = answer(row["pred"]), gold[row["question"]]
        p, g = Counter(normalize(prediction).split()), Counter(normalize(target).split())
        common = sum((p & g).values()); precision = common / sum(p.values()) if p else 0; recall = common / sum(g.values()) if g else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        values.append((int(bool(g) and normalize(target) in normalize(prediction)), precision, recall, f1)); predictions.append(prediction); targets.append(target)
    with torch.no_grad():
        pe = sbert.encode(predictions, convert_to_tensor=True, batch_size=200)
        ge = sbert.encode(targets, convert_to_tensor=True, batch_size=200)
        semantic = float(util.pytorch_cos_sim(pe, ge).diag().mean()) * 100
    means = [sum(x[i] for x in values) / len(values) * 100 for i in range(4)]
    return dict(zip(("SBERT", "Hit", "P", "R", "F1"), [semantic] + means)) | {"n": len(rows)}

def local_route(question):
    cleaned = " ".join(x for x in jieba.lcut(question) if len(x) > 1 and x not in stopwords)
    labels, probabilities = clf.predict(cleaned, k=2)
    return dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0) >= TAU

result = {"status": "pilot; optimized on this sample, not confirmatory", "seed": 42, "threshold": TAU, "datasets": {}}
for ds, (caliper_file, baseline_file, cloud_file) in FILES.items():
    caliper = list({x["question"]: x for x in json.load(open(caliper_file, encoding="utf-8"))}.values())
    questions = [x["question"] for x in caliper]
    assert len(questions) in (199, 200)
    gold = {x["question"]: (x.get("answer") if isinstance(x.get("answer"), str) else (x.get("answers") or [""])[0]) for x in json.load(open(BASE / f"data/{ds}/test.json", encoding="utf-8"))}
    baseline_all = {x["question"]: x.get("pred", "") for x in json.load(open(BASE / baseline_file, encoding="utf-8"))}
    cloud_all = {x["question"]: x.get("pred") or x.get("res", "") for x in json.load(open(BASE / cloud_file, encoding="utf-8"))}
    baseline = [{"question": q, "pred": baseline_all[q]} for q in questions]
    cloud = [{"question": q, "pred": cloud_all[q]} for q in questions]
    routed = [{"question": x["question"], "pred": baseline_all[x["question"]] if local_route(x["question"]) else x["pred"]} for x in caliper]
    result["datasets"][ds] = {"baseline": evaluate(baseline, gold), "caliper_clf": evaluate(routed, gold), "caliper": evaluate(caliper, gold), "cloud": evaluate(cloud, gold),
                                      "sample_questions": questions, "sources": {"caliper": caliper_file, "baseline": baseline_file, "cloud": cloud_file}}
    print(ds, json.dumps(result["datasets"][ds], ensure_ascii=False)[:1000], flush=True)

json.dump(result, open(BASE / "table-result/table3_pilot_seed42_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
