"""Add SBERT scores for fixed-threshold classifier routes."""
import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

BASE = Path("/mnt/data/caliper/code/CALIPER")
MODEL = SentenceTransformer("/mnt/data/caliper/models/paraphrase-mpnet-base-v2", device="cpu")


def answer(text):
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text)
        if match:
            return text[match.end():]
    line = text.splitlines()[-1] if text.splitlines() else ""
    match = re.search(": ", line)
    return line[match.end():] if match else line


scores = {}
for domain in ("hotpotqa", "musique", "2wikimultihopqa"):
    test = json.load(open(BASE / f"data/{domain}/test.json", encoding="utf-8"))
    gold = {x["question"]: (x.get("answer") if isinstance(x.get("answer"), str)
                             else (x.get("answers") or [""])[0]) for x in test}
    rows = json.load(open(BASE / f"data/{domain}/{domain}_clf_t2_fixed_threshold.json", encoding="utf-8"))
    predictions = [answer(x.get("pred")) for x in rows if x["question"] in gold]
    targets = [gold[x["question"]] for x in rows if x["question"] in gold]
    with torch.no_grad():
        p = MODEL.encode(predictions, convert_to_tensor=True, batch_size=256)
        g = MODEL.encode(targets, convert_to_tensor=True, batch_size=256)
        scores[domain] = round(float(util.pytorch_cos_sim(p, g).diag().mean()) * 100, 2)
    print(domain, scores[domain], flush=True)

path = BASE / "classifier_analysis/classifier_closure.json"
result = json.load(open(path, encoding="utf-8"))
for domain, score in scores.items():
    result["routing"][domain]["metrics"]["sbert"] = score
json.dump(result, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
assert len(scores) == 3
