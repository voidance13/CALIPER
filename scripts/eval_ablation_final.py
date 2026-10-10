"""消融 V0-V4 五列批量评估(统一 pa/norm + SBERT mpnet)"""
import json, re, glob, os, sys
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import torch

BASE = os.getenv("CALIPER_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUNCT = set(chr(i) for i in range(33, 48)) | set(chr(i) for i in range(58, 65)) | set(chr(i) for i in range(91, 97)) | set(chr(i) for i in range(123, 127))

def norm(s):
    t = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    t = "".join(ch for ch in t if ch not in PUNCT)
    return " ".join(t.split())

def pa(r):
    for p in ["Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"]:
        m = re.search(re.escape(p), r or "")
        if m:
            return r[m.end():]
    line = (r or "").split("\n")[-1]
    m = re.search(": ", line)
    return line[m.end():] if m else line

def f1p(pn, gn):
    pc, gc = Counter(pn.split()), Counter(gn.split())
    c = sum((pc & gc).values())
    if not c:
        return 0.0, 0.0, 0.0
    pr, rc = c / len(pn.split()), c / len(gn.split())
    return 2 * pr * rc / (pr + rc), pr, rc

def load_preds(patterns):
    """按 glob 模式读输出,按 question 去重"""
    d = {}
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if f.endswith(".tmp"):
                continue
            for s in json.load(open(f, encoding="utf-8")):
                q = s["question"]
                if q not in d:
                    d[q] = s.get("pred") or s.get("res", "")
    return d

# (变体, 数据集, 文件模式列表)
CONFIGS = [
    ("V1", "hotpotqa", [f"{BASE}/data/hotpotqa/hotpotqa_caliper_planA_top5.json"]),
    ("V1", "musique", [f"{BASE}/data/musique/musique_v1_ubh_2stream.json"]),
    ("V1", "2wikimultihopqa", [f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json"]),
    ("V2", "hotpotqa", [f"{BASE}/data/hotpotqa/hotpotqa_v2_cal_base_reflect.json.s0", f"{BASE}/data/hotpotqa/hotpotqa_v2_cal_base_reflect.json.s1"]),
    ("V2", "musique", [f"{BASE}/data/musique/musique_v2_cal_base_reflect.json.s0", f"{BASE}/data/musique/musique_v2_cal_base_reflect.json.s1"]),
    ("V2", "2wikimultihopqa", [f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v2_cal_base_reflect.json.s0", f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v2_cal_base_reflect.json.s1"]),
    ("V3", "musique", [f"{BASE}/data/musique/musique_v3_baseline_2stream.json"]),
    ("V4", "hotpotqa", [f"{BASE}/data/hotpotqa/hotpotqa_v4_baseline_base.json"]),
    ("V4", "musique", [f"{BASE}/data/musique/musique_v4_baseline_base.json"]),
    ("V4", "2wikimultihopqa", [f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v4_baseline_base.json"]),
    ("V0", "hotpotqa", [f"{BASE}/data/hotpotqa/hotpotqa_caliper_planA_top5_reflA_full.json"]),
    ("V0", "2wikimultihopqa", [f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v0_opt_reflect.json.s0", f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v0_opt_reflect.json.s1"]),
    ("V0", "musique", [f"{BASE}/data/musique/musique_v0_2stream_steps_reflect.json"]),
    ("V3", "hotpotqa", [f"{BASE}/data/hotpotqa/hotpotqa_v3_baseline_ft.json.s*"]),
    ("V3", "2wikimultihopqa", [f"{BASE}/data/2wikimultihopqa/2wikimultihopqa_v3_baseline_ft.json.s*"]),
]

model = SentenceTransformer(os.getenv("CALIPER_METRIC_MODEL", os.path.join(BASE, "models/paraphrase-mpnet-base-v2")), device="cpu")

results = {}
for variant, ds, pats in CONFIGS:
    key = f"{variant}-{ds}"
    if key in results:
        continue
    preds = load_preds(pats)
    test = json.load(open(f"{BASE}/data/{ds}/test.json", encoding="utf-8"))
    q2a = {t["question"]: (t.get("answer") if isinstance(t.get("answer"), str) else (t.get("answers") or [""])[0]) for t in test}
    hits = ps = rs = f1s = 0.0
    pp, gp = [], []
    n = 0
    for q, p in preds.items():
        g = q2a.get(q, "")
        if not g:
            continue
        pn, gn = norm(pa(p)), norm(g)
        hits += int(gn in pn)
        f1, pr, rc = f1p(pn, gn)
        ps += pr; rs += rc; f1s += f1
        pp.append(pa(p)); gp.append(g)
        n += 1
    if n:
        with torch.no_grad():
            e1 = model.encode(pp, convert_to_tensor=True, batch_size=256)
            e2 = model.encode(gp, convert_to_tensor=True, batch_size=256)
            sbert = float(util.pytorch_cos_sim(e1, e2).diag().mean()) * 100
        results[key] = {"SBERT": round(sbert, 2), "Hit": round(hits / n * 100, 2),
                        "P": round(ps / n * 100, 2), "R": round(rs / n * 100, 2),
                        "F1": round(f1s / n * 100, 2), "n": n,
                        "sources": [p.replace(f"{BASE}/", "") for p in pats]}
        print("%s: %s" % (key, results[key]), flush=True)

assert len(results) == 15 and all(v["n"] in (7405, 2412, 12576) for v in results.values())
out = f"{BASE}/table-result/ablation_final_metrics.json"
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"SAVED {out}")
