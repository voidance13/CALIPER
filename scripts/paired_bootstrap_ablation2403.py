"""Paired bootstrap for MuSiQue V1--V4 on the canonical 2,403 set."""
import glob, json, re, string
from collections import Counter
from pathlib import Path
import numpy as np

BASE = Path("/mnt/data/caliper/code/CALIPER")
SOURCES = {
    "V0": ["data/musique/musique_v0_2stream_steps_reflect.json"],
    "V1": ["data/musique/musique_v1_ubh_2stream.json"],
    # Keep the same two V2 shards used by the archived ablation result.
    "V2": ["data/musique/musique_v2_cal_base_reflect.json.s0", "data/musique/musique_v2_cal_base_reflect.json.s1"],
    "V3": ["data/musique/musique_v3_baseline_2stream.json"],
    "V4": ["data/musique/musique_v4_baseline_base.json"],
}
PUNCT = set(string.punctuation)

def canonical(text):
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())

def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(c for c in text if c not in PUNCT).split())

def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        m = re.search(re.escape(marker), text or "")
        if m:
            return text[m.end():]
    line = (text or "").split("\n")[-1]
    m = re.search(": ", line)
    return line[m.end():] if m else line

def f1(prediction, gold):
    pt, gt = normalize(parse_answer(prediction)).split(), normalize(gold).split()
    common = sum((Counter(pt) & Counter(gt)).values())
    p = common / len(pt) if pt else 0.0
    r = common / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def load(spec):
    rows = []
    for pattern in spec:
        for path in sorted(glob.glob(str(BASE / pattern))):
            rows.extend(json.load(open(path, encoding="utf-8")))
    return rows

def paired(left, right, seed, samples=10000):
    d = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        idx = rng.integers(0, len(d), size=(size, len(d)))
        boot[start:start+size] = d[idx].mean(axis=1)
    below = (np.count_nonzero(boot <= 0) + 1) / (samples + 1)
    above = (np.count_nonzero(boot >= 0) + 1) / (samples + 1)
    return {
        "n": len(d), "left_F1": round(float(left.mean()*100), 2),
        "right_F1": round(float(right.mean()*100), 2),
        "delta_F1": round(float(d.mean()*100), 2),
        "delta_CI95": [round(float(x*100), 2) for x in np.percentile(boot, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2*min(below, above)), 6),
        "bootstrap_samples": samples, "seed": seed,
    }

def main():
    test = json.load((BASE / "data/musique/test.json").open(encoding="utf-8"))
    test_counts = Counter(canonical(row["question"]) for row in test)
    loaded = {v: load(spec) for v, spec in SOURCES.items()}
    counts = {v: Counter(canonical(row["question"]) for row in rows) for v, rows in loaded.items()}
    keys = [canonical(row["question"]) for row in test if test_counts[canonical(row["question"])] == 1 and all(counts[v][canonical(row["question"])] == 1 for v in SOURCES)]
    assert len(keys) == 2403, len(keys)
    gold = {canonical(row["question"]): (row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0]) for row in test}
    arrays = {}
    for variant, rows in loaded.items():
        pred = {canonical(row["question"]): row.get("pred") or row.get("res", "") for row in rows if counts[variant][canonical(row.get("question", ""))] == 1}
        arrays[variant] = np.asarray([f1(pred[k], gold[k]) for k in keys])
    out = {"selection": "canonical question unique in test and exactly once in every V0--V4 prediction file", "paired_n": len(keys), "V0_vs_V1": paired(arrays["V0"], arrays["V1"], 44), "V1_vs_V2": paired(arrays["V1"], arrays["V2"], 45), "V1_vs_V3": paired(arrays["V1"], arrays["V3"], 46), "V1_vs_V4": paired(arrays["V1"], arrays["V4"], 47)}
    target = BASE / "significance_analysis/ablation_musique_paired2403_bootstrap.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print("SAVED", target, flush=True)

if __name__ == "__main__":
    main()
