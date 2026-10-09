"""Merge, audit, and score the Scheme-A routed end-to-end shards."""
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path("/mnt/data/caliper/code/CALIPER")
RUN = BASE / "router_e2e_runs/text_grouped_20260918"
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
EXPECTED = {"hotpotqa": 7405, "musique": 2403, "2wikimultihopqa": 12576}
PUNCT = set(string.punctuation)


def canonical(s):
    return " ".join("".join(c if c.isalnum() else " " for c in str(s).lower()).split())


def answer(text):
    text = text or ""
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        m = re.search(re.escape(marker), text)
        if m:
            return text[m.end():]
    line = text.splitlines()[-1] if text.splitlines() else ""
    m = re.search(": ", line)
    return line[m.end():] if m else line


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", str(s).lower())
    return " ".join("".join(c for c in s if c not in PUNCT).split())


def vals(pred, gold):
    p, g = norm(answer(pred)).split(), norm(gold).split()
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    precision = common / len(p) if p else 0.0
    recall = common / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return np.asarray([int(bool(g) and " ".join(g) in " ".join(p)), precision, recall, f1], dtype=float)


def load_gold(d):
    rows = json.loads((BASE / f"data/{d}/test.json").read_text(encoding="utf-8"))
    return {canonical(r["question"]): (r.get("answer") if isinstance(r.get("answer"), str) else (r.get("answers") or [""])[0]) for r in rows}


def bootstrap(a, b, seed=42):
    d = a - b
    rng = np.random.default_rng(seed)
    x = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1) * 100
    below = (np.count_nonzero(x <= 0) + 1) / 10001
    above = (np.count_nonzero(x >= 0) + 1) / 10001
    return {"n": len(d), "delta_f1": round(float(d.mean() * 100), 4), "ci95": [round(float(v), 4) for v in np.quantile(x, [0.025, 0.975])], "p_two_sided": round(float(min(1.0, 2 * min(below, above))), 6), "resamples": 10000, "seed": seed}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def merge():
    merged = {v: {} for v in ("v0", "v1", "v3")}
    for p in sorted(RUN.glob("shard_*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            for v in merged:
                q = row[v]["question"]
                if q in merged[v]:
                    raise RuntimeError(f"duplicate {v}: {q}")
                merged[v][q] = row[v]
    for v, rows in merged.items():
        if len(rows) != sum(EXPECTED.values()):
            raise RuntimeError(f"{v}: {len(rows)} rows; expected {sum(EXPECTED.values())}")
    return merged


def old_preds(relative, d):
    return {canonical(r["question"]): r.get("pred") or r.get("res", "") for r in json.loads((BASE / relative.format(d=d)).read_text(encoding="utf-8"))}


def score(ordered, d, gold):
    arr = np.asarray([vals(r["pred"], gold[canonical(r["question"])]) for r in ordered])
    mean = arr.mean(axis=0) * 100
    return {"n": len(arr), "Hit": round(float(mean[0]), 2), "P": round(float(mean[1]), 2), "R": round(float(mean[2]), 2), "F1": round(float(mean[3]), 2)}, arr[:, 3]


def main():
    merged = merge()
    pred_dir = RUN / "predictions"
    pred_dir.mkdir(exist_ok=True)
    by_variant = {}
    metrics = {v: {} for v in ("v0", "v1", "v3", "clf")}
    arrays = {v: {} for v in ("v0", "v1", "v3", "clf")}
    for d in DATASETS:
        gold = load_gold(d)
        for v in ("v0", "v1", "v3"):
            rows = sorted([r for r in merged[v].values() if r["dataset"] == d], key=lambda r: r["row_id"])
            if len(rows) != EXPECTED[d]:
                raise RuntimeError(f"{v}/{d}: {len(rows)}")
            path = pred_dir / f"{d}_{v}_router_textgrouped_20260918.json"
            path.write_text(json.dumps([{"question": r["question"], "pred": r["pred"]} for r in rows], ensure_ascii=False), encoding="utf-8")
            metrics[v][d], arrays[v][d] = score(rows, d, gold)
        # classifier composition uses the frozen operational classifier and baseline answers.
        import fasttext, jieba
        clf = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
        stop = {x.strip() for x in (BASE / "query_clf/stopwords.txt").open(encoding="utf-8")}
        def prob(q):
            clean = " ".join(x for x in jieba.lcut(q) if len(x) > 1 and x not in stop)
            labels, ps = clf.predict(clean, k=2)
            return dict(zip([x.removeprefix("__label__") for x in labels], ps)).get("easy", 0.0)
        base = old_preds("data/{d}/{d}_baseline_2stage_soft.json", d)
        rows = sorted([r for r in merged["v1"].values() if r["dataset"] == d], key=lambda r: r["row_id"])
        crows = [{"question": r["question"], "pred": base[canonical(r["question"])] if prob(r["question"]) >= 0.997 else r["pred"]} for r in rows]
        (pred_dir / f"{d}_clf_router_textgrouped_20260918.json").write_text(json.dumps(crows, ensure_ascii=False), encoding="utf-8")
        metrics["clf"][d], arrays["clf"][d] = score(crows, d, gold)

    comparisons = {"v1_vs_v0": {}, "v1_vs_v3": {}, "v1_vs_baselines": {}}
    for d in DATASETS:
        comparisons["v1_vs_v0"][d] = bootstrap(arrays["v1"][d], arrays["v0"][d])
        comparisons["v1_vs_v3"][d] = bootstrap(arrays["v1"][d], arrays["v3"][d], 43)
    for d in DATASETS:
        gold = load_gold(d)
        for label, rel in {
            "naive": "baseline_runs/naive_rag_{d}_full_seed42_w1.json",
            "adaptive": "adaptive_runs/adaptive_{d}_full_seed42_v3.json",
            "always_multi": "always_multi_runs/always_multi_ircot_{d}_full_seed42.json",
        }.items():
            old = old_preds(rel, d)
            rows = sorted([r for r in merged["v1"].values() if r["dataset"] == d], key=lambda r: r["row_id"])
            # For MuSiQue, the already frozen 2,403 rows define the comparison set.
            a = arrays["v1"][d]
            b = np.asarray([vals(old[canonical(r["question"])], gold[canonical(r["question"])])[3] for r in rows])
            comparisons["v1_vs_baselines"].setdefault(label, {})[d] = bootstrap(a, b, 50)
    payload = {"protocol": "Scheme A; current text-grouped router; local Qwen3.5-9B temp=0 seed=42; cloud steps cached", "expected": EXPECTED, "metrics": metrics, "comparisons": comparisons}
    (RUN / "metrics_and_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = []
    for p in sorted(RUN.rglob("*.jsonl")) + sorted(pred_dir.glob("*.json")) + [RUN / "metrics_and_bootstrap.json"]:
        manifest.append(f"{sha(p)}  {p.relative_to(RUN)}")
    (RUN / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
