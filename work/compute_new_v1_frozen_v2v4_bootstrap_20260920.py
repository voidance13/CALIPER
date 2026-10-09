#!/usr/bin/env python3
"""Recompute paired V1-vs-frozen-V2/V4 statistics after the MuSiQue V1 rerun."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path("/mnt/data/caliper/code/CALIPER")
TEST = BASE / "data/musique/test.json"
V1 = BASE / "router_e2e_runs/text_grouped_20260920_unified_bgebase/predictions/musique_v1_router_textgrouped_20260918.json"
V2 = [BASE / "router_e2e_runs/v2v4_text_grouped_20260919/predictions/musique_v2_router_textgrouped_20260919.json"]
V4 = BASE / "router_e2e_runs/old_v4_seed42_4gpu_serial_20260919/musique/musique_v4_seed42_4gpu.json"
PUNCT = set(chr(i) for i in range(33, 48)) | set(chr(i) for i in range(58, 65)) | set(chr(i) for i in range(91, 97)) | set(chr(i) for i in range(123, 127))


def canonical(text):
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(c for c in text if c not in PUNCT).split())


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def f1(prediction, gold):
    predicted, expected = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    p = common / len(predicted) if predicted else 0.0
    r = common / len(expected) if expected else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return {canonical(row["question"]): parse_answer(row.get("pred") or row.get("res", "")) for row in rows}


def bootstrap(left, right, seed):
    diff = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(10000, len(diff)))
    means = diff[draws].mean(axis=1)
    below = (np.count_nonzero(means <= 0) + 1) / 10001
    above = (np.count_nonzero(means >= 0) + 1) / 10001
    return {
        "n": len(diff),
        "delta_f1": round(float(diff.mean() * 100), 4),
        "ci95": [round(float(x * 100), 4) for x in np.percentile(means, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
        "resamples": 10000,
        "seed": seed,
    }


def main():
    test = json.loads(TEST.read_text(encoding="utf-8"))
    counts = Counter(canonical(row["question"]) for row in test)
    v1, v2, v4 = load([V1]), load(V2), load([V4])
    keys = [
        canonical(row["question"])
        for row in test
        if counts[canonical(row["question"])] == 1
        and all(mapping.get(canonical(row["question"])) is not None for mapping in (v1, v2, v4))
    ]
    gold = {canonical(row["question"]): (row.get("answer") if isinstance(row.get("answer"), str) else (row.get("answers") or [""])[0]) for row in test}
    vectors = {name: np.asarray([f1(mapping[key], gold[key]) for key in keys]) for name, mapping in (("V1", v1), ("V2", v2), ("V4", v4))}
    out = {
        "protocol": "MuSiQue common 2,403-question intersection; new unified-bge-base V1 versus frozen V2 and fixed-seed V4 predictions",
        "n": len(keys),
        "V1_minus_V2": bootstrap(vectors["V1"], vectors["V2"], 42),
        "V1_minus_V4": bootstrap(vectors["V1"], vectors["V4"], 42),
    }
    target = BASE / "table-result/musique_new_v1_frozen_v2v4_bootstrap_20260920.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
