"""Paired F1 bootstrap for Table 3 and ablation variants."""
import glob
import json
import re
import string
from collections import Counter
from pathlib import Path

import fasttext
import jieba
import numpy as np


BASE = Path("/mnt/data/caliper/code/CALIPER")
DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")
PUNCTUATION = set(string.punctuation)
TABLE3 = {
    "Qwen3.5-9B": {
        "hotpotqa": "data/hotpotqa/normal_pred_7b_sep_use_best_hit.json",
        "musique": "data/musique/normal_pred_7b_sep_use_best_hit_single_full.json",
        "2wikimultihopqa": "data/2wikimultihopqa/normal_pred_7b_sep_use_best_hit.json",
    },
    "CALIPER": {
        "hotpotqa": "data/hotpotqa/hybrid_pred_7b_72b_sep_use_best_hit_instr_extract_full.json",
        "musique": "data/musique/hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
        "2wikimultihopqa": "data/2wikimultihopqa/hybrid_pred_7b_72b_sep_use_best_hit_merge_full.json",
    },
    "Qwen3.7-plus": {
        "hotpotqa": "data/hotpotqa/normal_pred_72b_sep_use_best_hit.json",
        "musique": "data/musique/normal_pred_72b_sep_use_best_hit_merge.json",
        "2wikimultihopqa": "data/2wikimultihopqa/normal_pred_72b_sep_use_best_hit_merge.json",
    },
}
ABLATION = {
    "V1": {
        "hotpotqa": ["data/hotpotqa/hotpotqa_caliper_planA_top5.json"],
        "musique": ["data/musique/musique_v1_ubh_2stream.json"],
        "2wikimultihopqa": ["data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json"],
    },
    "V2": {
        "hotpotqa": ["data/hotpotqa/hotpotqa_v2_cal_base_reflect.json.s*"],
        "musique": ["data/musique/musique_v2_cal_base_reflect.json.s*"],
        "2wikimultihopqa": ["data/2wikimultihopqa/2wikimultihopqa_v2_cal_base_reflect.json.s*"],
    },
    "V3": {
        "hotpotqa": ["data/hotpotqa/hotpotqa_v3_baseline_ft.json.s*"],
        "musique": ["data/musique/musique_v3_baseline_2stream.json"],
        "2wikimultihopqa": ["data/2wikimultihopqa/2wikimultihopqa_v3_baseline_ft.json.s*"],
    },
    "V4": {
        "hotpotqa": ["data/hotpotqa/hotpotqa_v4_baseline_base.json"],
        "musique": ["data/musique/musique_v4_baseline_base.json"],
        "2wikimultihopqa": ["data/2wikimultihopqa/2wikimultihopqa_v4_baseline_base.json"],
    },
}


def parse_answer(text):
    for marker in ("Answer:\n", "answer:\n", "Answer: ", "answer: ", "Answer:", "answer:"):
        match = re.search(re.escape(marker), text or "")
        if match:
            return text[match.end():]
    line = (text or "").split("\n")[-1]
    match = re.search(": ", line)
    return line[match.end():] if match else line


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in PUNCTUATION).split())


def f1(prediction, gold):
    predicted, expected = normalize(parse_answer(prediction)).split(), normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0.0
    recall = common / len(expected) if expected else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load(patterns):
    rows = {}
    for pattern in patterns:
        for path in sorted(glob.glob(str(BASE / pattern))):
            for row in json.load(open(path, encoding="utf-8")):
                rows.setdefault(row["question"], row.get("pred") or row.get("res", ""))
    return rows


def paired(left, right, seed=42, samples=10000):
    difference = np.asarray(left) - np.asarray(right)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples)
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        indices = generator.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start:start + size] = difference[indices].mean(axis=1)
    below = (np.count_nonzero(bootstrap <= 0) + 1) / (samples + 1)
    above = (np.count_nonzero(bootstrap >= 0) + 1) / (samples + 1)
    return {
        "n": len(difference), "left_F1": round(float(np.mean(left) * 100), 2),
        "right_F1": round(float(np.mean(right) * 100), 2),
        "delta_F1": round(float(difference.mean() * 100), 2),
        "delta_CI95": [round(float(x * 100), 2) for x in np.percentile(bootstrap, (2.5, 97.5))],
        "p_two_sided": round(min(1.0, 2 * min(below, above)), 6),
        "bootstrap_samples": samples, "seed": seed,
    }


def main():
    classifier = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
    stopwords = {line.strip() for line in (BASE / "query_clf/stopwords.txt").open(encoding="utf-8")}
    table3_output, ablation_output = {}, {}
    for dataset in DATASETS:
        test = json.load((BASE / f"data/{dataset}/test.json").open(encoding="utf-8"))
        gold = {row["question"]: (row.get("answer") if isinstance(row.get("answer"), str)
                                   else (row.get("answers") or [""])[0]) for row in test}
        methods = {method: load([sources[dataset]]) for method, sources in TABLE3.items()}
        questions = list(dict.fromkeys(row["question"] for row in test))
        questions = [q for q in questions if all(q in rows for rows in methods.values())]
        scores = {method: np.asarray([f1(rows[q], gold[q]) for q in questions]) for method, rows in methods.items()}
        routed = []
        for question in questions:
            cleaned = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
            labels, probabilities = classifier.predict(cleaned, k=2)
            easy = dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)
            routed.append(methods["Qwen3.5-9B"][question] if easy >= 0.997 else methods["CALIPER"][question])
        scores["CALIPER+clf"] = np.asarray([f1(prediction, gold[q]) for prediction, q in zip(routed, questions)])
        table3_output[dataset] = {
            "CALIPER_vs_Qwen3.5-9B": paired(scores["CALIPER"], scores["Qwen3.5-9B"], 42),
            "CALIPER_vs_CALIPER+clf": paired(scores["CALIPER"], scores["CALIPER+clf"], 43),
            "CALIPER_vs_Qwen3.7-plus": paired(scores["CALIPER"], scores["Qwen3.7-plus"], 44),
        }

        variants = {variant: load(patterns[dataset]) for variant, patterns in ABLATION.items()}
        paired_questions = [q for q in questions if all(q in rows for rows in variants.values())]
        variant_scores = {variant: np.asarray([f1(rows[q], gold[q]) for q in paired_questions])
                          for variant, rows in variants.items()}
        ablation_output[dataset] = {
            f"V1_vs_{variant}": paired(variant_scores["V1"], variant_scores[variant], 45 + index)
            for index, variant in enumerate(("V2", "V3", "V4"))
        }
        print(dataset, json.dumps({"table3": table3_output[dataset], "ablation": ablation_output[dataset]}, ensure_ascii=False), flush=True)

    significance = BASE / "significance_analysis"
    (significance / "table3_paired_bootstrap.json").write_text(
        json.dumps(table3_output, ensure_ascii=False, indent=2), encoding="utf-8")
    (significance / "ablation_paired_bootstrap.json").write_text(
        json.dumps(ablation_output, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(table3_output[d]["CALIPER_vs_Qwen3.5-9B"]["n"] in (7405, 2412, 12576) for d in DATASETS)
    print("SAVED paired significance")


if __name__ == "__main__":
    main()
