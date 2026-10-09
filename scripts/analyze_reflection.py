import json
import string
from collections import Counter
from pathlib import Path

import numpy as np


def canonical(text):
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(text).lower()).split())


def normalize(text):
    text = "".join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def f1(prediction, answer):
    pred, gold = normalize(prediction).split(), normalize(answer).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    return 2 * common / (len(pred) + len(gold)) if common else float(pred == gold)


def load(paths):
    rows = {}
    for path in paths:
        for row in json.load(open(path)):
            rows[canonical(row["question"])] = row.get("pred", "")
    return rows


def answer(row):
    value = row.get("answer", row.get("answers", [""]))
    return value[0] if isinstance(value, list) else value


def compare(dataset, v0_paths, v1_paths):
    gold = {canonical(row["question"]): answer(row) for row in json.load(open(f"data/{dataset}/test.json"))}
    v0, v1 = load(v0_paths), load(v1_paths)
    keys = sorted(gold.keys() & v0.keys() & v1.keys())
    deltas, changed = [], []
    for key in keys:
        delta = f1(v0[key], gold[key]) - f1(v1[key], gold[key])
        deltas.append(delta)
        if normalize(v0[key]) != normalize(v1[key]):
            changed.append(delta)
    generator = np.random.default_rng(42)
    values = np.asarray(deltas)
    bootstrap = np.empty(10000)
    for start in range(0, 10000, 50):
        indices = generator.integers(0, len(values), size=(50, len(values)))
        bootstrap[start:start + 50] = values[indices].mean(axis=1)
    below = (np.count_nonzero(bootstrap <= 0) + 1) / 10001
    above = (np.count_nonzero(bootstrap >= 0) + 1) / 10001
    return {
        "v0_files": [str(path) for path in v0_paths],
        "v1_files": [str(path) for path in v1_paths],
        "v0_unique": len(v0), "v1_unique": len(v1), "paired_n": len(keys),
        "v0_mean_f1": sum(f1(v0[key], gold[key]) for key in keys) / len(keys),
        "v1_mean_f1": sum(f1(v1[key], gold[key]) for key in keys) / len(keys),
        "paired_delta_f1": sum(deltas) / len(deltas),
        "paired_delta_CI95": [float(x) for x in np.percentile(bootstrap, (2.5, 97.5))],
        "p_two_sided": min(1.0, 2 * min(below, above)),
        "bootstrap_samples": 10000,
        "seed": 42,
        "answer_changed_count": len(changed),
        "answer_changed_rate": len(changed) / len(keys),
        "changed_improved": sum(delta > 0 for delta in changed),
        "changed_hurt": sum(delta < 0 for delta in changed),
        "changed_tied": sum(delta == 0 for delta in changed),
        "changed_mean_delta_f1": sum(changed) / len(changed) if changed else 0,
        "iteration_stats": None,
        "iteration_stats_note": "Stored V0 artifacts contain only question/pred; reflection trace and iteration count were not persisted.",
    }


def paths(pattern, picks=None):
    found = [path for path in sorted(Path().glob(pattern)) if not path.name.endswith(".tmp")]
    return [found[i] for i in picks] if picks is not None else found


def main():
    candidates = {
        "hotpotqa": {
            "reflA_three_vote": paths("data/hotpotqa/hotpotqa_caliper_planA_top5_reflA_full.json"),
            "v0_opt_run01": paths("data/hotpotqa/hotpotqa_v0_opt_reflect.json.s*", [0, 1]),
            "v0_opt_run23": paths("data/hotpotqa/hotpotqa_v0_opt_reflect.json.s*", [2, 3]),
        },
        "musique": {
            "v0_2stream": paths("data/musique/musique_v0_2stream_steps_reflect.json"),
            "v0_opt": paths("data/musique/musique_v0_opt_reflect.json.s*"),
        },
        "2wikimultihopqa": {
            "v0_ft_run01": paths("data/2wikimultihopqa/2wikimultihopqa_v0_ft_steps_reflect.json.s*", [0, 1]),
            "v0_ft_run23": paths("data/2wikimultihopqa/2wikimultihopqa_v0_ft_steps_reflect.json.s*", [2, 3]),
            "v0_opt_run01": paths("data/2wikimultihopqa/2wikimultihopqa_v0_opt_reflect.json.s*", [0, 1]),
            "v0_opt_run23": paths("data/2wikimultihopqa/2wikimultihopqa_v0_opt_reflect.json.s*", [2, 3]),
        },
    }
    v1 = {
        "hotpotqa": paths("data/hotpotqa/hotpotqa_caliper_planA_top5.json"),
        "musique": paths("data/musique/musique_v1_ubh_2stream.json"),
        "2wikimultihopqa": paths("data/2wikimultihopqa/2wikimultihopqa_caliper_planA_top5.json"),
    }
    output = {dataset: {name: compare(dataset, value, v1[dataset]) for name, value in variants.items()} for dataset, variants in candidates.items()}
    target = Path("reflection_analysis")
    target.mkdir(exist_ok=True)
    (target / "reflection_analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    for dataset, variants in output.items():
        for name, result in variants.items():
            print(dataset, name, f"n={result['paired_n']} changed={result['answer_changed_rate']:.2%} delta={result['paired_delta_f1']:+.4%}")


if __name__ == "__main__":
    main()
