import json
import re
import string
from collections import Counter
from pathlib import Path


def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def f1(prediction, gold):
    pred, answer = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(pred) & Counter(answer)).values())
    return 2 * common / (len(pred) + len(answer)) if common else float(pred == answer)


for dataset in ("hotpotqa", "musique", "2wikimultihopqa"):
    tests = json.load(open(f"data/{dataset}/test.json"))
    expected = len(tests)
    rows = []
    for path in Path(f"data/{dataset}").glob("*.json"):
        if not re.search(r"(hybrid_pred|caliper_planA|normal_pred)", path.name) or path.name.startswith("eval_"):
            continue
        try:
            data = json.load(path.open())
            if not isinstance(data, list) or len(data) != expected or not data:
                continue
            scores = [f1(x.get("pred", ""), x.get("answer", x.get("answers", [tests[i].get("answer", tests[i].get("answers", [""])[0])])[0])) for i, x in enumerate(data)]
            rows.append((sum(scores) / len(scores) * 100, path.as_posix()))
        except Exception:
            pass
    print(f"\n{dataset} full={expected}")
    for score, path in sorted(rows):
        print(f"{score:.4f}\t{path}")
