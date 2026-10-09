"""Evaluate cached per-query domain routing with current official gold."""
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import fasttext
import jieba
import nltk


ROOT = Path("/mnt/data/caliper/code/CALIPER")
nltk.data.path.append("/home/gpt/nltk_data")
LABELS = {"hotpotqa": "hp", "musique": "ms", "2wikimultihopqa": "tw"}


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def recalls(scores):
    return {f"R@{k}": round(statistics.mean(row[k] for row in scores) * 100, 2) for k in (3, 5, 10)}


def canonical(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def sentence_scores(dataset):
    tests = load(f"data/{dataset}/test.json")
    path = f"data/{dataset}/test_retrieved_mix_ft.json"
    routed = load(path)
    assert len(tests) == len(routed) and all(a["question"] == b["question"] for a, b in zip(tests, routed))
    output = []
    for test, prediction in zip(tests, routed):
        contexts = {context["title"]: context for context in test["ctxs"]}
        gold = [contexts[title]["sentences"][index] for title, index in test["supporting_facts"]
                if title in contexts and index < len(contexts[title]["sentences"])]
        assert gold
        retrieved = prediction["retrieved_text"]
        output.append({k: sum(item in retrieved[:k] for item in gold) / len(gold) for k in (3, 5, 10)})
    return routed, output, path


def musique_scores():
    tests = [json.loads(line) for line in open("/tmp/musique_full_v1.0_dev.jsonl", encoding="utf-8")]
    path = "data/musique/test_retrieved_mix_ft.json"
    routed = load(path)
    assert len(tests) == len(routed) and all(a["question"] == b["question"] for a, b in zip(tests, routed))
    output = []
    kept = []
    for test, prediction in zip(tests, routed):
        sentence_paragraphs, supporting = {}, set()
        for paragraph_index, paragraph in enumerate(test["paragraphs"]):
            if paragraph["is_supporting"]:
                supporting.add(paragraph_index)
            for sentence in nltk.sent_tokenize(paragraph["paragraph_text"]):
                sentence_paragraphs.setdefault(canonical(sentence), set()).add(paragraph_index)
        if not supporting or not sentence_paragraphs:
            continue
        retrieved = [canonical(text) for text in prediction["retrieved_text"]]
        row = {}
        for k in (3, 5, 10):
            found = set().union(*(sentence_paragraphs.get(text, set()) for text in retrieved[:k]))
            row[k] = len(found & supporting) / len(supporting)
        output.append(row)
        kept.append(prediction)
    return kept, output, path


model_path = "query_clf/train/domain_clf.bin"
model = fasttext.load_model(str(ROOT / model_path))
stopwords = {line.strip() for line in (ROOT / "query_clf/stopwords.txt").open(encoding="utf-8")}


def predict(question):
    text = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
    return model.predict(text)[0][0].removeprefix("__label__")


datasets = {}
confusion = Counter()
for dataset in LABELS:
    routed, scores, source = musique_scores() if dataset == "musique" else sentence_scores(dataset)
    for row in routed:
        confusion[(LABELS[dataset], predict(row["question"]))] += 1
    datasets[dataset] = {
        "n": len(scores),
        **recalls(scores),
        "source": source,
        "source_sha256": sha(source),
    }

domain_accuracy = sum(value for (true, pred), value in confusion.items() if true == pred) / sum(confusion.values())
single = load("table-result/table6_sentence_recall_final.json")["datasets"]
musique = {
    name: load(f"table-result/table6_musique_paragraph_recall_{name.lower().replace('-', '')}.json")
    for name in ("BASE", "FT-SEP", "FT-MER")
}
methods = {}
for method in ("BASE", "FT-SEP", "FT-MER"):
    methods[method] = {
        f"R@{k}": round(statistics.mean((single["hotpotqa"][method][f"R@{k}"], musique[method][f"R@{k}"], single["2wikimultihopqa"][method][f"R@{k}"])), 2)
        for k in (3, 5, 10)
    }
methods["CALIPER-router"] = {
    f"R@{k}": round(statistics.mean(datasets[name][f"R@{k}"] for name in LABELS), 2) for k in (3, 5, 10)
}

output = {
    "definition": "full per-query domain-classifier routing caches re-evaluated with current official gold; Mixed is equal-weight macro over the three full domain results",
    "gold": {"hotpotqa": "supporting sentences", "musique": "official supporting paragraphs", "2wikimultihopqa": "supporting sentences"},
    "router": {"model": model_path, "model_sha256": sha(model_path), "accuracy": round(domain_accuracy * 100, 2),
               "confusion_true_by_pred": {true: {pred: confusion[(true, pred)] for pred in LABELS.values()} for true in LABELS.values()}},
    "caliper_router_by_domain": datasets,
    "mixed_macro": methods,
    "evaluator": "scripts/eval_table6_mixed_router_official.py",
}
target = ROOT / "table-result/table6_mixed_router_official.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(output, ensure_ascii=False, indent=2))
