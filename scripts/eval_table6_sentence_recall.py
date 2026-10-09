"""Table 6a: per-question supporting-sentence recall on frozen retrieval caches."""
import hashlib
import json
from pathlib import Path

BASE = Path("/mnt/data/caliper/code/CALIPER")
SOURCES = {
    "BASE": "test_retrieved.json",
    "FT-SEP": "test_retrieved_embedderft_rerankerft_v2.json",
    "FT-MER": "test_retrieved_mergedft_rerankerft.json",
}


def sha256(path):
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def gold_sentences(row):
    contexts = {context["title"]: context for context in row["ctxs"]}
    gold = []
    for title, sentence_index in row["supporting_facts"]:
        context = contexts.get(title)
        if context and sentence_index < len(context["sentences"]):
            gold.append(context["sentences"][sentence_index])
    return gold


results = {}
for dataset in ("hotpotqa", "2wikimultihopqa"):
    tests = json.load((BASE / f"data/{dataset}/test.json").open(encoding="utf-8"))
    gold = [gold_sentences(row) for row in tests]
    assert all(gold)
    results[dataset] = {}
    for method, filename in SOURCES.items():
        path = BASE / f"data/{dataset}/{filename}"
        rows = json.load(path.open(encoding="utf-8"))
        assert len(rows) == len(tests) and all(row["question"] == test["question"] for row, test in zip(rows, tests))
        sums = {3: 0.0, 5: 0.0, 10: 0.0}
        for row, expected in zip(rows, gold):
            retrieved = row["retrieved_text"]
            for k in sums:
                sums[k] += sum(sentence in retrieved[:k] for sentence in expected) / len(expected)
        results[dataset][method] = {
            "n": len(rows), **{f"R@{k}": round(sums[k] / len(rows) * 100, 2) for k in sums},
            "source": str(path.relative_to(BASE)), "source_sha256": sha256(path),
        }
        print(dataset, method, results[dataset][method], flush=True)
    results[dataset]["CALIPER"] = {**results[dataset]["FT-SEP"], "definition": "dataset-known domain setting; same retriever as FT-SEP"}

output = {
    "metric": "mean per-question fraction of official supporting sentences present in top-k retrieved sentences",
    "datasets": results,
    "evaluator": "scripts/eval_table6_sentence_recall.py",
}
(BASE / "table-result/table6_sentence_recall_final.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
