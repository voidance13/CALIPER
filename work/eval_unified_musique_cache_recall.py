"""Audit unified MuSiQue test caches with official supporting facts."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/mnt/data/caliper/code/CALIPER")
TEST = BASE / "data/musique/test.json"
SOURCES = {
    "BASE": BASE / "data/musique/test_retrieved_unified_bgebase_base.json",
    "FT-SEP (e2e)": BASE / "data/musique/test_retrieved_unified_bgebase_ftsep_e2e.json",
    "FT-SEP (Table 8)": BASE / "data/musique/test_retrieved_unified_bgebase_ftsep_table8.json",
    "FT-MER": BASE / "data/musique/test_retrieved_unified_bgebase_ftmer.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gold(row):
    contexts = {context["title"]: context for context in row["ctxs"]}
    facts = []
    for title, sentence_index in row["supporting_facts"]:
        context = contexts.get(title)
        if context and sentence_index < len(context["sentences"]):
            facts.append((title, context["sentences"][sentence_index]))
    return facts


def evaluate(rows, tests):
    assert len(rows) == len(tests)
    assert all(row.get("question") == test["question"] for row, test in zip(rows, tests))
    sentence_scores = {k: [] for k in (3, 5, 10)}
    paragraph_scores = {k: [] for k in (3, 5, 10)}
    for row, test in zip(rows, tests):
        facts = gold(test)
        assert facts
        sentence_to_titles = defaultdict(set)
        for context in test["ctxs"]:
            for sentence in context["sentences"]:
                sentence_to_titles[sentence].add(context["title"])
        supporting_titles = {title for title, _ in facts}
        for k in sentence_scores:
            retrieved = row["retrieved_text"][:k]
            sentence_scores[k].append(sum(sentence in retrieved for _, sentence in facts) / len(facts))
            retrieved_titles = set().union(*(sentence_to_titles[text] for text in retrieved))
            paragraph_scores[k].append(len(retrieved_titles & supporting_titles) / len(supporting_titles))
    return {
        "n": len(rows),
        "supporting_sentence_recall": {
            f"R@{k}": round(sum(values) / len(values) * 100, 2)
            for k, values in sentence_scores.items()
        },
        "supporting_paragraph_recall": {
            f"R@{k}": round(sum(values) / len(values) * 100, 2)
            for k, values in paragraph_scores.items()
        },
    }


def integrity(rows, tests):
    questions = [row.get("question") for row in rows]
    return {
        "rows": len(rows),
        "test_rows": len(tests),
        "question_order_match": all(
            row.get("question") == test["question"] for row, test in zip(rows, tests)
        ),
        "unique_questions": len(set(questions)),
        "bad_ctx_count_not_20": sum(len(row.get("ctxs", [])) != 20 for row in rows),
        "bad_retrieved_text_count_not_10": sum(
            len(row.get("retrieved_text", [])) != 10 for row in rows
        ),
        "empty_retrieved_text": sum(not row.get("retrieved_text") for row in rows),
    }


tests = json.loads(TEST.read_text(encoding="utf-8"))
results = {}
for name, path in SOURCES.items():
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = evaluate(rows, tests)
    result.update({
        "source": str(path.relative_to(BASE)),
        "source_sha256": sha256(path),
        "integrity": integrity(rows, tests),
    })
    results[name] = result
    print(name, json.dumps(result, ensure_ascii=False), flush=True)

output = {
    "evaluator": "work/eval_unified_musique_cache_recall.py",
    "dataset": "MuSiQue test.json",
    "scope": "2,417-question test-cache audit; not the 4,293-question official-dev Table 8 protocol",
    "alignment": "cache rows must match test rows by question in order",
    "results": results,
}
target = BASE / "table-result/musique_unified_bgebase_test_cache_recall.json"
target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print("SAVED", target, flush=True)
