"""Lightweight Adaptive-RAG baseline for the CALIPER closed-context setting."""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


MODEL = "/mnt/data/caliper/models/Qwen3.5-9B"
ANSWER_SYSTEM = "Answer the question. Output only the shortest final answer."
SEARCH_SYSTEM = (
    "Given a multi-hop question and current evidence, output one focused search query "
    "for the missing fact. Output the query only: no explanation and no prefix."
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--naive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--force-route", choices=["A", "B", "C"])
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def answer(row):
    value = row.get("answer")
    return value if isinstance(value, str) else (row.get("answers") or [""])[0]


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def bm25(query, documents, excluded, count=3):
    tokenized = [tokens(document) for document in documents]
    query_tokens = tokens(query)
    n = len(documents)
    average = sum(map(len, tokenized)) / max(n, 1)
    frequencies = Counter(word for doc in tokenized for word in set(doc))
    scores = []
    for index, doc in enumerate(tokenized):
        if index in excluded:
            continue
        counts = Counter(doc)
        score = 0.0
        for word in query_tokens:
            frequency = counts[word]
            if not frequency:
                continue
            inverse = math.log(1 + (n - frequencies[word] + 0.5) / (frequencies[word] + 0.5))
            score += inverse * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * len(doc) / average))
        scores.append((score, index))
    return [index for _, index in sorted(scores, reverse=True)[:count]]


def call(client, system, prompt, seed, max_tokens=128):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,
        seed=seed,
        timeout=120,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (response.choices[0].message.content or "").strip()


def main():
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    root = args.base / "data" / args.dataset
    tests = load(root / "test.json")
    retrieval_cache = load(root / "test_retrieved.json")
    routes = load(args.routes)
    naive = {row["index"]: row for row in load(args.naive)}
    if len(routes) != len(tests) or any(r["question"] != q["question"] for r, q in zip(routes, tests)):
        raise ValueError("route/test mismatch")
    if len(naive) != len(tests):
        raise ValueError("naive/test mismatch")
    if len(retrieval_cache) != len(tests):
        raise ValueError("retrieval cache/test mismatch")

    indices = list(range(len(tests)))
    if args.sample_size:
        indices = sorted(random.Random(args.seed).sample(indices, args.sample_size))
    indices = indices[args.shard_index::args.shard_count]
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    def predict(index):
        row = tests[index]
        route = args.force_route or routes[index]["route"]
        started = time.perf_counter()
        if route == "B":
            cached = naive[index]
            return {
                "index": index, "id": row.get("id", str(index)), "question": row["question"],
                "answer": answer(row), "pred": cached["pred"], "route": route,
                "latency_seconds": cached.get("latency_seconds", 0), "retrieval_count": 1,
                "llm_calls": 1, "trace": [], "error": cached.get("error", ""),
            }

        error = ""
        prediction = ""
        trace = []
        calls = retrievals = 0
        try:
            if route == "A":
                prediction = call(client, ANSWER_SYSTEM, row["question"], args.seed)
                calls = 1
            else:
                documents = [sentence for ctx in row["ctxs"] for sentence in ctx["sentences"]]
                evidence = list(retrieval_cache[index]["retrieved_text"][:5])
                selected = {i for i, document in enumerate(documents) if document in set(evidence)}
                query = row["question"]
                retrievals = 1
                for step in range(1, args.max_steps):
                    prompt = (
                        f"Question:\n{row['question']}\n\nEvidence:\n" +
                        "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence))
                    )
                    response = call(client, SEARCH_SYSTEM, prompt, args.seed, max_tokens=64)
                    calls += 1
                    searches = re.findall(r"(?im)^SEARCH\s*:\s*(.+)$", response)
                    if searches:
                        query = " ".join(searches[-2:]).strip()
                    else:
                        query = next((line.strip() for line in response.splitlines() if line.strip()), response)
                    picked = bm25(query, documents, selected)
                    selected.update(picked)
                    evidence.extend(documents[i] for i in picked)
                    retrievals += 1
                    trace.append({"step": step, "query": query, "document_indices": picked, "response": response})
                prompt = (
                    f"Question:\n{row['question']}\n\nEvidence:\n" +
                    "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence))
                )
                prediction = call(client, ANSWER_SYSTEM, prompt, args.seed)
                calls += 1
        except Exception as exception:
            error = repr(exception)
        return {
            "index": index, "id": row.get("id", str(index)), "question": row["question"],
            "answer": answer(row), "pred": prediction, "route": route,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "retrieval_count": retrievals, "llm_calls": calls, "trace": trace, "error": error,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(predict, indices))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    counts = {label: sum(row["route"] == label for row in results) for label in "ABC"}
    empty = sum(not row["pred"].strip() for row in results)
    print(f"dataset={args.dataset} rows={len(results)} routes={counts} empty={empty} seconds={time.perf_counter()-started:.1f}")
    if empty:
        raise RuntimeError(f"{empty} empty predictions")


if __name__ == "__main__":
    main()
