"""Reproducible Naive RAG: base retrieval top-5 + one Qwen3.5-9B call."""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


MODEL = "/mnt/data/caliper/models/Qwen3.5-9B"
SYSTEM = (
    "You are a helpful assistant. According to given articles, answer the given "
    "question. Output the final answer as short as possible."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def gold_answer(row):
    answer = row.get("answer")
    return answer if isinstance(answer, str) else (row.get("answers") or [""])[0]


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    root = args.base.resolve() / "data" / args.dataset
    tests = load_json(root / "test.json")
    cache = load_json(root / "test_retrieved.json")
    if len(tests) != len(cache):
        raise ValueError("test/cache row count mismatch")
    if any(test["question"] != item["question"] for test, item in zip(tests, cache)):
        raise ValueError("test/cache question order mismatch")

    indices = list(range(len(cache)))
    if args.sample_size:
        indices = sorted(random.Random(args.seed).sample(indices, args.sample_size))

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    def predict(index):
        item = cache[index]
        context = "\n".join(
            f"[{rank + 1}] {text}"
            for rank, text in enumerate(item["retrieved_text"][:5])
        )
        prompt = f"Question:\n{item['question']}\n\nCorpus:\n{context}"
        error = ""
        prediction = ""
        started = time.perf_counter()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=256,
                    temperature=0,
                    seed=args.seed,
                    timeout=120,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                prediction = response.choices[0].message.content or ""
                error = ""
                break
            except Exception as exception:
                error = repr(exception)
                if attempt < 2:
                    time.sleep(2)
        return {
            "index": index,
            "question": item["question"],
            "answer": gold_answer(tests[index]),
            "pred": prediction,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": error,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(predict, indices))
    empty = sum(not row["pred"].strip() for row in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    temporary.replace(args.output)
    print(
        f"dataset={args.dataset} rows={len(results)} empty={empty} "
        f"seconds={time.perf_counter() - started:.1f} output={args.output}"
    )
    if empty:
        raise RuntimeError(f"{empty} empty predictions")


if __name__ == "__main__":
    main()
