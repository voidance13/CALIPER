"""Measure two-stage cloud reasoning latency on the fixed Table 4 sample."""
import argparse
import concurrent.futures
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


BASE = Path("/mnt/data/caliper/code/CALIPER")
MODEL = "qwen3.7-plus"


def percentile(values, ratio):
    ordered = sorted(values)
    return ordered[max(0, int(ratio * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("hotpotqa", "musique", "2wikimultihopqa"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, choices=(256, 512), default=256)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()

    for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
        if line.strip() and "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    client = OpenAI(api_key=os.environ["ALIYUN_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=180)
    cache = json.load((BASE / f"data/{args.dataset}/test_retrieved_embedderft_rerankerft_v2.json").open(encoding="utf-8"))
    sample = random.Random(42).sample(cache, args.n)

    def context(item):
        texts = item.get("retrieved_text") or [sentence for ctx in item.get("ctxs", []) for sentence in ctx.get("sentences", [])]
        return "\n".join(f"[{index + 1}] {sentence}" for index, sentence in enumerate(texts[:5]))

    def call(item):
        question = item["question"]
        first_prompt = f"Question:\n{question}\n\nCorpus:\n{context(item)}"
        started = time.perf_counter()
        try:
            stage_started = time.perf_counter()
            first = client.chat.completions.create(
                model=MODEL, temperature=0.2, max_tokens=256,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. According to given articles, answer the given question. Think step by step. Output the executing result of every step."},
                    {"role": "user", "content": first_prompt},
                ],
                extra_body={"enable_thinking": True, "thinking_budget": args.thinking_budget},
            )
            stage1_seconds = time.perf_counter() - stage_started
            reasoning = first.choices[0].message.content or ""
            stage_started = time.perf_counter()
            second = client.chat.completions.create(
                model=MODEL, temperature=0.2, max_tokens=64,
                messages=[
                    {"role": "system", "content": "According to the given question and reasoning steps, output the complete final answer as short as possible. If the question is yes-no, output only yes or no."},
                    {"role": "user", "content": f"----------Question----------\n{question}\n\n----------Steps----------\n{reasoning}"},
                ],
                extra_body={"enable_thinking": False},
            )
            stage2_seconds = time.perf_counter() - stage_started
            usage1 = first.usage.model_dump() if first.usage else {}
            usage2 = second.usage.model_dump() if second.usage else {}
            return {
                "seconds": time.perf_counter() - started, "stage1_seconds": stage1_seconds,
                "stage2_seconds": stage2_seconds, "error": None,
                "prompt_tokens": (usage1.get("prompt_tokens") or 0) + (usage2.get("prompt_tokens") or 0),
                "completion_tokens": (usage1.get("completion_tokens") or 0) + (usage2.get("completion_tokens") or 0),
                "prediction": second.choices[0].message.content or "",
            }
        except Exception as error:
            return {"seconds": time.perf_counter() - started, "error": repr(error)[:500]}

    def run(rows):
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            return list(pool.map(call, rows))

    run(sample[:args.warmup])
    rows = run(sample)
    valid = [row for row in rows if row["error"] is None]
    seconds = [row["seconds"] for row in valid]
    summary = {
        "valid_n": len(valid), "errors": len(rows) - len(valid),
        "mean": statistics.mean(seconds), "p50": statistics.median(seconds), "p95": percentile(seconds, 0.95),
        "stage1_mean": statistics.mean(row["stage1_seconds"] for row in valid),
        "stage2_mean": statistics.mean(row["stage2_seconds"] for row in valid),
        "completion_tokens_mean": statistics.mean(row["completion_tokens"] for row in valid),
    }
    result = {
        "dataset": args.dataset, "n": args.n, "seed": 42, "warmup": args.warmup,
        "workers": args.workers, "model": MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": f"two-stage cloud reasoning; stage1 thinking_budget={args.thinking_budget},max_tokens=256; stage2 thinking off,max_tokens=64",
        "summary_seconds": summary,
        "rows": [{"question": item["question"], **rows[index]} for index, item in enumerate(sample)],
    }
    target = BASE / f"table-result/table4_cloud_deep_{args.dataset}_{args.tag}_seed42.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    assert len(result["rows"]) == args.n and summary["valid_n"] + summary["errors"] == args.n
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("SAVED", target, flush=True)


if __name__ == "__main__":
    main()
