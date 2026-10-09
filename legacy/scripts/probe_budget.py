"""探测 thinking_budget 对纯指令 steps 的 token 消耗影响。

用法: python probe_budget.py --budget 256 [--budget 512]
输出: batch_hotpotqa_caliper/probe_budget_{N}.jsonl（与 hybrid_predict_from_steps.py 兼容）
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


BASE = Path("/mnt/data/caliper/code/CALIPER")
load_dotenv(BASE / ".env")

STEPS_SYS = (
    "You are an instructor. Given the question, output a numbered list of instructions for solving it. "
    "Each instruction must be an action such as \"Identify...\", \"Find...\", \"Check...\" or \"Compare...\". "
    "Do NOT include any conclusions, facts, or answers in the instructions. Do not answer the question."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()

    dev = json.load(open(BASE / "data/hotpotqa/test.json", encoding="utf-8"))[: args.limit]
    client = OpenAI(
        api_key=os.getenv("ALIYUN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=180,
    )
    out_dir = BASE / "batch_hotpotqa_caliper"
    out_dir.mkdir(exist_ok=True)

    for budget in args.budget:
        def call(idx_item):
            idx, item = idx_item
            try:
                resp = client.chat.completions.create(
                    model="qwen3.7-plus",
                    messages=[
                        {"role": "system", "content": STEPS_SYS},
                        {"role": "user", "content": item["question"]},
                    ],
                    max_tokens=512,
                    extra_body={"enable_thinking": True, "thinking_budget": budget},
                )
                msg = resp.choices[0].message
                usage = resp.usage
                return {
                    "idx": idx,
                    "content": msg.content or "",
                    "reasoning": getattr(msg, "reasoning_content", None) or "",
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "raw_usage": usage.model_dump() if usage else None,
                }
            except Exception as e:
                return {"idx": idx, "error": repr(e)[:200], "content": "", "reasoning": ""}

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in tqdm(ex.map(call, enumerate(dev)), total=len(dev)):
                results.append(r)
        results.sort(key=lambda r: r["idx"])

        cts = [r["completion_tokens"] for r in results if r.get("completion_tokens")]
        reason_lens = [len(r.get("reasoning", "")) for r in results]
        content_lens = [len(r.get("content", "")) for r in results]
        errs = sum(1 for r in results if r.get("error"))
        print(f"budget={budget}: n={len(results)} errs={errs} "
              f"completion_tokens avg={sum(cts)/len(cts) if cts else 0:.0f} "
              f"reasoning_chars avg={sum(reason_lens)/len(reason_lens):.0f} "
              f"content_chars avg={sum(content_lens)/len(content_lens):.0f}")
        if results and results[0].get("raw_usage"):
            print("  raw_usage sample:", json.dumps(results[0]["raw_usage"]))

        out_path = out_dir / f"probe_budget_{budget}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                rec = {
                    "custom_id": f"hp_{r['idx']}",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {"message": {"content": r.get("content", "")}}
                            ]
                        },
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("written:", out_path)


if __name__ == "__main__":
    main()
