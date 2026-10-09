"""从 batch 结果读取云端 steps，本地 7B 执行 + 提取 + 反思，输出 CALIPER 全量预测。

用法:
  python hybrid_predict_from_steps.py --result batch_hotpotqa_caliper/steps_result.jsonl \
      --tag instr_extract_full --limit 0
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


BASE = Path("/mnt/data/caliper/code/CALIPER")
sys.path.insert(0, str(BASE))
load_dotenv(BASE / ".env")

LOCAL_MODEL = "Qwen2.5-7B"
VLLM_BASE = "http://localhost:6010/v1"

REFLECT_INSTRUCTION_FIX = (
    "You are an advanced reasoning agent that can improve based on self reflection. "
    "You will be given a question, reasoning steps, relevant corpus and a previous answer. "
    "Check whether the previous answer is consistent with the reasoning steps and the corpus. "
    "If it is consistent and reasonable, only output \"Correct\". Otherwise, diagnose a possible "
    "reason for failure and devise a new, concise, high level plan that aims to mitigate the same "
    "failure in a few sentences. Use complete sentences."
)

INSTRUCTION1 = (
    "You are a helpful assistant. According to the given steps and corpus, answer the given question. "
    "Output the executing result of every step in simple sentences."
)
INSTRUCTION2 = "Extract the final answer from the given reasoning steps. Output only the final answer, as short as possible."


def gold_text(item):
    ctxs = [ctx["sentences"] for ctx in item["ctxs"]]
    title_to_idx = {ctx["title"]: i for i, ctx in enumerate(item["ctxs"])}
    parts = []
    for title, sent_idx in item["supporting_facts"]:
        ci = title_to_idx.get(title)
        if ci is not None and sent_idx < len(ctxs[ci]):
            parts.append(ctxs[ci][sent_idx])
    return "\n".join(parts)


def call(client, system, user, max_tokens=256):
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=LOCAL_MODEL,
                max_tokens=max_tokens,
                timeout=120,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception:
            time.sleep(2)
    return ""


def reflect_answer(client, item, text, res1, current, max_reflect_iter):
    reflection = ""
    iterations = 0
    for _ in range(max_reflect_iter):
        user_content = ("----------Steps----------\n" + res1
                        + "\n\n----------Corpus----------\n" + text
                        + "\n\n----------Question----------\n" + item["question"]
                        + "\n\n----------Previous answer----------\n" + current)
        response_text = call(client, REFLECT_INSTRUCTION_FIX, user_content, 256)
        reflection = response_text
        if ("Correct" in reflection or "correct" in reflection) and "incorrect" not in reflection and "Incorrect" not in reflection:
            break
        iterations += 1
        user2 = ("----------Steps----------\n" + res1 + "\n\n" + reflection
                 + "\n\n----------Question----------\n" + item["question"])
        current = call(client, INSTRUCTION2, user2, 128)
    return current, reflection, iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="batch 结果 jsonl（custom_id -> response）")
    parser.add_argument("--tag", default="from_steps", type=str)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--max_workers", default=8, type=int)
    parser.add_argument("--use_reflection", action="store_true")
    parser.add_argument("--max_reflect_iter", default=3, type=int)
    args = parser.parse_args()

    steps = {}
    for line in open(BASE / args.result, encoding="utf-8"):
        rec = json.loads(line)
        choices = (rec.get("response") or {}).get("choices") or []
        steps[rec["custom_id"]] = (choices[0].get("message", {}).get("content") or "") if choices else ""
    print("steps loaded:", len(steps))

    dev = json.load(open(BASE / "data/hotpotqa/test.json", encoding="utf-8"))
    if args.limit > 0:
        dev = dev[: args.limit]

    client = OpenAI(api_key="vllm", base_url=VLLM_BASE, timeout=120)

    def process_item(idx, item):
        text = gold_text(item)
        step_text = steps.get(f"hp_{idx}", "")
        prompt = ("----------Question----------\n" + item["question"]
                  + "----------Corpus----------\n" + text
                  + "\n\n----------Steps----------\n" + step_text
                  + "\n\n----------Question----------\n" + item["question"])
        res1 = call(client, INSTRUCTION1, prompt, 256)
        res = call(client, INSTRUCTION2, "----------Steps----------\n" + res1 + "\n\n----------Question----------\n" + item["question"], 128)
        pred = res
        reflection = ""
        reflect_iters = 0
        if args.use_reflection:
            pred, reflection, reflect_iters = reflect_answer(client, item, text, res1, res, args.max_reflect_iter)
        return idx, {
            "question": item["question"],
            "answer": item["answers"][0],
            "steps": step_text,
            "res1": res1,
            "reflection": reflection,
            "reflection_iterations": reflect_iters,
            "pred": pred,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, answer in tqdm(executor.map(process_item, range(len(dev)), dev), total=len(dev)):
            results.append((idx, answer))
    results.sort(key=lambda x: x[0])
    answers = [a for _, a in results]

    tag_suffix = f"_{args.tag}" if args.tag else ""
    out_path = BASE / f"data/hotpotqa/hybrid_pred_7b_72b_sep_use_best_hit{tag_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    print("written:", out_path, len(answers))

    from evaluate import evaluate
    eval_res = evaluate(answers)
    eval_path = BASE / f"data/hotpotqa/eval_hybrid_pred_7b_72b_sep_use_best_hit{tag_suffix}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, ensure_ascii=False, indent=4)
    print("eval written:", eval_path)


if __name__ == "__main__":
    main()
