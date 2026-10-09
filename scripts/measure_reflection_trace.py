"""HotpotQA reflection transparency audit on 200 seed-42 UBH questions."""
import json
import os
import random
import re
import statistics
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


ROOT = Path("/mnt/data/caliper/code/CALIPER")
MODEL = "/mnt/data/caliper/models/Qwen3.5-9B"
PORTS = (6010, 6011, 6012, 6013)
MAX_ITER = 3
N = int(os.getenv("REFLECTION_N", "200"))
SEED = 42
REFLECT = ("You are an advanced reasoning agent that can improve based on self reflection. You will be given a question, "
           "reasoning steps, relevant corpus and a previous answer. Check whether the previous answer is consistent with the reasoning "
           "steps and the corpus. If it is consistent and reasonable, only output 'Correct'. Otherwise, diagnose a possible reason for "
           "failure and devise a new, concise, high level plan in a few sentences.")
EXECUTE = "According to the given steps and corpus, answer the question. Output the executing result of every step in simple sentences."
EXTRACT = "Extract the final answer from the given reasoning steps. Output only the final answer, as short as possible."
punctuation = set(string.punctuation)


def call(client, system, user, max_tokens):
    started = time.perf_counter()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=max_tokens, timeout=120,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return response.choices[0].message.content or "", time.perf_counter() - started, None
        except Exception as error:
            if attempt == 2:
                return "", time.perf_counter() - started, repr(error)
            time.sleep(2)


def normalize(text):
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    return " ".join("".join(char for char in text if char not in punctuation).split())


def f1(prediction, gold):
    predicted, expected = normalize(prediction).split(), normalize(gold).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    precision = common / len(predicted) if predicted else 0
    recall = common / len(expected) if expected else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def corpus(item):
    contexts = {row["title"]: row["sentences"] for row in item["ctxs"]}
    return "\n".join(contexts[title][index] for title, index in item["supporting_facts"]
                     if title in contexts and index < len(contexts[title]))


steps = {}
for line in (ROOT / "batch_hotpotqa_caliper/hotpotqa_caliper_steps_result.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    response = row.get("response") or {}
    choices = (response.get("body") or response).get("choices") or []
    steps[row["custom_id"]] = choices[0].get("message", {}).get("content", "") if choices else ""
tests = json.loads((ROOT / "data/hotpotqa/test.json").read_text(encoding="utf-8"))
indices = sorted(random.Random(SEED).sample(range(len(tests)), N))


def process(pair):
    position, index = pair
    item = tests[index]
    client = OpenAI(api_key="vllm", base_url=f"http://localhost:{PORTS[position % len(PORTS)]}/v1", timeout=120)
    evidence, step_text = corpus(item), steps[f"hp_{index}"]
    prompt = f"----------Question----------\n{item['question']}\n----------Corpus----------\n{evidence}\n----------Steps----------\n{step_text}"
    reasoning, t1, e1 = call(client, EXECUTE, prompt, 256)
    initial, t2, e2 = call(client, EXTRACT, f"----------Steps----------\n{reasoning}\n----------Question----------\n{item['question']}", 128)
    current, evaluator_calls, revisions, reflection_seconds, trace = initial, 0, 0, 0.0, []
    for _ in range(MAX_ITER):
        user = (f"----------Steps----------\n{reasoning}\n----------Corpus----------\n{evidence}\n"
                f"----------Question----------\n{item['question']}\n----------Previous answer----------\n{current}")
        verdict, elapsed, error = call(client, REFLECT, user, 256)
        reflection_seconds += elapsed
        evaluator_calls += 1
        accepted = "correct" in verdict.lower() and "incorrect" not in verdict.lower()
        event = {"verdict": verdict, "accepted": accepted, "evaluator_seconds": elapsed, "error": error}
        trace.append(event)
        if accepted or error:
            break
        revised, elapsed, error = call(client, EXTRACT, f"----------Steps----------\n{reasoning}\n\n{verdict}\n\n----------Question----------\n{item['question']}", 128)
        reflection_seconds += elapsed
        revisions += 1
        event.update({"revision": revised, "revision_seconds": elapsed, "revision_error": error})
        if error:
            break
        current = revised
    gold = item["answers"][0]
    return {
        "index": index, "question": item["question"], "gold": gold, "initial_prediction": initial,
        "final_prediction": current, "initial_F1": f1(initial, gold), "final_F1": f1(current, gold),
        "base_seconds": t1 + t2, "reflection_seconds": reflection_seconds, "total_seconds": t1 + t2 + reflection_seconds,
        "evaluator_calls": evaluator_calls, "revision_iterations": revisions, "trace": trace,
        "base_errors": [error for error in (e1, e2) if error],
    }


with ThreadPoolExecutor(max_workers=16) as pool:
    rows = list(pool.map(process, enumerate(indices)))


def stats(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}


valid = [row for row in rows if not row["base_errors"]]
summary = {
    "n": len(rows), "valid_n": len(valid), "seed": SEED, "max_reflection_iterations": MAX_ITER,
    "model": MODEL, "setting": "UBH gold supporting sentences with frozen cloud-generated steps",
    "initial_F1": statistics.mean(row["initial_F1"] for row in valid) * 100,
    "final_F1": statistics.mean(row["final_F1"] for row in valid) * 100,
    "delta_F1": statistics.mean(row["final_F1"] - row["initial_F1"] for row in valid) * 100,
    "evaluator_calls": stats([row["evaluator_calls"] for row in valid]),
    "revision_iterations": stats([row["revision_iterations"] for row in valid]),
    "immediate_accept_rate": sum(row["revision_iterations"] == 0 for row in valid) / len(valid) * 100,
    "max_iteration_rate": sum(row["revision_iterations"] == MAX_ITER for row in valid) / len(valid) * 100,
    "base_seconds": stats([row["base_seconds"] for row in valid]),
    "reflection_overhead_seconds": stats([row["reflection_seconds"] for row in valid]),
    "total_seconds": stats([row["total_seconds"] for row in valid]),
}
target = ROOT / f"reflection_analysis/reflection_trace_hotpotqa_n{N}_seed42.json"
target.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
