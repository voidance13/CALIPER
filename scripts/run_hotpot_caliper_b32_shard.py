"""Generate one resumable shard of HotpotQA CALIPER B32 predictions."""
import concurrent.futures
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE = Path("/mnt/data/caliper/code/CALIPER")
SHARD, SHARDS, PORT = map(int, sys.argv[1:4])
OUT = BASE / f"full_runs/hotpotqa_caliper_B32/part{SHARD}.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="EMPTY", timeout=180)
model = "/mnt/data/caliper/models/Qwen3.5-9B"

cache = json.load(open(BASE / "data/hotpotqa/test_retrieved_embedderft_rerankerft_v2.json", encoding="utf-8"))
steps = {}
for row in json.load(open(BASE / "data/hotpotqa/hybrid_pred_7b_72b_sep_clf_t2caliper_think.json", encoding="utf-8")):
    value = row.get("steps") or row.get("res1") or ""
    if isinstance(value, str) and value.strip():
        steps[row["question"]] = value

done = set()
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["question"])
        except (json.JSONDecodeError, KeyError):
            pass
items = [row for index, row in enumerate(cache) if index % SHARDS == SHARD and row["question"] not in done]


def call(prompt, max_tokens):
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or ""
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def generate(row):
    started = time.time()
    question = row["question"]
    context = row.get("retrieved_text") or [sentence for ctx in row.get("ctxs", []) for sentence in ctx.get("sentences", [])]
    context = "\n".join(f"[{index + 1}] {sentence}" for index, sentence in enumerate(context))
    hints = "\n".join(f"Step {index + 1}: {step}" for index, step in enumerate(steps.get(question, "").split("\n")))
    reasoning = call(
        f"Question: {question}\n\nReference: {context}\n\nThe following steps are hints from an expert (for reference only):\n{hints}\n\nReason and answer, but do not be limited by the hints.",
        256,
    )
    prediction = call(
        f"Question: {question}\n\nGiven the reasoning process:\n{reasoning}\n\nExtract and output the final answer concisely. Output only the answer itself, no extra words.",
        32,
    )
    return {"question": question, "pred": prediction, "latency_seconds": time.time() - started}


print(f"shard={SHARD} port={PORT} total={len(items) + len(done)} remaining={len(items)}", flush=True)
started = time.time()
with open(OUT, "a", encoding="utf-8", buffering=1) as output, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(generate, row) for row in items]
    for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
        output.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
        if completed % 100 == 0 or completed == len(items):
            print(f"shard={SHARD} completed={completed}/{len(items)} elapsed={time.time() - started:.0f}s", flush=True)

