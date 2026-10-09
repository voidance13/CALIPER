"""Table 4: fixed seed-42 sample, identical concurrency, persisted per-question latency."""
import concurrent.futures, json, os, random, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

import fasttext, jieba
from openai import OpenAI

BASE = Path("/mnt/data/caliper/code/CALIPER")
DS, PORT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "full"
N, WARMUP, WORKERS, TAU = 200, 20, 4, 0.997
for line in (BASE / ".env").read_text(encoding="utf-8").splitlines():
    if line.strip() and "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())

local = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="EMPTY", timeout=120)
cloud = OpenAI(api_key=os.environ["ALIYUN_API_KEY"], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=120)
classifier = fasttext.load_model(str(BASE / "query_clf/train_new/9b_clf.bin"))
stopwords = {x.strip() for x in open(BASE / "query_clf/stopwords.txt", encoding="utf-8")}
router_ms = {"hotpotqa": 32.48, "musique": 32.05, "2wikimultihopqa": 32.13}[DS]

cache = json.load(open(BASE / f"data/{DS}/test_retrieved_embedderft_rerankerft_v2.json", encoding="utf-8"))
rng = random.Random(42)
sample = rng.sample(cache, N)

def context(item):
    texts = item.get("retrieved_text") or [s for c in item.get("ctxs", []) for s in c.get("sentences", [])]
    return "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(texts[:5]))

def call_local(prompt, max_tokens):
    result = local.chat.completions.create(
        model="/mnt/data/caliper/models/Qwen3.5-9B", temperature=0.1, max_tokens=max_tokens,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return result.choices[0].message.content or ""

def base(item):
    q, ctx = item["question"], context(item)
    start = time.perf_counter()
    try:
        call_local(f"Question: {q}\n\nReference: {ctx}\n\nReason step by step, then output the final answer on the last line as 'Answer: <entity>'. Answer as briefly as possible.", 128)
        return {"seconds": time.perf_counter() - start, "error": None}
    except Exception as error:
        return {"seconds": time.perf_counter() - start, "error": str(error)}

def caliper(item):
    q, ctx = item["question"], context(item)
    start = time.perf_counter()
    try:
        guide = cloud.chat.completions.create(
            model="qwen3.7-plus", temperature=0.2, max_tokens=64,
            messages=[{"role": "system", "content": "You are an expert. Given a question, output step-by-step instructions."}, {"role": "user", "content": q}],
            extra_body={"enable_thinking": False},
        ).choices[0].message.content or ""
        reasoning = call_local(f"Question: {q}\n\nReference: {ctx}\n\nThe following steps are hints from an expert (for reference only):\n{guide}\n\nReason and answer, but do not be limited by the hints.", 96)
        call_local(f"Question: {q}\n\nGiven the reasoning process:\n{reasoning}\n\nExtract and output the final answer concisely. Output only the answer itself, no extra words.", 32)
        return {"seconds": time.perf_counter() - start, "error": None}
    except Exception as error:
        return {"seconds": time.perf_counter() - start, "error": str(error)}

def direct(item):
    q, ctx = item["question"], context(item)
    start = time.perf_counter()
    try:
        cloud.chat.completions.create(
            model="qwen3.7-plus", temperature=0.2, max_tokens=128,
            messages=[{"role": "system", "content": "Answer the question using the reference. Output only a short final answer."}, {"role": "user", "content": f"Question: {q}\n\nReference: {ctx}"}],
            extra_body={"enable_thinking": False},
        )
        return {"seconds": time.perf_counter() - start, "error": None}
    except Exception as error:
        return {"seconds": time.perf_counter() - start, "error": str(error)}

def use_local(question):
    text = " ".join(x for x in jieba.lcut(question) if len(x) > 1 and x not in stopwords)
    labels, probabilities = classifier.predict(text, k=2)
    easy = dict(zip((x.removeprefix("__label__") for x in labels), probabilities)).get("easy", 0.0)
    return easy >= TAU

def run(fn, rows):
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(fn, rows))

def caliper_clf_actual(item):
    start = time.perf_counter()
    route = "local" if use_local(item["question"]) else "collaborative"
    selected = base(item) if route == "local" else caliper(item)
    return {"seconds": time.perf_counter() - start, "error": selected["error"], "route": route}

if MODE == "actual_clf":
    run(caliper_clf_actual, sample[:WARMUP])
    values = run(caliper_clf_actual, sample)
    valid = sorted(x["seconds"] for x in values if x["error"] is None)
    summary_actual = {
        "valid_n": len(valid), "errors": len(values) - len(valid),
        "mean": statistics.mean(valid), "p50": statistics.median(valid),
        "p95": valid[int(0.95 * len(valid)) - 1],
    }
    result = {
        "dataset": DS, "n": N, "seed": 42, "warmup": WARMUP, "workers": WORKERS,
        "hardware": "4xA40 server; one vLLM replica assigned per dataset", "threshold": TAU,
        "latency_type": "measured end-to-end classifier routing and selected execution path",
        "created_at": datetime.now(timezone.utc).isoformat(), "summary_seconds": summary_actual,
        "rows": [{"question": item["question"], **values[i]} for i, item in enumerate(sample)],
    }
    out = BASE / f"table-result/table4_clf_actual_{DS}_seed42.json"
    json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    assert len(result["rows"]) == N and summary_actual["valid_n"] + summary_actual["errors"] == N
    print(json.dumps(summary_actual, ensure_ascii=False), flush=True)
    sys.exit(0)

run(base, sample[:WARMUP]); run(caliper, sample[:WARMUP]); run(direct, sample[:WARMUP])
base_times, caliper_times, cloud_times = run(base, sample), run(caliper, sample), run(direct, sample)
routes = [use_local(x["question"]) for x in sample]
clf_times = []
for i, route in enumerate(routes):
    selected = base_times[i] if route else caliper_times[i]
    clf_times.append({"seconds": selected["seconds"] + router_ms / 1000, "error": selected["error"]})

def summary(values):
    valid = [x["seconds"] for x in values if x["error"] is None]
    ordered = sorted(valid)
    return {"valid_n": len(valid), "errors": len(values) - len(valid), "mean": statistics.mean(valid),
            "p50": statistics.median(valid), "p95": ordered[int(0.95 * len(ordered)) - 1]}

result = {
    "dataset": DS, "n": N, "seed": 42, "warmup": WARMUP, "workers": WORKERS,
    "hardware": "4xA40 server; one vLLM replica assigned per dataset", "threshold": TAU,
    "clf_latency_type": "per-question counterfactual estimate from measured base/CALIPER latency plus measured batch-1 router mean",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "summary_seconds": {"baseline": summary(base_times), "caliper_clf": summary(clf_times), "caliper": summary(caliper_times), "cloud": summary(cloud_times)},
    "rows": [{"question": item["question"], "route": "local" if routes[i] else "collaborative",
              "baseline": base_times[i], "caliper_clf": clf_times[i], "caliper": caliper_times[i], "cloud": cloud_times[i]}
             for i, item in enumerate(sample)],
}
out = BASE / f"table-result/table4_timing_{DS}_seed42.json"
json.dump(result, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
assert len(result["rows"]) == N and result["summary_seconds"]["baseline"]["errors"] == 0
print(json.dumps(result["summary_seconds"], ensure_ascii=False), flush=True)
