"""musique 种子云端步骤生成的 Batch File 提交/轮询/下载 + 本地打标。

用法:
  python musique_seed_batch.py --stage build          # 构建 JSONL + 映射文件
  python musique_seed_batch.py --stage submit         # 上传 JSONL 并创建 batch
  python musique_seed_batch.py --stage poll           # 轮询 batch 状态
  python musique_seed_batch.py --stage download       # 下载结果 JSONL
  python musique_seed_batch.py --stage local          # 本地 7B 打标（需 vllm 空闲）
"""
import argparse
import json
import os
import string
import sys
import time
from pathlib import Path

import regex
from dotenv import load_dotenv
from openai import OpenAI


BASE = Path("/mnt/data/caliper/code/CALIPER")
sys.path.insert(0, str(BASE))
load_dotenv(BASE / ".env")

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
CLOUD_MODEL = "qwen3.7-plus"
LOCAL_MODEL = "Qwen2.5-7B"
DASH_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLLM_BASE = "http://localhost:6011/v1"

WORK = BASE / "batch_musique"
TRAIN_JSONL = WORK / "musique_train_steps.jsonl"
TEST_JSONL = WORK / "musique_test_steps.jsonl"
META = WORK / "meta.json"
RESULT_JSONL = WORK / "musique_steps_result.jsonl"
BATCH_STATE = WORK / "batch_state.json"

STEPS_SYS = (
    "You are a helpful assistant. Just output steps to answer the given question. "
    "Don't use any information that is not given. Don't give any example."
)


def build():
    WORK.mkdir(exist_ok=True)
    with open(BASE / "data/musique/test.json", encoding="utf-8") as f:
        test = json.load(f)
    test_ids = {str(s["id"]) for s in test}
    with open(BASE / "data/musique/train.json", encoding="utf-8") as f:
        train_full = json.load(f)

    meta = {"train": [], "test": []}
    n_train = 0
    with open(TRAIN_JSONL, "w", encoding="utf-8") as out:
        for orig_idx, s in enumerate(train_full[:10000]):
            if str(s["id"]) in test_ids:
                continue
            cid = f"train_{orig_idx}"
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CLOUD_MODEL,
                    "messages": [
                        {"role": "system", "content": STEPS_SYS},
                        {"role": "user", "content": s["question"]},
                    ],
                    "max_tokens": 256,
                    "enable_thinking": True,
                    "thinking_budget": 512,
                },
            }
            out.write(json.dumps(req, ensure_ascii=False) + "\n")
            meta["train"].append({"custom_id": cid, "orig_idx": orig_idx, "id": str(s["id"])})
            n_train += 1

    n_test = 0
    with open(TEST_JSONL, "w", encoding="utf-8") as out:
        for orig_idx, s in enumerate(test[:2000]):
            cid = f"test_{orig_idx}"
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CLOUD_MODEL,
                    "messages": [
                        {"role": "system", "content": STEPS_SYS},
                        {"role": "user", "content": s["question"]},
                    ],
                    "max_tokens": 256,
                    "enable_thinking": True,
                    "thinking_budget": 512,
                },
            }
            out.write(json.dumps(req, ensure_ascii=False) + "\n")
            meta["test"].append({"custom_id": cid, "orig_idx": orig_idx, "id": str(s["id"])})
            n_test += 1

    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"train requests: {n_train}, test requests: {n_test}")
    print(f"wrote {TRAIN_JSONL}, {TEST_JSONL}, {META}")


def _client():
    return OpenAI(api_key=ALIYUN_API_KEY, base_url=DASH_BASE, timeout=300)


def submit():
    client = _client()
    if not TRAIN_JSONL.exists() or not TEST_JSONL.exists():
        print("JSONL 不存在，先运行 --stage build")
        return
    # 合并成一个输入文件
    merged = WORK / "musique_all_steps.jsonl"
    with open(merged, "w", encoding="utf-8") as out:
        for p in (TRAIN_JSONL, TEST_JSONL):
            for line in open(p, encoding="utf-8"):
                out.write(line)
    print(f"merged {merged} ({sum(1 for _ in open(merged, encoding='utf-8'))} lines)")

    file_obj = client.files.create(file=open(merged, "rb"), purpose="batch")
    print("file_id:", file_obj.id)
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print("batch_id:", batch.id)
    print("status:", batch.status)
    with open(BATCH_STATE, "w", encoding="utf-8") as f:
        json.dump(
            {"file_id": file_obj.id, "batch_id": batch.id, "submitted_at": time.time()},
            f,
            indent=2,
        )


def poll():
    state = json.load(open(BATCH_STATE, encoding="utf-8"))
    client = _client()
    b = client.batches.retrieve(batch_id=state["batch_id"])
    print("status:", b.status)
    print("request_counts:", getattr(b, "request_counts", None))
    print("output_file_id:", getattr(b, "output_file_id", None))
    if getattr(b, "errors", None):
        print("errors:", b.errors)


def download():
    state = json.load(open(BATCH_STATE, encoding="utf-8"))
    client = _client()
    b = client.batches.retrieve(batch_id=state["batch_id"])
    if b.status != "completed":
        print(f"batch 状态 {b.status}，未完成")
        return
    out_id = getattr(b, "output_file_id", None)
    if not out_id:
        print("无输出文件")
        return
    content = client.files.content(out_id)
    content.write_to_file(str(RESULT_JSONL))
    n = sum(1 for _ in open(RESULT_JSONL, encoding="utf-8"))
    print(f"downloaded {RESULT_JSONL}: {n} lines")


def _gold_text(item):
    ctxs = [ctx["sentences"] for ctx in item["ctxs"]]
    title_to_idx = {ctx["title"]: i for i, ctx in enumerate(item["ctxs"])}
    parts = []
    for title, sent_idx in item["supporting_facts"]:
        ci = title_to_idx.get(title)
        if ci is not None and sent_idx < len(ctxs[ci]):
            parts.append(ctxs[ci][sent_idx])
    return "\n".join(parts)


def _normalize(s):
    def remove_articles(t):
        return regex.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _eval_hit(prediction, answer):
    return int(_normalize(answer) in _normalize(prediction))


def _local_call(client, system, user, max_tokens=256):
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


def local():
    if not RESULT_JSONL.exists():
        print("结果文件不存在，先 --stage download")
        return
    meta = json.load(open(META, encoding="utf-8"))
    steps = {}
    for line in open(RESULT_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        cid = rec["custom_id"]
        resp = rec.get("response", {})
        choices = resp.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content") or ""
        else:
            content = ""
        steps[cid] = content

    with open(BASE / "data/musique/test.json", encoding="utf-8") as f:
        test = json.load(f)
    with open(BASE / "data/musique/train.json", encoding="utf-8") as f:
        train_full = json.load(f)

    client = OpenAI(api_key="vllm", base_url=VLLM_BASE, timeout=120)
    sys1 = (
        "You are a helpful assistant. According to given articles, answer the given question. "
        "Think step by step. Output the executing result of every step. Then output the final "
        "answer as short as possible. If the question is a yes-no question, just output \"yes\" or \"no\"."
    )
    sys2 = (
        "You are a helpful assistant. According to given steps and articles, answer the given question. "
        "Output the executing result of every step. Then output the final answer as short as possible."
    )

    def label(cid, item):
        text = _gold_text(item)
        question = item["question"]
        pred_local = _local_call(client, sys1, "Question:\n" + question + "\n\nCorpus:\n" + text)
        hit_local = _eval_hit(pred_local, item["answers"][0])
        step_text = steps.get(cid, "")
        pred_hybrid = _local_call(
            client, sys2, "Question:\n" + question + "\n\nSteps:\n" + step_text + "\n\nCorpus:\n" + text
        )
        hit_hybrid = _eval_hit(pred_hybrid, item["answers"][0])
        if hit_local and hit_hybrid:
            intent = "easy"
        elif hit_hybrid:
            intent = "middle"
        else:
            intent = "hard"
        return {"id": item["id"], "question": question, "intent_type": intent}

    results = []
    missing = 0
    for m in meta["train"]:
        cid = m["custom_id"]
        if cid not in steps:
            missing += 1
            continue
        item = train_full[m["orig_idx"]]
        results.append(label(cid, item))
    out_path = BASE / "query_clf/train/musique10000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("written:", out_path, len(results), "missing:", missing)

    buckets = {"easy": [], "middle": [], "hard": []}
    missing = 0
    for m in meta["test"]:
        cid = m["custom_id"]
        if cid not in steps:
            missing += 1
            continue
        item = test[m["orig_idx"]]
        buckets[label(cid, item)["intent_type"]].append(label(cid, item))
    for intent in ("easy", "middle", "hard"):
        out_path = BASE / f"query_clf/train/musique_test2000_{intent}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(buckets[intent], f, ensure_ascii=False, indent=4)
        print("written:", out_path, len(buckets[intent]))
    print("test missing:", missing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["build", "submit", "poll", "download", "local"])
    args = parser.parse_args()
    {"build": build, "submit": submit, "poll": poll, "download": download, "local": local}[args.stage]()


if __name__ == "__main__":
    main()
