"""hotpotqa CALIPER (pure-instruction + extract) 云端 steps 批处理。
用法（在远程 CALIPER 目录下）:
  python scripts/hotpotqa_batch.py --stage check   # 查看 batch 状态
  python scripts/hotpotqa_batch.py --stage submit  # 上传 JSONL 并创建 batch
  python scripts/hotpotqa_batch.py --stage poll    # 轮询 batch 状态
  python scripts/hotpotqa_batch.py --stage download # 下载结果 JSONL
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE = Path("/mnt/data/caliper/code/CALIPER")
load_dotenv(BASE / ".env")

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
CLOUD_MODEL = "qwen3.7-plus"
DASH_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

WORK = BASE / "batch_hotpotqa_caliper"
INPUT_JSONL = WORK / "hotpotqa_caliper_steps.jsonl"
RESULT_JSONL = WORK / "hotpotqa_caliper_steps_result.jsonl"
BATCH_STATE = WORK / "batch_state.json"


def _client():
    return OpenAI(api_key=ALIYUN_API_KEY, base_url=DASH_BASE, timeout=300)


def check():
    state = json.load(open(BATCH_STATE, encoding="utf-8"))
    client = _client()
    b = client.batches.retrieve(batch_id=state["batch_id"])
    print("batch_id:", state["batch_id"])
    print("status:", b.status)
    print("request_counts:", getattr(b, "request_counts", None))
    print("output_file_id:", getattr(b, "output_file_id", None))


def submit():
    if not INPUT_JSONL.exists():
        print("input jsonl not found:", INPUT_JSONL)
        return
    client = _client()
    n = sum(1 for _ in open(INPUT_JSONL, encoding="utf-8"))
    print("input lines:", n)
    file_obj = client.files.create(file=open(INPUT_JSONL, "rb"), purpose="batch")
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
    check()


def download():
    state = json.load(open(BATCH_STATE, encoding="utf-8"))
    client = _client()
    b = client.batches.retrieve(batch_id=state["batch_id"])
    if b.status != "completed":
        print("batch status:", b.status, "- not completed")
        return
    out_id = getattr(b, "output_file_id", None)
    if not out_id:
        print("no output file")
        return
    content = client.files.content(out_id)
    content.write_to_file(str(RESULT_JSONL))
    n = sum(1 for _ in open(RESULT_JSONL, encoding="utf-8"))
    print("downloaded", RESULT_JSONL, "lines:", n)


def list_batches():
    client = _client()
    bs = client.batches.list(limit=20)
    for b in bs.data:
        print(
            b.id,
            "|", b.status,
            "|", getattr(b, "request_counts", None),
            "|", b.created_at,
        )


def file_info():
    state = json.load(open(BATCH_STATE, encoding="utf-8"))
    client = _client()
    f = client.files.retrieve(state["file_id"])
    print("filename:", getattr(f, "filename", None))
    print("bytes:", getattr(f, "bytes", None))
    print("created_at:", getattr(f, "created_at", None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["check", "submit", "poll", "download", "list", "file"])
    args = parser.parse_args()
    {"check": check, "submit": submit, "poll": poll, "download": download, "list": list_batches, "file": file_info}[args.stage]()


if __name__ == "__main__":
    main()
