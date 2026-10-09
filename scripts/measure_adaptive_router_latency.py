"""Measure online batch-1 latency of the trained Adaptive-RAG router."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda().eval()
    option_ids = torch.tensor(
        [tokenizer(x, add_special_tokens=False).input_ids[0] for x in "ABC"], device="cuda"
    )
    start_id = model.config.decoder_start_token_id

    @torch.inference_mode()
    def classify(question):
        inputs = tokenizer(
            question, truncation=True, max_length=384, return_tensors="pt"
        ).to("cuda")
        decoder = torch.full((1, 1), start_id, dtype=torch.long, device="cuda")
        logits = model(**inputs, decoder_input_ids=decoder).logits[0, 0, option_ids]
        torch.cuda.synchronize()
        return "ABC"[int(logits.argmax())]

    first = json.loads(
        (args.base / "data" / DATASETS[0] / "test.json").read_text(encoding="utf-8")
    )[0]["question"]
    for _ in range(args.warmup):
        classify(first)

    output = {}
    for dataset in DATASETS:
        tests = json.loads(
            (args.base / "data" / dataset / "test.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (args.routes / f"routes_{dataset}.json").read_text(encoding="utf-8")
        )
        indices = list(range(len(tests)))
        if args.sample_size:
            indices = sorted(random.Random(args.seed).sample(indices, min(args.sample_size, len(indices))))
        rows = []
        mismatches = 0
        for index in indices:
            test = tests[index]
            started = time.perf_counter()
            route = classify(test["question"])
            latency = time.perf_counter() - started
            mismatches += route != expected[index]["route"]
            rows.append({"index": index, "route": route, "latency_seconds": round(latency, 6)})
        output[dataset] = rows
        values = torch.tensor([row["latency_seconds"] for row in rows])
        print(
            dataset,
            {"n": len(rows), "mismatches": mismatches, "mean_ms": round(values.mean().item() * 1000, 3),
             "p95_ms": round(torch.quantile(values, 0.95).item() * 1000, 3)},
            flush=True,
        )
        args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    args.output.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
