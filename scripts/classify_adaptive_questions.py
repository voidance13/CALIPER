"""Classify CALIPER test questions into Adaptive-RAG routes A/B/C."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATASETS = ("hotpotqa", "musique", "2wikimultihopqa")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).cuda().eval()
    option_ids = torch.tensor(
        [tokenizer(x, add_special_tokens=False).input_ids[0] for x in "ABC"],
        device="cuda",
    )
    start_id = model.config.decoder_start_token_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        rows = json.loads(
            (args.base / "data" / dataset / "test.json").read_text(encoding="utf-8")
        )
        questions = [row["question"] for row in rows]
        results = []
        loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                inputs = tokenizer(
                    list(batch), padding=True, truncation=True, max_length=384,
                    return_tensors="pt",
                ).to("cuda")
                decoder = torch.full(
                    (len(batch), 1), start_id, dtype=torch.long, device="cuda"
                )
                logits = model(**inputs, decoder_input_ids=decoder).logits[:, 0, option_ids]
                probs = logits.float().softmax(dim=-1).cpu().tolist()
                labels = logits.argmax(dim=-1).cpu().tolist()
                for local_index, (label, scores) in enumerate(zip(labels, probs)):
                    index = offset + local_index
                    results.append({
                        "index": index,
                        "id": rows[index].get("id", str(index)),
                        "question": questions[index],
                        "route": "ABC"[label],
                        "probabilities": dict(zip("ABC", [round(x, 6) for x in scores])),
                    })
                offset += len(batch)
        output = args.output_dir / f"routes_{dataset}.json"
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = {label: sum(row["route"] == label for row in results) for label in "ABC"}
        print(dataset, len(results), counts, output, flush=True)


if __name__ == "__main__":
    main()
