"""Train the official Adaptive-RAG T5-Large A/B/C router on bundled labels."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class RouterDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["question"], max_length=384, truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=row["answer"], max_length=3, truncation=True, return_tensors="pt"
        )["input_ids"][0]
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
        }


def make_collator(tokenizer):
    def collate(rows):
        inputs = tokenizer.pad(
            [{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows],
            return_tensors="pt",
        )
        width = max(len(r["labels"]) for r in rows)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            labels[i, : len(row["labels"])] = row["labels"]
        inputs["labels"] = labels
        return inputs

    return collate


@torch.inference_mode()
def accuracy(model, loader, option_ids, device):
    model.eval()
    correct = total = 0
    start_id = model.config.decoder_start_token_id
    choices = torch.tensor(option_ids, device=device)
    for batch in loader:
        gold = batch.pop("labels")[:, 0].to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        decoder = torch.full((len(gold), 1), start_id, dtype=torch.long, device=device)
        logits = model(**batch, decoder_input_ids=decoder).logits[:, 0]
        pred = choices[logits[:, choices].argmax(dim=1)]
        correct += (pred == gold).sum().item()
        total += len(gold)
    return correct / total


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    train_rows = json.loads(args.train.read_text(encoding="utf-8"))
    valid_rows = json.loads(args.valid.read_text(encoding="utf-8"))
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_valid_samples:
        valid_rows = valid_rows[: args.max_valid_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    option_ids = [tokenizer(x, add_special_tokens=False).input_ids[0] for x in "ABC"]
    if len(set(option_ids)) != 3:
        raise ValueError(f"A/B/C token IDs are not distinct: {option_ids}")

    collator = make_collator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        RouterDataset(train_rows, tokenizer), batch_size=args.batch_size,
        shuffle=True, generator=generator, collate_fn=collator,
    )
    valid_loader = DataLoader(
        RouterDataset(valid_rows, tokenizer), batch_size=args.batch_size * 2,
        shuffle=False, collate_fn=collator,
    )
    optimizer = AdamW(model.parameters(), lr=3e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            loss_sum += loss.item() * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        score = accuracy(model, valid_loader, option_ids, device)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_loader), "valid_accuracy": score}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best = score
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            (args.output / "best.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
