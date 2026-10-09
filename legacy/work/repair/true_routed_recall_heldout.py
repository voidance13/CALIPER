#!/usr/bin/env python3
"""True per-question CALIPER retrieval routing and aggregation."""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import fasttext
import jieba
import nltk


ROOT = Path("/mnt/data/caliper/code/CALIPER")
sys.path.insert(0, str(ROOT))
ROUTER = ROOT / "query_clf/train/domain_clf.bin"
STOPWORDS = ROOT / "query_clf/stopwords.txt"
MU_DEV = Path("/tmp/musique_official_dev_full.json")
LABELS = ("hp", "ms", "tw")
DATASET_NAMES = {"hp": "hotpotqa", "ms": "musique", "tw": "2wikimultihopqa"}
MODEL_MAP = {
    "hp": {
        "embedder": ROOT / "retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft",
        "reranker": ROOT / "retrieve_fine_tuning/models/hotpotqa_bge-reranker-v2-m3-ft",
    },
    "ms": {
        "embedder": ROOT / "retrieve_fine_tuning/models/musique_firstpos_bge-embedder-ft",
        "reranker": ROOT / "retrieve_fine_tuning/models/musique_bge-reranker-v2-m3-ft_v2",
    },
    "tw": {
        "embedder": ROOT / "retrieve_fine_tuning/models/2wikimultihopqa_bge-embedder-ft",
        "reranker": ROOT / "retrieve_fine_tuning/models/2wikimultihopqa_bge-reranker-v2-m3-ft_v2",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_or_jsonl(path):
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_rows(dataset):
    if dataset == "ms":
        raw = load_json_or_jsonl(MU_DEV)
        rows = []
        for index, item in enumerate(raw):
            # Match the existing Table 6 evaluator: keep only official-dev
            # questions with at least one supporting paragraph.
            if not any(paragraph.get("is_supporting") for paragraph in item["paragraphs"]):
                continue
            ctxs = [{"title": p["title"], "sentences": p["paragraph_text"].split(". ")} for p in item["paragraphs"]]
            rows.append({"row_id": index, "question": item["question"], "ctxs": ctxs, "raw": item})
        return rows
    raw = json.loads((ROOT / f"data/{DATASET_NAMES[dataset]}/test.json").read_text(encoding="utf-8"))
    return [{"row_id": i, "question": item["question"], "ctxs": item["ctxs"], "raw": item} for i, item in enumerate(raw)]


def make_router(router_path, normalize_input=False):
    nltk.data.path.append("/home/gpt/nltk_data")
    model = fasttext.load_model(str(router_path))
    stopwords = {line.strip() for line in STOPWORDS.open(encoding="utf-8")}

    def predict(question):
        text = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
        if normalize_input:
            text = " ".join(text.lower().split())
        return model.predict(text)[0][0].removeprefix("__label__")

    return model, predict


def canonical(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def recall_for(dataset, row, retrieved):
    if dataset == "ms":
        sentence_paragraphs = {}
        supporting = set()
        for paragraph_index, paragraph in enumerate(row["raw"]["paragraphs"]):
            if paragraph.get("is_supporting"):
                supporting.add(paragraph_index)
            for sentence in nltk.sent_tokenize(paragraph["paragraph_text"]):
                sentence_paragraphs.setdefault(canonical(sentence), set()).add(paragraph_index)
        if not supporting or not sentence_paragraphs:
            return None
        retrieved = [canonical(text) for text in retrieved]
        return {
            f"R@{k}": len(set().union(*(sentence_paragraphs.get(text, set()) for text in retrieved[:k])) & supporting) / len(supporting)
            for k in (3, 5, 10)
        }

    contexts = {context["title"]: context for context in row["raw"]["ctxs"]}
    gold = [
        contexts[title]["sentences"][index]
        for title, index in row["raw"]["supporting_facts"]
        if title in contexts and index < len(contexts[title]["sentences"])
    ]
    if not gold:
        return None
    return {
        f"R@{k}": sum(item in retrieved[:k] for item in gold) / len(gold)
        for k in (3, 5, 10)
    }


def run_task(args):
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from BCERerank import BCERerank

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite only for a deliberate rerun")

    rows = load_rows(args.dataset)
    _, predict = make_router(Path(args.router), normalize_input=args.normalize_router)
    selected = []
    counts = Counter()
    for row in rows:
        predicted = predict(row["question"])
        counts[predicted] += 1
        if predicted == args.predicted_route:
            row["predicted_route"] = predicted
            selected.append(row)
    selected = selected[args.shard_index :: args.shards]
    if args.row_ids:
        wanted = {int(value) for value in args.row_ids.split(",") if value.strip()}
        selected = [row for row in selected if row["row_id"] in wanted]
    if args.limit:
        selected = selected[: args.limit]

    model = MODEL_MAP[args.predicted_route]
    emb = HuggingFaceBgeEmbeddings(
        model_name=str(model["embedder"]),
        model_kwargs={"device": "cuda:0"},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction="Represent this sentence for searching relevant passages:",
    )
    reranker = BCERerank(model=str(model["reranker"]), top_n=10, device="cuda:0", use_fp16=True)

    started = time.time()
    errors = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected, start=1):
            # Empty/whitespace-only sentence fragments are non-content and
            # crash BCERerank's tokenizer; exclude them before indexing.
            texts = [sentence for context in row["ctxs"] for sentence in context["sentences"] if sentence.strip()]
            result = {
                "dataset": args.dataset,
                "row_id": row["row_id"],
                "question": row["question"],
                "predicted_route": args.predicted_route,
                "model": {key: str(value) for key, value in model.items()},
            }
            try:
                vectorstore = FAISS.from_texts(texts=texts, embedding=emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                compressor = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
                retrieved = [document.page_content for document in compressor.get_relevant_documents(row["question"])]
                result["retrieved_text"] = retrieved
                result["recall"] = recall_for(args.dataset, row, retrieved)
                result["error"] = None
            except Exception as exc:  # keep one failed question visible without masking the run
                errors += 1
                if args.traceback:
                    traceback.print_exc()
                result["retrieved_text"] = []
                result["recall"] = None
                result["error"] = repr(exc)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 1 or index % 100 == 0 or index == len(selected):
                print(f"{args.dataset}->{args.predicted_route} {index}/{len(selected)} errors={errors} elapsed={time.time()-started:.0f}s", flush=True)
    print(json.dumps({"dataset": args.dataset, "predicted_route": args.predicted_route, "selected": len(selected), "router_counts": counts, "errors": errors, "output": str(output)}, ensure_ascii=False), flush=True)


def aggregate(args):
    files = sorted(Path(args.input_dir).glob("*.jsonl"))
    records = []
    for path in files:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    keys = [(row["dataset"], row["row_id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate dataset/row_id records")
    by_dataset = {}
    for dataset in LABELS:
        rows = [row for row in records if row["dataset"] == dataset and row["recall"] is not None]
        by_dataset[dataset] = {
            "n": len(rows),
            "R@3": round(statistics.mean(row["recall"]["R@3"] for row in rows) * 100, 2),
            "R@5": round(statistics.mean(row["recall"]["R@5"] for row in rows) * 100, 2),
            "R@10": round(statistics.mean(row["recall"]["R@10"] for row in rows) * 100, 2),
            "predicted_route_counts": dict(Counter(row["predicted_route"] for row in rows)),
        }
    correct = sum(row["dataset"] == row["predicted_route"] for row in records)
    total = len(records)
    summary = {
        "definition": "true per-question source-router retrieval using Table 2 final model mapping",
        "total_records": total,
        "router_accuracy": round(correct / total * 100, 2) if total else None,
        "errors": sum(row["error"] is not None for row in records),
        "by_dataset": by_dataset,
        "mixed_macro": {
            f"R@{k}": round(statistics.mean(by_dataset[d][f"R@{k}"] for d in LABELS), 2)
            for k in (3, 5, 10)
        },
        "files": [str(path) for path in files],
        "router_path": str(args.router),
        "router_sha256": sha256(Path(args.router)),
        "model_mapping": {key: {field: str(value) for field, value in value.items()} for key, value in MODEL_MAP.items()},
    }
    target = Path(args.output)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("run", "aggregate"), required=True)
parser.add_argument("--dataset", choices=LABELS)
parser.add_argument("--predicted-route", choices=LABELS)
parser.add_argument("--gpu", default="0")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--shards", type=int, default=1)
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--output")
parser.add_argument("--input-dir")
parser.add_argument("--router", default=str(ROUTER))
parser.add_argument("--normalize-router", action="store_true",
                    help="lowercase and collapse spaces to match normalized router training input")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--row-ids", help="comma-separated original row ids to run")
parser.add_argument("--traceback", action="store_true")
args = parser.parse_args()
if args.mode == "run":
    if not all((args.dataset, args.predicted_route, args.output)):
        parser.error("run requires --dataset, --predicted-route, and --output")
    run_task(args)
else:
    if not all((args.input_dir, args.output)):
        parser.error("aggregate requires --input-dir and --output")
    aggregate(args)
