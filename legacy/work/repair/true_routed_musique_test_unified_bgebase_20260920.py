#!/usr/bin/env python3
"""Build versioned MuSiQue test routed-retrieval records for the E2E rerun.

Only the MuSiQue specialist branch changes to the unified bge-base setup;
the source-router, question rows, and retrieval protocol are frozen.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import fasttext
import jieba

ROOT = Path("/mnt/data/caliper/code/CALIPER")
sys.path.insert(0, str(ROOT))
STOPWORDS = ROOT / "query_clf/stopwords.txt"
ROUTER = ROOT / "query_clf/text_grouped_source_router_20260917/domain_clf.bin"
MODELS = {
    "hp": {
        "embedder": ROOT / "retrieve_fine_tuning/models/hotpotqa_bge-embedder-ft",
        "reranker": ROOT / "retrieve_fine_tuning/models/hotpotqa_bge-reranker-v2-m3-ft",
    },
    "ms": {
        "embedder": ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder",
        "reranker": ROOT / "retrieve_fine_tuning/models/unified_large_ftsep_musique",
    },
    "tw": {
        "embedder": ROOT / "retrieve_fine_tuning/models/2wikimultihopqa_bge-embedder-ft",
        "reranker": ROOT / "retrieve_fine_tuning/models/2wikimultihopqa_bge-reranker-v2-m3-ft_v2",
    },
}


def load_router():
    nltk_path = "/home/gpt/nltk_data"
    # Kept identical to the frozen text-grouped router inference path.
    import nltk
    nltk.data.path.append(nltk_path)
    model = fasttext.load_model(str(ROUTER))
    stopwords = {x.strip() for x in STOPWORDS.open(encoding="utf-8") if x.strip()}

    def predict(question):
        text = " ".join(x for x in jieba.lcut(question) if len(x) > 1 and x not in stopwords)
        return model.predict(text)[0][0].removeprefix("__label__")

    return predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predicted-route", choices=("hp", "ms", "tw"), required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}")

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from BCERerank import BCERerank

    rows = json.loads((ROOT / "data/musique/test.json").read_text(encoding="utf-8"))
    predict = load_router()
    routed = [(i, row, predict(row["question"])) for i, row in enumerate(rows)]
    counts = Counter(route for _, _, route in routed)
    selected = [(i, row, route) for i, row, route in routed if route == args.predicted_route]
    selected = selected[args.shard_index :: args.shards]
    model = MODELS[args.predicted_route]
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
        for n, (row_id, row, route) in enumerate(selected, 1):
            record = {
                "dataset": "ms",
                "row_id": row_id,
                "question": row["question"],
                "predicted_route": route,
                "model": {key: str(value) for key, value in model.items()},
            }
            try:
                texts = [s for ctx in row["ctxs"] for s in ctx.get("sentences", []) if s.strip()]
                vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=emb,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                compressor = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
                record["retrieved_text"] = [doc.page_content for doc in compressor.get_relevant_documents(row["question"])]
                record["error"] = None
            except Exception as exc:
                errors += 1
                record["retrieved_text"] = []
                record["error"] = repr(exc)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if n == 1 or n % 100 == 0 or n == len(selected):
                print(f"ms->{route} {n}/{len(selected)} errors={errors} elapsed={time.time()-started:.0f}s", flush=True)
    print(json.dumps({"route": args.predicted_route, "selected": len(selected), "router_counts": counts, "errors": errors, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
