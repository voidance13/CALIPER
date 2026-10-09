import argparse
import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from BCERerank import BCERerank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--embed_model", default="/mnt/data/caliper/models/bge-base-en-v1.5", type=str)
    parser.add_argument("--rerank_model", default="/mnt/data/caliper/models/bge-reranker-large", type=str)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    retrieved_path = f"data/{args.dataset}/test_retrieved.json"
    if os.path.exists(retrieved_path):
        print(f"cache already exists: {retrieved_path}")
        return

    with open(f"data/{args.dataset}/test.json", encoding="utf-8") as f:
        items = json.load(f)
    print(f"dataset={args.dataset} items={len(items)} device={args.device} workers={args.workers}")

    def create_models():
        emb = HuggingFaceBgeEmbeddings(
                model_name=args.embed_model,
                model_kwargs={"device": args.device},
                encode_kwargs={"normalize_embeddings": True},
                query_instruction="Represent this sentence for searching relevant passages:",
            )
        rer = BCERerank(
                model=args.rerank_model,
                top_n=10,
                device=args.device,
                use_fp16=True,
            )
        return emb, rer

    # Create all model pairs sequentially in the main thread before starting
    # workers; concurrent model loading triggers transformers meta-tensor bugs.
    model_queue = queue.Queue()
    for _ in range(args.workers):
        model_queue.put(create_models())

    def retrieve_one(item):
        emb, rer = model_queue.get()
        texts = [s for ctx in item["ctxs"] for s in ctx["sentences"]]
        try:
            faiss = FAISS.from_texts(
                texts=texts, embedding=emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=rer,
                base_retriever=faiss.as_retriever(search_type="similarity", search_kwargs={"k": 20}),
            )
            item["retrieved_text"] = [d.page_content for d in retriever.get_relevant_documents(item["question"])]
            return item
        finally:
            model_queue.put((emb, rer))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for r in tqdm(executor.map(retrieve_one, items), total=len(items)):
            results.append(r)

    with open(retrieved_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"wrote {retrieved_path} ({len(results)} items)")


if __name__ == "__main__":
    main()
