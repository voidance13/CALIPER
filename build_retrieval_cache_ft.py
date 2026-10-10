"""用 ft embedder+reranker 构建检索缓存（输出 test_retrieved_embedderft_rerankerft.json）。

用法:
  python build_retrieval_cache_ft.py --dataset hotpotqa --device cuda:0
  python build_retrieval_cache_ft.py recall --dataset hotpotqa
"""
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
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()

    ft_root = os.getenv("CALIPER_FT_MODEL_ROOT", "retrieve_fine_tuning/models")
    embed_model = f"{ft_root}/{args.dataset}_bge-embedder-ft"
    rerank_model = f"{ft_root}/{args.dataset}_bge-reranker-ft"
    retrieved_path = f"data/{args.dataset}/test_retrieved_embedderft_rerankerft.json"
    if os.path.exists(retrieved_path):
        print(f"cache already exists: {retrieved_path}")
        return

    with open(f"data/{args.dataset}/test.json", encoding="utf-8") as f:
        items = json.load(f)
    print(f"dataset={args.dataset} items={len(items)} device={args.device} workers={args.workers}")
    print(f"embed={embed_model} rerank={rerank_model}")

    def create_models():
        emb = HuggingFaceBgeEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": args.device},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages:",
        )
        rer = BCERerank(model=rerank_model, top_n=10, device=args.device, use_fp16=True)
        return emb, rer

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


def recall_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    args = parser.parse_args()
    samples = json.load(open(f"data/{args.dataset}/test_retrieved_embedderft_rerankerft.json"))

    hit3_total = hit5_total = hit10_total = 0.0
    n = 0
    for sample in samples:
        ctxs = [c["sentences"] for c in sample["ctxs"]]
        for sf in sample["supporting_facts"]:
            title, si = sf
            ci = None
            for i, c in enumerate(sample["ctxs"]):
                if c["title"] == title:
                    ci = i
                    break
            if ci is None or si >= len(ctxs[ci]):
                continue
            sent = ctxs[ci][si]
            rt = sample["retrieved_text"]
            n += 1
            if sent in rt[:10]:
                hit10_total += 1
            if sent in rt[:5]:
                hit5_total += 1
            if sent in rt[:3]:
                hit3_total += 1
    if n == 0:
        print("no supporting facts matched")
        return
    print(f"{args.dataset}: R@3={hit3_total/n:.4f} R@5={hit5_total/n:.4f} R@10={hit10_total/n:.4f} (n={n})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "recall":
        sys.argv.pop(1)
        recall_main()
    else:
        main()
