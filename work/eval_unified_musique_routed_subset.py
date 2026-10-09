"""Recompute only the MuSiQue questions routed to the MuSiQue specialist.

The retrieval path mirrors work/repair/true_routed_recall_heldout.py; only the
MuSiQue specialist model is changed to the unified bge-base configuration.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import fasttext
import jieba
import nltk

ROOT = Path("/mnt/data/caliper/code/CALIPER")
sys.path.insert(0, str(ROOT))
ROUTER = ROOT / "query_clf/text_grouped_source_router_20260917/domain_clf.bin"
STOPWORDS = ROOT / "query_clf/stopwords.txt"
DEV = Path("/tmp/musique_official_dev_full.json")
EMBEDDER = ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"
RERANKER = ROOT / "retrieve_fine_tuning/models/unified_large_ftsep_musique"


def canonical(text):
    import re
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def recall_for(row, retrieved):
    nltk.data.path.append("/home/gpt/nltk_data")
    sentence_paragraphs, supporting = {}, set()
    for paragraph_index, paragraph in enumerate(row["paragraphs"]):
        if paragraph.get("is_supporting"):
            supporting.add(paragraph_index)
        for sentence in nltk.sent_tokenize(paragraph["paragraph_text"]):
            sentence_paragraphs.setdefault(canonical(sentence), set()).add(paragraph_index)
    normalized = [canonical(text) for text in retrieved]
    return {
        f"R@{k}": len(set().union(*(sentence_paragraphs.get(text, set()) for text in normalized[:k])) & supporting) / len(supporting)
        for k in (3, 5, 10)
    }


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--output", required=True)
args = parser.parse_args()

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from BCERerank import BCERerank

nltk.data.path.append("/home/gpt/nltk_data")
router = fasttext.load_model(str(ROUTER))
stopwords = {line.strip() for line in STOPWORDS.open(encoding="utf-8")}


def predict(question):
    text = " ".join(token for token in jieba.lcut(question) if len(token) > 1 and token not in stopwords)
    text = " ".join(text.lower().split())
    return router.predict(text)[0][0].removeprefix("__label__")


raw = json.loads(DEV.read_text(encoding="utf-8"))
selected = []
for row_id, row in enumerate(raw):
    if not any(paragraph.get("is_supporting") for paragraph in row["paragraphs"]):
        continue
    if predict(row["question"]) == "ms":
        selected.append((row_id, row))
selected = selected[args.shard :: args.nshards]

emb = HuggingFaceBgeEmbeddings(
    model_name=str(EMBEDDER),
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="Represent this sentence for searching relevant passages:",
)
reranker = BCERerank(model=str(RERANKER), top_n=10, device="cuda:0", use_fp16=True)
started = time.time()
errors = 0
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for index, (row_id, row) in enumerate(selected, start=1):
        texts = [
            sentence
            for paragraph in row["paragraphs"]
            for sentence in nltk.sent_tokenize(paragraph["paragraph_text"])
            if sentence.strip()
        ]
        result = {
            "dataset": "ms",
            "row_id": row_id,
            "question": row["question"],
            "predicted_route": "ms",
            "model": {"embedder": str(EMBEDDER), "reranker": str(RERANKER)},
        }
        try:
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=emb,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
            compressor = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
            retrieved = [document.page_content for document in compressor.get_relevant_documents(row["question"])]
            result["retrieved_text"] = retrieved
            result["recall"] = recall_for(row, retrieved)
            result["error"] = None
        except Exception as exc:
            errors += 1
            result["retrieved_text"] = []
            result["recall"] = None
            result["error"] = repr(exc)
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        if index == 1 or index % 100 == 0 or index == len(selected):
            print(f"ms->ms {index}/{len(selected)} errors={errors} elapsed={time.time()-started:.0f}s", flush=True)
print(json.dumps({"selected": len(selected), "errors": errors, "output": str(output)}, ensure_ascii=False), flush=True)
