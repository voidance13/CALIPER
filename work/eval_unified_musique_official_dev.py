"""MuSiQue official-dev paragraph recall with the unified bge-base configurations."""
import argparse
import hashlib
import json
from pathlib import Path

import faiss
import nltk
import numpy as np
from BCEmbedding import RerankerModel
from sentence_transformers import SentenceTransformer

BASE = Path("/mnt/data/caliper/code/CALIPER")
DEV = Path("/tmp/musique_full_v1.0_dev.jsonl")
MODELS = {
    "base": (
        "/mnt/data/caliper/models/bge-base-en-v1.5",
        "/mnt/data/caliper/models/bge-reranker-large",
    ),
    "ftsep_e2e": (
        str(BASE / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"),
        str(BASE / "retrieve_fine_tuning/models/unified_large_ftsep_musique"),
    ),
    "ftsep_table8": (
        str(BASE / "retrieve_fine_tuning/models/unified_bgebase_musique_fulltrain_embedder"),
        str(BASE / "retrieve_fine_tuning/models/unified_large_ftsep_musique"),
    ),
    "ftmer": (
        str(BASE / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"),
        str(BASE / "retrieve_fine_tuning/models/merged_bge-reranker-ft_v2"),
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("name", choices=MODELS)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

nltk.data.path.append("/home/gpt/nltk_data")
embedder_path, reranker_path = MODELS[args.name]
embedder = SentenceTransformer(embedder_path, device=args.device).half()
reranker = RerankerModel(model_name_or_path=reranker_path, use_fp16=True, device=args.device)
rows = [json.loads(line) for line in DEV.open(encoding="utf-8")]
sums = {3: 0.0, 5: 0.0, 10: 0.0}
evaluated = 0
skipped = 0

for index, row in enumerate(rows):
    sentences, sentence_paragraphs, supporting = [], {}, set()
    for paragraph_index, paragraph in enumerate(row["paragraphs"]):
        if paragraph["is_supporting"]:
            supporting.add(paragraph_index)
        for sentence in nltk.sent_tokenize(paragraph["paragraph_text"]):
            sentences.append(sentence)
            sentence_paragraphs.setdefault(sentence, set()).add(paragraph_index)
    if not supporting or not sentences:
        skipped += 1
        continue
    embeddings = embedder.encode(
        sentences, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    search = faiss.IndexFlatIP(embeddings.shape[1])
    search.add(embeddings)
    query = embedder.encode(
        [row["question"]], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    _, candidates = search.search(query, min(20, len(sentences)))
    candidate_texts = [sentences[i] for i in candidates[0]]
    scores = reranker.compute_score([[row["question"], text] for text in candidate_texts])
    if not isinstance(scores, list):
        scores = [scores]
    ranked = [candidate_texts[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    for k in sums:
        retrieved_paragraphs = set().union(
            *(sentence_paragraphs[text] for text in ranked[:k])
        )
        sums[k] += len(retrieved_paragraphs & supporting) / len(supporting)
    evaluated += 1
    if (index + 1) % 500 == 0:
        print(args.name, index + 1, "/", len(rows), "evaluated=", evaluated, flush=True)

result = {
    "dataset": "MuSiQue official dev",
    "metric": "mean supporting-paragraph recall from top-k retrieved sentences",
    "answer_derived_gold": False,
    "protocol_n": len(rows),
    "n": evaluated,
    "skipped": skipped,
    "method": args.name,
    "R@3": sums[3] / evaluated * 100,
    "R@5": sums[5] / evaluated * 100,
    "R@10": sums[10] / evaluated * 100,
    "embedder": embedder_path,
    "reranker": reranker_path,
    "dev_sha256": sha256(DEV),
}
target = BASE / f"table-result/table6_musique_unified_bgebase_paragraph_recall_{args.name}.json"
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False), flush=True)
