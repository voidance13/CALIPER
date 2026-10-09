"""Rebuild MuSiQue test retrieval caches with the unified bge-base embedder setup."""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer


ROOT = Path("/mnt/data/caliper/code/CALIPER")
TEST = ROOT / "data/musique/test.json"
MODELS = {
    "base": {
        "embedder": "/mnt/data/caliper/models/bge-base-en-v1.5",
        "reranker": "/mnt/data/caliper/models/bge-reranker-large",
        "prefix": "test_retrieved_unified_bgebase_base",
    },
    "ftsep_e2e": {
        "embedder": str(ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"),
        "reranker": str(ROOT / "retrieve_fine_tuning/models/unified_large_ftsep_musique"),
        "prefix": "test_retrieved_unified_bgebase_ftsep_e2e",
    },
    "ftsep_table8": {
        "embedder": str(ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_fulltrain_embedder"),
        "reranker": str(ROOT / "retrieve_fine_tuning/models/unified_large_ftsep_musique"),
        "prefix": "test_retrieved_unified_bgebase_ftsep_table8",
    },
    "ftmer": {
        "embedder": str(ROOT / "retrieve_fine_tuning/models/unified_bgebase_musique_firstpos_embedder"),
        "reranker": str(ROOT / "retrieve_fine_tuning/models/merged_bge-reranker-ft_v2"),
        "prefix": "test_retrieved_unified_bgebase_ftmer",
    },
}


def retrieve(args):
    spec = MODELS[args.variant]
    rows = json.loads(TEST.read_text(encoding="utf-8"))
    selected = list(enumerate(rows))[args.shard :: args.nshards]
    device = f"cuda:{args.gpu}"
    embedder = SentenceTransformer(spec["embedder"], device=device).half()
    reranker = FlagReranker(spec["reranker"], use_fp16=True, device=device)
    out = []
    start = time.time()
    for row_index, row in selected:
        ctxs = [{"title": c["title"], "sentences": c["sentences"]} for c in row["ctxs"]]
        sentences = [sentence for ctx in ctxs for sentence in ctx["sentences"]]
        query_vec = embedder.encode([row["question"]], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        passage_vec = embedder.encode(sentences, batch_size=256, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        index = faiss.IndexFlatIP(passage_vec.shape[1])
        index.add(passage_vec)
        _, candidate_ids = index.search(query_vec, min(20, len(sentences)))
        candidates = [sentences[i] for i in candidate_ids[0]]
        scores = reranker.compute_score([[row["question"], candidate] for candidate in candidates])
        if np.isscalar(scores):
            scores = [scores]
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        out.append({"row_index": row_index, "question": row["question"], "ctxs": ctxs, "retrieved_text": [candidates[i] for i in order[:10]]})
        if len(out) % 200 == 0:
            print(f"{args.variant} shard={args.shard} {len(out)}/{len(selected)} {time.time() - start:.0f}s", flush=True)
    dest = ROOT / f"data/musique/{spec['prefix']}.json.s{args.shard}"
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {args.variant} shard={args.shard} rows={len(out)} -> {dest}", flush=True)


def merge(args):
    spec = MODELS[args.variant]
    parts = []
    for shard in range(args.nshards):
        path = ROOT / f"data/musique/{spec['prefix']}.json.s{shard}"
        parts.extend(json.loads(path.read_text(encoding="utf-8")))
    parts.sort(key=lambda row: row["row_index"])
    final = [{k: value for k, value in row.items() if k != "row_index"} for row in parts]
    dest = ROOT / f"data/musique/{spec['prefix']}.json"
    dest.write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")
    print(f"MERGED {args.variant} rows={len(final)} -> {dest}", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=sorted(MODELS))
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--nshards", type=int, default=1)
parser.add_argument("--merge", action="store_true")
args = parser.parse_args()
merge(args) if args.merge else retrieve(args)
