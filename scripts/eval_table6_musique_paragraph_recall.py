"""MuSiQue official-dev supporting-paragraph recall; no answer-derived sentence gold."""
import json, sys
from pathlib import Path

import faiss, nltk, numpy as np
from BCEmbedding import RerankerModel
from sentence_transformers import SentenceTransformer

BASE = Path("/mnt/data/caliper/code/CALIPER")
NAME, DEVICE = sys.argv[1], sys.argv[2]
nltk.data.path.append("/home/gpt/nltk_data")
MODELS = {
    "base": ("/mnt/data/caliper/models/bge_large_local", "/mnt/data/caliper/models/bge-reranker-large"),
    "ftsep": (str(BASE / "retrieve_fine_tuning/models/musique_bge-m3-embedder-ft"), str(BASE / "retrieve_fine_tuning/models/musique_bge-reranker-v2-m3-ft_v2")),
    "ftmer": (str(BASE / "retrieve_fine_tuning/models/musique_firstpos_bge-embedder-ft"), str(BASE / "retrieve_fine_tuning/models/merged_bge-reranker-ft_v2")),
}
embedder = SentenceTransformer(MODELS[NAME][0], device=DEVICE).half()
reranker = RerankerModel(model_name_or_path=MODELS[NAME][1], use_fp16=True, device=DEVICE)
rows = [json.loads(line) for line in open("/tmp/musique_full_v1.0_dev.jsonl", encoding="utf-8")]
sums, evaluated = {3: 0.0, 5: 0.0, 10: 0.0}, 0

for index, row in enumerate(rows):
    sentences, sentence_paragraphs, supporting = [], {}, set()
    for paragraph_index, paragraph in enumerate(row["paragraphs"]):
        if paragraph["is_supporting"]: supporting.add(paragraph_index)
        for sentence in nltk.sent_tokenize(paragraph["paragraph_text"]):
            sentences.append(sentence); sentence_paragraphs.setdefault(sentence, set()).add(paragraph_index)
    if not supporting or not sentences: continue
    embeddings = embedder.encode(sentences, batch_size=32, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    search = faiss.IndexFlatIP(embeddings.shape[1]); search.add(embeddings)
    query = embedder.encode([row["question"]], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    _, candidates = search.search(query, min(20, len(sentences)))
    candidate_texts = [sentences[i] for i in candidates[0]]
    scores = reranker.compute_score([[row["question"], text] for text in candidate_texts])
    ranked = [candidate_texts[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
    for k in sums:
        retrieved_paragraphs = set().union(*(sentence_paragraphs[text] for text in ranked[:k]))
        sums[k] += len(retrieved_paragraphs & supporting) / len(supporting)
    evaluated += 1
    if (index + 1) % 500 == 0: print(NAME, index + 1, "/", len(rows), flush=True)

result = {"dataset": "MuSiQue official dev", "metric": "mean supporting-paragraph recall from top-k retrieved sentences",
          "answer_derived_gold": False, "method": NAME, "n": evaluated,
          "R@3": sums[3] / evaluated * 100, "R@5": sums[5] / evaluated * 100, "R@10": sums[10] / evaluated * 100,
          "embedder": MODELS[NAME][0], "reranker": MODELS[NAME][1]}
json.dump(result, open(BASE / f"table-result/table6_musique_paragraph_recall_{NAME}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False), flush=True)
