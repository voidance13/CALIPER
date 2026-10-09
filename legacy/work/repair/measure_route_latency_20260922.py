"""Versioned retrieval-path timing only. No generation, API calls or frozen writes."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path('/mnt/data/caliper/code/CALIPER')
DS = ('hotpotqa', 'musique', '2wikimultihopqa')
ALIASES = dict(zip(DS, ('hp', 'ms', 'tw')))
MODELDIR = ROOT / 'retrieve_fine_tuning/models'
FIXED = {
    'hotpotqa': ('hotpotqa_bge-embedder-ft', 'hotpotqa_bge-reranker-v2-m3-ft'),
    'musique': ('musique_firstpos_bge-embedder-ft', 'musique_bge-reranker-v2-m3-ft_v2'),
    '2wikimultihopqa': ('2wikimultihopqa_bge-embedder-ft', '2wikimultihopqa_bge-reranker-v2-m3-ft_v2'),
}


def resolve_route(dataset, index, live, frozen):
    if live == frozen:
        return frozen, False
    if dataset == 'musique' and index in (72, 86) and (live, frozen) == ('hp', 'ms'):
        return frozen, True
    raise ValueError(f'Unapproved route mismatch: {dataset} {index} {live} {frozen}')


def model_for(records, question, predicted_route):
    record = records[question]
    if record['predicted_route'] != predicted_route:
        raise ValueError('Live route differs from frozen route')
    return record['model']['embedder'], record['model']['reranker']


def frozen_texts(question, row_id, rows, official_dev):
    row = rows[row_id]
    if row['question'] != question:
        raise ValueError('Frozen row ID does not match question')
    if official_dev:
        return [s for p in row['paragraphs'] for s in p['paragraph_text'].split('. ') if s.strip()]
    return [s for c in row['ctxs'] for s in c['sentences'] if s.strip()]


def align_rows(questions, rows):
    by_question = defaultdict(list)
    for r in rows:
        by_question[r['question']].append(r)
    used = defaultdict(int)
    result = []
    for q in questions:
        pos = used[q]
        if pos >= len(by_question[q]):
            raise ValueError(f'Missing question occurrence {pos}: {q}')
        result.append(by_question[q][pos])
        used[q] += 1
    return result


def contexts(dataset, ft, base):
    if dataset != 'musique':
        return ft[:5]
    result, seen = [], set()
    for s in ft[:5] + base[:3]:
        key = ' '.join(s.strip().lower().split())
        if key and key not in seen:
            result.append(s)
            seen.add(key)
    return result


def composable_overhead(record, generation_includes_classifier):
    return record['total_seconds'] - (record['classifier_seconds'] if generation_includes_classifier else 0)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT))
    import fasttext
    import jieba
    import nltk
    import numpy as np
    import torch
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain.retrievers import ContextualCompressionRetriever
    from BCERerank import BCERerank
    torch.set_num_threads(2)
    nltk.data.path.append('/home/gpt/nltk_data')
    jieba.initialize()
    dataset = args.dataset
    dest = args.output / dataset
    dest.mkdir(parents=True, exist_ok=False)
    timing_path = ROOT / f'table-result/table4_timing_{dataset}_seed42.json'
    test_path = ROOT / f'data/{dataset}/test.json'
    timing = read(timing_path)['rows']
    assert len(timing) == 200
    tests = read(test_path)
    test_by_question = {r['question']: r for r in tests}
    sample = align_rows([r['question'] for r in timing], tests)[:args.limit]
    official_path = Path('/tmp/musique_official_dev_full.json')
    official = read(official_path) if dataset == 'musique' else None
    snapshot = ROOT / ('router_e2e_runs/text_grouped_20260920_unified_bgebase/routed_records.jsonl'
                       if dataset == 'musique' else 'router_e2e_runs/text_grouped_20260918/routed_records.jsonl')
    routed = {}
    with snapshot.open() as f:
        for line in f:
            r = json.loads(line)
            if r['dataset'] in (dataset, ALIASES[dataset]):
                routed[r['question']] = r  # mirrors frozen E2E question-key lookup
    assert all(r['question'] in routed for r in sample), 'missing routed evidence'
    source_path = ROOT / 'query_clf/text_grouped_source_router_20260917/domain_clf.bin'
    classifier_path = ROOT / 'query_clf/train_new/9b_clf.bin'
    source = fasttext.load_model(str(source_path))
    classifier = fasttext.load_model(str(classifier_path))
    stopwords_path = ROOT / 'query_clf/stopwords.txt'
    stopwords = set(stopwords_path.read_text().splitlines())
    def tokens(q):
        return ' '.join(s for s in jieba.lcut(q) if len(s) > 1 and s not in stopwords)
    def route(q):
        return source.predict(' '.join(tokens(q).lower().split()))[0][0].removeprefix('__label__')
    replay = []
    for index, r in enumerate(sample):
        live = route(r['question'])
        selected, overridden = resolve_route(dataset, index, live, routed[r['question']]['predicted_route'])
        if overridden:
            replay.append({'index': index, 'question': r['question'], 'live': live, 'frozen': selected})
    fixed_paths = tuple(str(MODELDIR / n) for n in FIXED[dataset])
    maps = {}
    for r in routed.values():
        if r['question'] not in {s['question'] for s in sample}:
            continue
        pair = (r['model']['embedder'], r['model']['reranker'])
        key = r['question']
        maps[key] = pair
    base_paths = ('/mnt/data/caliper/models/bge-base-en-v1.5', '/mnt/data/caliper/models/bge-reranker-large')
    pairs = set(maps.values()) | {fixed_paths}
    evidence = [timing_path, test_path, snapshot, source_path, classifier_path, stopwords_path, Path(__file__)]
    if official is not None:
        evidence.append(official_path)
    for pair in pairs | ({base_paths} if dataset == 'musique' else set()):
        for directory in pair:
            p = Path(directory)
            weights = list(p.glob('*.safetensors')) + list(p.glob('pytorch_model*.bin'))
            assert weights, f'no weights {p}'
            evidence += weights + [p / 'config.json']
    manifest = {'status': 'RUNNING_NOT_FROZEN', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'dataset': dataset, 'n': len(sample), 'warmup': min(20, len(sample)), 'workers': 4,
        'authorized_frozen_route_replay': replay,
        'gpu': args.gpu, 'gpu_name': torch.cuda.get_device_name(0), 'model_map': maps,
        'fixed_reference_models': fixed_paths, 'base_models': base_paths if dataset == 'musique' else None,
        'scope': 'Table2/4 routed model snapshots; old Table5 fixed reference; retrieval-only; no cloud calls',
        'limitations': ['New contexts are not inputs to old frozen generation. No integrated E2E result.',
                       'Model loading excluded; four concurrent requests on one GPU; no retrieval cache.',
                       'Classifier time is separately reported; do not add it again to actual clf generation timing.'],
        'sha256': {str(p): digest(p) for p in dict.fromkeys(evidence)}}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('MANIFEST_READY', dataset, flush=True)
    pool = queue.Queue()
    for i in range(4):
        models = {}
        for pair in pairs:
            emb = HuggingFaceBgeEmbeddings(model_name=pair[0], model_kwargs={'device': 'cuda:0'},
                encode_kwargs={'normalize_embeddings': True}, query_instruction='Represent this sentence for searching relevant passages:')
            rer = BCERerank(model=pair[1], top_n=10, device='cuda:0', use_fp16=True)
            models[pair] = (emb, rer)
        base = None
        if dataset == 'musique':
            from sentence_transformers import SentenceTransformer
            from FlagEmbedding import FlagReranker
            base = (SentenceTransformer(base_paths[0], device='cuda:0').half(),
                    FlagReranker(base_paths[1], use_fp16=True, device='cuda:0'))
        pool.put((models, base))
        print('REPLICA_READY', dataset, i, flush=True)
    def retrieve(q, texts, model):
        emb, rer = model
        index = FAISS.from_texts(texts=texts, embedding=emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        comp = ContextualCompressionRetriever(base_compressor=rer,
            base_retriever=index.as_retriever(search_type='similarity', search_kwargs={'k': 20}))
        return [d.page_content for d in comp.get_relevant_documents(q)]
    def retrieve_base(q, texts, model):
        import faiss
        emb, rer = model
        query_vec = emb.encode([q], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        vec = emb.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        index = faiss.IndexFlatIP(vec.shape[1]); index.add(vec)
        _, ids = index.search(query_vec, min(20, len(texts)))
        cand = [texts[i] for i in ids[0]]
        scores = np.atleast_1d(rer.compute_score([[q, s] for s in cand]))
        return [cand[i] for i in sorted(range(len(scores)), key=lambda i: -float(scores[i]))[:10]]
    old_cache_path = ROOT / f'data/{dataset}/test_retrieved_embedderft_rerankerft_v2.json'
    old_cache = {r['question']: r for r in read(old_cache_path)}
    def texts_of(row):
        return row.get('retrieved_text') or [s for c in row['ctxs'] for s in c['sentences']]
    def one(task):
        idx, row, condition = task
        models, base_model = pool.get()
        q = row['question']
        try:
            start = time.perf_counter()
            clf_s, branch, score = 0., None, None
            if condition == 'routed_clf':
                t = time.perf_counter()
                labels, probs = classifier.predict(tokens(q), k=2)
                score = next((float(p) for label, p in zip(labels, probs) if label == '__label__easy'), 0.)
                branch = 'local' if score >= .997 else 'collaborative'
                clf_s = time.perf_counter() - t
            t = time.perf_counter()
            live = route(q) if condition != 'fixed_reference' else None
            route_s = time.perf_counter() - t if live is not None else 0.
            predicted, overridden = resolve_route(dataset, idx, live, routed[q]['predicted_route']) if live is not None else (None, False)
            pair = model_for(routed, q, predicted) if predicted is not None else fixed_paths
            t = time.perf_counter()
            texts = [s for c in row['ctxs'] for s in c['sentences'] if s.strip()]
            if predicted is not None:
                # Assembly used test row IDs only for newly replaced ms->ms;
                # retained MuSiQue branches retain official-dev row IDs.
                is_old_ms = dataset == 'musique' and 'unified_bgebase_musique_firstpos_embedder' not in pair[0]
                texts = frozen_texts(q, routed[q]['row_id'], official if is_old_ms else tests, is_old_ms)
            ft = retrieve(q, texts, models[pair])
            ft_s = time.perf_counter() - t
            t = time.perf_counter()
            base_text = retrieve_base(q, [s for c in test_by_question[q]['ctxs'] for s in c['sentences']], base_model) if predicted is not None and dataset == 'musique' else []
            base_s = time.perf_counter() - t if base_text else 0.
            t = time.perf_counter()
            context = contexts(dataset, ft, base_text) if predicted is not None else ft[:5]
            merge_s = time.perf_counter() - t
            total = time.perf_counter() - start
            old = texts_of(old_cache[q])[:5]
            return {'index': idx, 'question': q, 'condition': condition, 'source_route': predicted,
                'live_source_route': live, 'frozen_route_replayed': overridden,
                'model': {'embedder': pair[0], 'reranker': pair[1]},
                'classifier_branch': branch, 'easy_score': score, 'classifier_seconds': clf_s,
                'source_router_seconds': route_s, 'ft_seconds': ft_s, 'base_seconds': base_s,
                'merge_seconds': merge_s, 'total_seconds': total, 'context': context,
                'matches_old_generation_context': context == old,
                'matches_frozen_routed_top5': ft[:5] == routed[q]['retrieved_text'][:5] if predicted else None,
                'error': None}
        finally:
            pool.put((models, base_model))
    all_records = []
    for condition in ('fixed_reference', 'routed_core', 'routed_clf'):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(one, [(i, r, condition) for i, r in enumerate(sample[:20])]))
            print('WARMUP_DONE', dataset, condition, flush=True)
            with (dest / f'{condition}.jsonl').open('x') as out:
                for record in executor.map(one, [(i, r, condition) for i, r in enumerate(sample)]):
                    out.write(json.dumps(record, ensure_ascii=False) + '\n'); out.flush()
                    all_records.append(record)
                    if (record['index'] + 1) % 20 == 0:
                        print('PROGRESS', dataset, condition, record['index'] + 1, flush=True)
    summary = {'status': 'COMPLETED_CANDIDATE_NOT_FROZEN', 'dataset': dataset, 'conditions': {}}
    for condition in ('fixed_reference', 'routed_core', 'routed_clf'):
        records = [r for r in all_records if r['condition'] == condition]
        values = [r['total_seconds'] for r in records]
        summary['conditions'][condition] = {'n': len(values), 'mean': float(np.mean(values)),
            'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
            'old_generation_context_matches': sum(r['matches_old_generation_context'] for r in records),
            'frozen_routed_top5_matches': sum(r['matches_frozen_routed_top5'] is True for r in records)}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=200, choices=range(1, 201))
    main(parser.parse_args())
